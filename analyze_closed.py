"""Closed-set aggregation, segment-level paired statistics and closed-set figures."""
import json, glob, numpy as np, pandas as pd, os
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aegis_data import load_all, session_frames, CLASSES
from aegis_metrics import softmax, closed_metrics

W = '/home/claude/work'; FIG = '/home/claude/figs'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
MODELS = ['RandomForest', 'LightGBM', 'MLP', 'E-GraphSAGE', 'HGNN', 'GRU-Seq', 'Transformer-Seq', 'SSM-Seq', 'AegisFlow']
df = load_all(); _, _, te = session_frames(df, 'B', [0, 6, 7.5, 13])
t = te.Stime.values; y = te.y.values
seg = ((t - t.min()) // 1800).astype(int); seg = np.minimum(seg, 8)


def load_pred(m, s):
    f = f'{W}/pred_B_{m}_s{s}.npz'
    if not os.path.exists(f):
        return None
    return np.load(f)['lt'].astype(np.float32)


res = {}; unit_f1 = {}
for m in MODELS:
    rows = []
    for s in [0, 1, 2]:
        lt = load_pred(m, s)
        if lt is None:
            continue
        r = closed_metrics(lt, y, 10)
        j = f'{W}/runs/{m}_closed_s{s}.json'
        if os.path.exists(j):
            jj = json.load(open(j)); r['conf'] = jj['conf_eps0.01']; r['params'] = jj['params']
            r['train_s'] = jj['train_s']; r['infer_us'] = jj['infer_us_per_flow']; r['cov'] = jj['coverage_per_class']
            r['conf_all'] = {k: jj[k] for k in jj if k.startswith('conf_eps')}
        else:   # tree models: conformal layer on -log probabilities, times from res_trees_B.json
            from aegis_train import mondrian_pvalues, decide
            lc = np.load(f'{W}/pred_B_{m}_s{s}.npz')['lc'].astype(np.float32); yc = np.load(f'{W}/y_B_ca.npy') if os.path.exists(f'{W}/y_B_ca.npy') else None
            if yc is None:
                _, ca_, _ = session_frames(df, 'B', [0, 6, 7.5, 13]); yc = ca_.y.values; np.save(f'{W}/y_B_ca.npy', yc)
            Sc, St = -lc, -lt; P = mondrian_pvalues(Sc, yc, St, 10)
            r['conf_all'] = {}
            for e in [0.001, 0.005, 0.01, 0.05]:
                al, nv, pr, fs = decide(P, St, 0, e, 0.05)
                r['conf_all'][f'conf_eps{e}'] = dict(far=float(al[y == 0].mean()), det=float(al[y != 0].mean()), macro_f1=float(f1_score(y, pr, average='macro')),
                                                    set_size=float(fs[al].sum(1).mean()), novel_known=float(nv[y != 0].mean()))
            r['conf'] = r['conf_all']['conf_eps0.01']; r['cov'] = [float((P[y == c, c] > 0.05).mean()) for c in range(10)]
            T = json.load(open(f'{W}/res_trees_B.json'))[m][s]; r['train_s'] = T['train_s']; r['infer_us'] = 1e6 * T['infer_s'] / len(y); r['params'] = None
        rows.append(r)
        pred = lt.argmax(1)
        unit_f1.setdefault(m, []).extend([f1_score(y[seg == k], pred[seg == k], average='macro', labels=list(range(10)), zero_division=0) for k in range(9)])
    if rows:
        res[m] = rows
summary = {}
for m, rows in res.items():
    d = {}
    for k in ['acc', 'macro_f1', 'bal_acc', 'mcc', 'det_rate', 'far', 'ece', 'brier', 'nll']:
        v = np.array([r[k] for r in rows]); d[k] = (float(v.mean()), float(v.std()))
    d['per_class_f1'] = np.mean([r['per_class_f1'] for r in rows], 0).tolist()
    if 'conf' in rows[0]:
        d['conf_far'] = (float(np.mean([r['conf']['far'] for r in rows])), float(np.std([r['conf']['far'] for r in rows])))
        d['conf_det'] = (float(np.mean([r['conf']['det'] for r in rows])), float(np.std([r['conf']['det'] for r in rows])))
        d['conf_f1'] = (float(np.mean([r['conf']['macro_f1'] for r in rows])), float(np.std([r['conf']['macro_f1'] for r in rows])))
        d['set_size'] = float(np.mean([r['conf']['set_size'] for r in rows]))
        d['params'] = rows[0]['params']; d['train_s'] = float(np.mean([r['train_s'] for r in rows])); d['infer_us'] = float(np.mean([r['infer_us'] for r in rows]))
        d['cov'] = np.mean([r['cov'] for r in rows], 0).tolist()
        d['conf_all'] = {k: {kk: float(np.mean([r['conf_all'][k][kk] for r in rows])) for kk in rows[0]['conf_all'][k]} for k in rows[0]['conf_all']}
    else:
        d['train_s'] = float(np.mean([r['train_s'] for r in rows])); d['infer_us'] = float(np.mean([r['infer_s'] for r in rows])) * 1e6 / len(y)
    d['n_seeds'] = len(rows)
    summary[m] = d
# paired statistics
stats = {}
ref = np.array(unit_f1.get('AegisFlow', []))
pv = {}
for m in MODELS:
    if m == 'AegisFlow' or m not in unit_f1 or len(unit_f1[m]) != len(ref):
        continue
    a = np.array(unit_f1[m]); dlt = ref - a
    w = wilcoxon(ref, a, zero_method='wilcox')
    pos = dlt[dlt != 0]; rk = pd.Series(np.abs(pos)).rank().values
    rbc = (rk[pos > 0].sum() - rk[pos < 0].sum()) / rk.sum() if len(rk) else 0.0
    stats[m] = dict(mean_diff=float(dlt.mean()), median_diff=float(np.median(dlt)), p=float(w.pvalue), rbc=float(rbc), n=len(dlt), wins=int((dlt > 0).sum()))
    pv[m] = w.pvalue
order = sorted(pv, key=pv.get); kk = len(order); run = 0
for i, m in enumerate(order):
    run = max(run, min(1.0, pv[m] * (kk - i))); stats[m]['p_holm'] = float(run)
avail = [m for m in MODELS if m in unit_f1 and len(unit_f1[m]) == len(ref)]
if len(avail) >= 3:
    fr = friedmanchisquare(*[unit_f1[m] for m in avail]); stats['friedman'] = dict(chi2=float(fr.statistic), p=float(fr.pvalue), k=len(avail), n=len(ref))
    ranks = pd.DataFrame({m: unit_f1[m] for m in avail}).rank(axis=1, ascending=False).mean().to_dict(); stats['mean_rank'] = ranks
json.dump(dict(summary=summary, stats=stats), open(f'{W}/res_closed.json', 'w'), indent=1)
for m in summary:
    print(m, summary[m]['n_seeds'], np.round(summary[m]['macro_f1'], 4), np.round(summary[m]['far'], 4), summary[m].get('conf_far'))
print(json.dumps(stats, indent=0)[:1500])

# ---------------- Fig: per-family F1 ----------------
show = [m for m in ['LightGBM', 'MLP', 'E-GraphSAGE', 'HGNN', 'SSM-Seq', 'AegisFlow'] if m in summary]
STY = {'LightGBM': ('white', ''), 'MLP': ('white', '////'), 'E-GraphSAGE': ('white', '....'), 'HGNN': ('white', 'xxxx'),
       'SSM-Seq': ('#bdbdbd', ''), 'AegisFlow': ('#3a3a3a', '')}
fig, ax = plt.subplots(figsize=(7.0, 2.35))
xs = np.arange(10); wbar = 0.8 / len(show)
for i, m in enumerate(show):
    ax.bar(xs - 0.4 + wbar * (i + 0.5), summary[m]['per_class_f1'], wbar, label=m, facecolor=STY[m][0], edgecolor='black', hatch=STY[m][1], lw=0.5)
ax.set_xticks(xs); ax.set_xticklabels(['Normal', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Recon.', 'Shellcode', 'Worms'])
ax.set_ylabel('F1 (mean of seeds)'); ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(ncol=len(show), loc='upper center', frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, 1.02))
for s_ in ['top', 'right']:
    ax.spines[s_].set_visible(False)
ax.grid(axis='y', lw=0.3, color='#bbbbbb'); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_perfamily.png', dpi=600); plt.close(fig)

# ---------------- Fig: conformal validity ----------------
eps_list = [0.001, 0.005, 0.01, 0.05]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={'width_ratios': [1, 1.35], 'wspace': 0.3})
ax = axs[0]
mk = {'AegisFlow': ('o', '-'), 'MLP': ('s', '--'), 'HGNN': ('^', ':'), 'SSM-Seq': ('D', '-.'), 'E-GraphSAGE': ('v', (0, (1, 1)))}
for m, (marker, ls) in mk.items():
    if m in summary and 'conf_all' in summary[m]:
        far = [summary[m]['conf_all'][f'conf_eps{e}']['far'] for e in eps_list]
        ax.plot(eps_list, far, marker=marker, ls=ls, color='black', ms=3.5, lw=0.8, label=m, mfc='white' if m != 'AegisFlow' else 'black')
ax.plot([0.0008, 0.06], [0.0008, 0.06], color='#888888', lw=0.6)
ax.text(0.013, 0.0042, 'FAR $=\\varepsilon$ line', fontsize=6.5, color='#555555')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('benign alarm level $\\varepsilon$'); ax.set_ylabel('empirical benign FAR (test)')
ax.legend(fontsize=6.5, frameon=False, loc='upper left'); ax.set_title('(a) false-alarm control', fontsize=8)
ax = axs[1]
if 'AegisFlow' in summary:
    cov = summary['AegisFlow']['cov']
    ax.bar(np.arange(10), cov, 0.6, facecolor='#bdbdbd', edgecolor='black', lw=0.5)
    ax.axhline(0.95, color='black', ls='--', lw=0.7); ax.text(9.45, 0.953, '$1-\\alpha$', fontsize=7, ha='right', va='bottom')
    ax.set_xticks(np.arange(10)); ax.set_xticklabels(['Norm.', 'Anal.', 'Backd.', 'DoS', 'Expl.', 'Fuzz.', 'Gen.', 'Rec.', 'Shell.', 'Worms'], fontsize=7)
    ax.set_ylim(0.80, 1.0); ax.set_ylabel('class-conditional coverage'); ax.set_title('(b) coverage at $\\alpha=0.05$ (AegisFlow)', fontsize=8)
for a in axs:
    for s_ in ['top', 'right']:
        a.spines[s_].set_visible(False)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.88)
fig.savefig(f'{FIG}/fig4_conformal.png', dpi=600); plt.close(fig)
