"""LOAFO open-world aggregation and figure."""
import json, glob, numpy as np, os
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W = '/home/claude/work'; FIG = '/home/claude/figs'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.linewidth': 0.6})
FAMS = ['DoS', 'Exploits', 'Fuzzers', 'Reconnaissance', 'Shellcode']
R = {}
for f in glob.glob(f'{W}/runs/*_loafo-*.json'):
    r = json.load(open(f)); fam = r['protocol'].split(':')[1]
    R.setdefault(r['variant'], {}).setdefault(fam, []).append(r)
out = {'auroc_known': {}, 'auroc_att': {}, 'op': {}, 'units': {}, 'dual': {}}
from sklearn.metrics import f1_score, roc_auc_score
from aegis_data import CLASSES
for v, d in R.items():
    for fam, rs in d.items():
        for r in rs:
            fz = f"{W}/runs/{v}_loafo-{fam}_s{r['seed']}_scores.npz"
            if not os.path.exists(fz):
                continue
            Z = np.load(fz)
            if 'Lc' not in Z:
                continue
            P = Z['P'].astype(np.float64); St = Z['St'].astype(np.float64); y = Z['y'].astype(int); Lt = Z['Lt'].astype(np.float64); Lc = Z['Lc'].astype(np.float64); yc = Z['yc'].astype(int)
            h = CLASSES.index(fam); known = [c for c in range(10) if c != h]; ymap = np.full(10, -1); ymap[known] = np.arange(9)
            unk = y == h; yk = np.where(unk, -1, ymap[np.clip(y, 0, 9)]); katt = (~unk) & (yk != 0); C = 9
            en = lambda L: -np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) - L.max(1)
            ref = np.sort(en(Lc)[yc != 0]); pE = (1 + len(ref) - np.searchsorted(ref, en(Lt), side='left')) / (len(ref) + 1)
            al = P[:, 0] <= 0.01
            Sa = St.copy(); Sa[:, 0] = np.inf; pr = np.where(al, Sa.argmin(1), 0)
            res_ = {}
            ncal = np.bincount(yc[yc >= 0], minlength=C)
            r05 = (ncal + 1 >= 1 / 0.05); r025 = (ncal + 1 >= 1 / 0.025); r05[0] = r025[0] = False
            for name, nv in [('credibility', al & ((P[:, r05] > 0.05).sum(1) == 0)), ('energy', al & (pE <= 0.05)),
                             ('dual', al & (((P[:, r025] > 0.025).sum(1) == 0) | (pE <= 0.025)))]:
                osr_true = np.where(unk, C, yk); osr_pred = np.where(nv, C, pr)
                res_[name] = dict(unk_novel=float(nv[unk].mean()), known_att_novel=float(nv[katt].mean()), unk_alerted=float(al[unk].mean()),
                                  unk_missed=float((~al)[unk].mean()), benign_far=float(al[yk == 0].mean()),
                                  open_macro_f1=float(f1_score(osr_true, osr_pred, average='macro')),
                                  known_macro_f1=float(f1_score(yk[~unk], pr[~unk], average='macro')))
            # dual novelty score for ranking: min of the two p-values (higher = more novel)
            nu = -np.minimum(P[:, 1:].max(1), pE)
            res_['dual_auroc_att'] = float(roc_auc_score(unk[unk | katt], nu[unk | katt]))
            out['dual'].setdefault(v, {}).setdefault(fam, []).append(res_)
for v, d in R.items():
    for fam, rs in d.items():
        rs = [r for r in rs if 'Conformal credibility' in r['auroc_unk_vs_knownattack']]
        if not rs:
            continue
        for key in ['auroc_unk_vs_known', 'auroc_unk_vs_knownattack']:
            for sc in rs[0][key]:
                vals = [r[key][sc] for r in rs]
                out['auroc_known' if key.endswith('known') else 'auroc_att'].setdefault(f'{v}|{sc}', {})[fam] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
                out['units'].setdefault(f'{v}|{sc}|{key}', {}).update({f'{fam}|{r["seed"]}': r[key][sc] for r in rs})
        ops = {k: float(np.mean([r['ours_op'][k] for r in rs])) for k in rs[0]['ours_op']}
        out['op'].setdefault(f'{v}|Conformal', {})[fam] = ops
        for sc in rs[0]['baseline_op']:
            out['op'].setdefault(f'{v}|{sc}', {})[fam] = {k: float(np.mean([r['baseline_op'][sc][k] for r in rs])) for k in rs[0]['baseline_op'][sc]}
# paired tests: AegisFlow conformal vs every other (backbone, score) on family x seed units (AUROC vs known attacks)
ref_key = 'AegisFlow|Energy|auroc_unk_vs_knownattack'
tests = {}
if ref_key in out['units']:
    ref = out['units'][ref_key]
    for k, u in out['units'].items():
        if k == ref_key or not k.endswith('knownattack'):
            continue
        common = sorted(set(ref) & set(u))
        if len(common) < 5:
            continue
        a = np.array([ref[c] for c in common]); b = np.array([u[c] for c in common])
        try:
            p = wilcoxon(a, b).pvalue
        except ValueError:
            p = 1.0
        tests[k] = dict(n=len(common), mean_diff=float((a - b).mean()), wins=int((a > b).sum()), p=float(p))
    order = sorted(tests, key=lambda k: tests[k]['p']); run = 0
    for i, k in enumerate(order):
        run = max(run, min(1.0, tests[k]['p'] * (len(order) - i))); tests[k]['p_holm'] = float(run)
out['tests'] = tests
json.dump(out, open(f'{W}/res_open.json', 'w'), indent=1)
for k in sorted(out['auroc_att']):
    v = out['auroc_att'][k]
    print(k, {f: round(v[f][0], 3) for f in v}, round(np.mean([v[f][0] for f in v]), 3))

# --------- figure: (a) AUROC vs known attacks, (b) outcome of unknown flows under the conformal rule
cfg = [('AegisFlow|Conformal credibility', 'AegisFlow, credibility', '#3a3a3a', ''),
       ('AegisFlow|Energy', 'AegisFlow, energy', '#9a9a9a', ''),
       ('AegisFlow|kNN', 'AegisFlow, kNN', '#d4d4d4', ''),
       ('MLP|Energy', 'MLP, energy', 'white', '////'),
       ('MLP|kNN', 'MLP, kNN', 'white', 'xxxx')]
cfg = [c for c in cfg if c[0] in out['auroc_att']]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'width_ratios': [1.55, 1], 'wspace': 0.28})
ax = axs[0]; xs = np.arange(len(FAMS)); wb = 0.8 / max(1, len(cfg))
for i, (k, lab, fc, h) in enumerate(cfg):
    vals = [out['auroc_att'][k].get(f, (np.nan,))[0] for f in FAMS]
    ax.bar(xs - 0.4 + wb * (i + 0.5), vals, wb, facecolor=fc, edgecolor='black', hatch=h, lw=0.5, label=lab)
ax.axhline(0.5, color='#777777', lw=0.6, ls='--')
ax.set_xticks(xs); ax.set_xticklabels(['DoS', 'Exploits', 'Fuzzers', 'Recon.', 'Shellcode'])
ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel('AUROC, unknown vs known attacks')
ax.legend(ncol=3, fontsize=6.3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.01), columnspacing=0.9, handlelength=1.5)
ax.set_title('(a) family-level novelty ranking', fontsize=8)
ax = axs[1]
dd = out['dual'].get('AegisFlow', {})
if all(f_ in dd for f_ in FAMS):
    g_ = lambda f_, k_: float(np.mean([r['dual'][k_] for r in dd[f_]]))
    nov = np.array([g_(f_, 'unk_novel') for f_ in FAMS]); alr = np.array([g_(f_, 'unk_alerted') for f_ in FAMS]); miss = np.array([g_(f_, 'unk_missed') for f_ in FAMS])
    mis = alr - nov
    ax.bar(xs, nov, 0.6, facecolor='#3a3a3a', edgecolor='black', lw=0.5, label='zero-day report')
    ax.bar(xs, mis, 0.6, bottom=nov, facecolor='white', edgecolor='black', hatch='////', lw=0.5, label='alert, known family')
    ax.bar(xs, miss, 0.6, bottom=nov + mis, facecolor='#d0d0d0', edgecolor='black', lw=0.5, label='missed (benign)')
    ax.set_xticks(xs); ax.set_xticklabels(['DoS', 'Expl.', 'Fuzz.', 'Recon.', 'Shell.'])
    ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel('fraction of unknown-family flows')
    ax.legend(fontsize=6.3, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.01), columnspacing=0.8, handlelength=1.5)
    ax.set_title('(b) AegisFlow decisions, dual rule', fontsize=8)
for a in axs:
    for s_ in ['top', 'right']:
        a.spines[s_].set_visible(False)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.9)
fig.savefig(f'{FIG}/fig5_openworld.png', dpi=600); plt.close(fig)
