"""Stream adaptation, ablation, sensitivity, ToN-IoT and latency summaries + figures."""
import json, glob, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W = '/home/claude/work'; FIG = '/home/claude/figs'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.linewidth': 0.6})
out = {}
# ---------------- stream
ST = {}
for f in glob.glob(f'{W}/stream/*_s0.json'):
    r = json.load(open(f)); ST[r['method']] = r
summ = {}
for m, r in ST.items():
    rows = r['rows']
    summ[m] = dict(mean_f1=float(np.mean([x['macro_f1'] for x in rows])), mean_f1_last6=float(np.mean([x['macro_f1'] for x in rows[6:]])),
                   mean_far=float(np.mean([x['far'] for x in rows])), max_far=float(np.max([x['far'] for x in rows])),
                   mean_det=float(np.mean([x['det'] for x in rows])), n_adapt=int(sum(x['adapted'] for x in rows)),
                   alarms=[x['seg'] for x in rows if x['alarm']], src_before=r['src_f1_before'], src_after=r['src_f1_after'],
                   trusted_frac=float(np.mean([x.get('trusted_frac', np.nan) for x in rows if x['adapted']])) if any(x['adapted'] for x in rows) else None,
                   trusted_acc=float(np.mean([x.get('trusted_acc', np.nan) for x in rows if x['adapted']])) if any(x['adapted'] for x in rows) else None,
                   per_seg_f1=[x['macro_f1'] for x in rows], per_seg_far=[x['far'] for x in rows], logM=[x['logM'] for x in rows],
                   attack_frac=[x['attack_frac'] for x in rows])
out['stream'] = summ
for m, s in summ.items():
    print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if not k.startswith(('per_seg', 'logM', 'attack'))})
if summ:
    order = [m for m in ['Static', 'SelfTrain', 'GatedST', 'GatedST+ER', 'GatedST+EWC', 'AegisFlow-CA', 'Label-1%'] if m in summ]
    sty = {'Static': ('o', '-', 'white'), 'SelfTrain': ('x', ':', 'black'), 'GatedST': ('s', '--', 'white'), 'GatedST+ER': ('^', '-.', 'white'),
           'GatedST+EWC': ('v', (0, (1, 1)), 'white'), 'AegisFlow-CA': ('o', '-', 'black'), 'Label-1%': ('D', (0, (5, 2)), '#9e9e9e')}
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw={'wspace': 0.36})
    segs = np.arange(len(summ[order[0]]['per_seg_f1']))
    for m in order:
        mk, ls, mf = sty[m]
        axs[0].plot(segs, summ[m]['per_seg_f1'], marker=mk, ls=ls, color='black', mfc=mf, ms=3.2, lw=0.8, label=m)
        axs[1].plot(segs, np.array(summ[m]['per_seg_far']) * 100, marker=mk, ls=ls, color='black', mfc=mf, ms=3.2, lw=0.8, label=m)
    axs[0].set_ylabel('macro-F1'); axs[0].set_title('(a) prequential macro-F1', fontsize=8)
    axs[1].set_ylabel('benign FAR (%)'); axs[1].set_title('(b) conformal rule, $\\varepsilon=0.01$', fontsize=8)
    axs[1].axhline(1.0, color='#777777', lw=0.6, ls='--')
    sp_ = json.load(open(f'{W}/res_spill.json')) if os.path.exists(f'{W}/res_spill.json') else None
    af_ = summ['Static']['attack_frac']
    axs[2].bar(segs, np.array(af_) * 100, 0.6, facecolor='#d9d9d9', edgecolor='black', lw=0.4)
    axs[2].set_ylabel('attack share of segment (%)'); axs[2].set_title('(c) static FAR vs attack share', fontsize=8)
    ax2 = axs[2].twinx()
    ax2.plot(segs, np.array(summ['Static']['per_seg_far']) * 100, 'o-', color='black', ms=3, lw=0.8)
    if sp_:
        ax2.plot(segs, np.array(sp_['MLP']['far_seg']) * 100, 's--', color='#555555', mfc='white', ms=3, lw=0.8)
    ax2.set_ylabel('benign FAR (%)'); ax2.set_ylim(0, 40); axs[2].set_ylim(0, 40)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hh = [Patch(facecolor='#d9d9d9', edgecolor='black', lw=0.4, label='attack share'), Line2D([], [], color='black', marker='o', ms=3, lw=0.8, label='AegisFlow FAR')]
    if sp_:
        hh.append(Line2D([], [], color='#555555', marker='s', mfc='white', ls='--', ms=3, lw=0.8, label='MLP FAR'))
    axs[2].legend(handles=hh, fontsize=6.0, frameon=False, loc='upper left', ncol=1)
    axs[2].set_ylim(0, 55); ax2.set_ylim(0, 55)
    ax2.spines['top'].set_visible(False)
    for a in axs:
        a.set_xlabel('hourly segment of session B'); a.set_xticks(segs[::2])
        for s_ in ['top', 'right']:
            if a is not axs[2] or s_ == 'top':
                a.spines[s_].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(order), loc='upper center', fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=2.2)
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.18, top=0.78)
    fig.savefig(f'{FIG}/fig6_stream.png', dpi=600); plt.close(fig)

# ---------------- ablation
AB = {}
for f in glob.glob(f'{W}/runs/*_closed_s[01].json'):
    r = json.load(open(f)); AB.setdefault(r['variant'], []).append(r)
abl = {}
for v in ['AegisFlow', 'A-noSSL', 'A-attrSSL', 'A-noHyper', 'A-noTemporal', 'A-Transformer', 'A-GRU', 'A-softmax']:
    rs = [r for r in AB.get(v, []) if r['seed'] in (0, 1)]
    if not rs:
        continue
    g = lambda k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
    abl[v] = dict(n=len(rs), macro_f1=g('macro_f1'), mcc=g('mcc'), ece=g('ece'), far=g('far'),
                  conf_far=float(np.mean([r['conf_eps0.01']['far'] for r in rs])), conf_det=float(np.mean([r['conf_eps0.01']['det'] for r in rs])),
                  set_size=float(np.mean([r['conf_eps0.01']['set_size'] for r in rs])), train_s=float(np.mean([r['train_s'] for r in rs])),
                  infer_us=float(np.mean([r['infer_us_per_flow'] for r in rs])), params=rs[0]['params'])
out['ablation'] = abl
for v, a in abl.items():
    print('abl', v, a['n'], np.round(a['macro_f1'], 4), round(a['conf_far'], 4), round(a['set_size'], 3))
# ablation in open world (A-softmax LOAFO vs AegisFlow)
# ---------------- sensitivity
SE = {}
for tag, key, val in [('_L4', 'L', 4), ('_L32', 'L', 32), ('_t2', 'tbin', 2), ('_t60', 'tbin', 60), ('_r15', 'rho', 0.15), ('_r60', 'rho', 0.6)]:
    f = f'{W}/runs/AegisFlow_closed_s0{tag}.json'
    if os.path.exists(f):
        r = json.load(open(f)); SE.setdefault(key, {})[val] = dict(f1=r['macro_f1'], far=r['conf_eps0.01']['far'], ece=r['ece'], train_s=r['train_s'], infer_us=r['infer_us_per_flow'])
base = f'{W}/runs/AegisFlow_closed_s0.json'
if os.path.exists(base):
    r = json.load(open(base))
    for key, val in [('L', 16), ('tbin', 10), ('rho', 0.35)]:
        SE.setdefault(key, {})[val] = dict(f1=r['macro_f1'], far=r['conf_eps0.01']['far'], ece=r['ece'], train_s=r['train_s'], infer_us=r['infer_us_per_flow'])
out['sensitivity'] = SE
print('sens', json.dumps(SE)[:600])
if all(len(SE.get(k, {})) >= 2 for k in ['L', 'tbin', 'rho']):
    from matplotlib.ticker import NullFormatter, NullLocator
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw={'wspace': 0.42})
    allf1 = [v['f1'] for d in SE.values() for v in d.values()]
    for i, (a, (key, lab, logx)) in enumerate(zip(axs, [('L', 'trajectory length $L$', True), ('tbin', 'time-bin width $\\Delta$ (s)', False), ('rho', 'mask ratio $\\rho$', False)])):
        xs = sorted(SE[key], key=float); xv = [float(x) for x in xs]
        f1 = [SE[key][x]['f1'] for x in xs]; far = [100 * SE[key][x]['far'] for x in xs]
        a.plot(xv, f1, 'o-', color='black', ms=3.5, lw=0.8)
        a2 = a.twinx(); a2.plot(xv, far, 's--', color='#6e6e6e', mfc='white', ms=3.5, lw=0.8)
        a.set_xlabel(lab); a.set_ylim(0.55, 0.59); a2.set_ylim(0.0, 3.0)
        if logx:
            a.set_xscale('log', base=2)
        a.xaxis.set_minor_locator(NullLocator()); a.set_xticks(xv); a.set_xticklabels([x if key != 'rho' else f'{float(x):.2f}' for x in xs])
        a.set_xlim(min(xv) / (1.4 if logx else 1) - (0 if logx else 0.08 * (max(xv) - min(xv))), max(xv) * (1.4 if logx else 1) + (0 if logx else 0.08 * (max(xv) - min(xv))))
        if i == 0:
            a.set_ylabel('macro-F1')
        else:
            a.set_yticklabels([])
        if i == 2:
            a2.set_ylabel('conformal FAR (%)', color='#4d4d4d')
        else:
            a2.set_yticklabels([])
        a.spines['top'].set_visible(False); a2.spines['top'].set_visible(False)
    fig.legend([Line2D([], [], color='black', marker='o', ms=3.5, lw=0.8), Line2D([], [], color='#6e6e6e', marker='s', mfc='white', ls='--', ms=3.5, lw=0.8)],
               ['macro-F1 (left axis)', 'conformal benign FAR, $\\varepsilon=0.01$ (right axis)'], ncol=2, loc='upper center', frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.22, top=0.84)
    fig.savefig(f'{FIG}/fig7_sensitivity.png', dpi=600); plt.close(fig)
# ---------------- ToN-IoT
if os.path.exists(f'{W}/res_toniot.json'):
    T = json.load(open(f'{W}/res_toniot.json')); tt = {}
    for r in T:
        tt.setdefault((r['head'], r['held']), []).append(r)
    ton = {}
    for (h, fam), rs in tt.items():
        ton.setdefault(h, {})[fam] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if k not in ('seed', 'held', 'head')}
    out['toniot'] = ton
    for h in ton:
        print('ton', h, {k: round(np.mean([ton[h][f][k] for f in ton[h]]), 4) for k in ['auroc_conf', 'auroc_msp', 'auroc_conf_vs_att', 'unk_novel', 'unk_alerted', 'benign_far', 'known_f1', 'known_att_novel']})
# ---------------- latency
if os.path.exists(f'{W}/res_latency.json'):
    LT = json.load(open(f'{W}/res_latency.json')); out['latency'] = LT
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for (name, d), (mk, ls) in zip(LT.items(), [('o', '-'), ('s', '--'), ('^', ':')]):
        xs = sorted(int(k) for k in d); ax.plot(xs, [d[str(x)] for x in xs], marker=mk, ls=ls, color='black', mfc='white' if mk != 'o' else 'black', ms=3.5, lw=0.8, label=name)
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(['4', '8', '16', '32', '64'])
    ax.set_xlabel('trajectory length $L$'); ax.set_ylabel('forward latency ($\\mu$s per flow)')
    ax.legend(fontsize=6.5, frameon=False, loc='upper left')
    for s_ in ['top', 'right']:
        ax.spines[s_].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.97)
    fig.savefig(f'{FIG}/fig8_latency.png', dpi=600); plt.close(fig)
json.dump(out, open(f'{W}/res_rest.json', 'w'), indent=1, default=float)
