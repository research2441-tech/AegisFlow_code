"""Grayscale architecture diagrams for AegisFlow (Fig. 1-3)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
import numpy as np

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7})
OUT = '/home/claude/figs'
import os; os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc='white', lw=0.9, fs=7, ls='-', bold_title=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.8', fc=fc, ec='black', lw=lw, ls=ls)
    ax.add_patch(p)
    if bold_title:
        ax.text(x + w / 2, y + h - 1.6, bold_title, ha='center', va='top', fontsize=fs, fontweight='bold')
        ax.text(x + w / 2, y + (h - 3.2) / 2, text, ha='center', va='center', fontsize=fs - 0.6, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p0, p1, ls='-', lw=0.9, rad=0.0, color='black', head=7):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=head, lw=lw, ls=ls, color=color,
                        connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def poly_arrow(ax, pts, ls='-', lw=0.9, head=7):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=lw, ls=ls, solid_capstyle='butt')
    arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw, head=head)


def R(b):  # right-middle
    return (b[0] + b[2], b[1] + b[3] / 2)
def Lm(b):
    return (b[0], b[1] + b[3] / 2)
def T(b, f=0.5):
    return (b[0] + b[2] * f, b[1] + b[3])
def Bm(b, f=0.5):
    return (b[0] + b[2] * f, b[1])


# ------------------------------------------------------------------ Fig. 1
def fig1():
    fig = plt.figure(figsize=(7.16, 3.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 54.5); ax.axis('off')
    G1, G2 = '#efefef', '#d9d9d9'
    ax.text(1, 53.2, 'Stage II: supervised metric learning and conformal inference (per block of flows)', fontsize=7, style='italic', va='center')
    y0, h = 33, 15
    b1 = box(ax, 1, y0, 13, h, 'Time-ordered\nflow records\n' + r'$(\mathbf{x}_i,\mathbf{c}_i,s_i,r_i,p_i,t_i)$', fc=G1)
    b2 = box(ax, 17.5, y0, 14.5, h, '5 typed hyperedge\nfamilies per time\nbin of width $\\Delta$', bold_title='Temporal hypergraph')
    b3 = box(ax, 35.5, y0, 14.5, h, 'flow encoder +\ntype-attentive\nhyperconv. (\u00d72)', bold_title='Relational encoder')
    b4 = box(ax, 53.5, y0, 13.5, h, 'selective SSM over\nlast $L$ flows of\nthe source host', bold_title='Trajectory encoder')
    b5 = box(ax, 70.5, y0, 12.5, h, 'fusion to $\\mathbf{u}_i$;\nprototypes\n$\\boldsymbol{\\mu}_1,\\ldots,\\boldsymbol{\\mu}_K$', bold_title='Metric head')
    b6 = box(ax, 86.5, y0, 12.5, h, 'Mondrian\np-values;\nrules $(\\varepsilon,\\alpha)$', bold_title='Conformal layer', fc=G2)
    for a_, b_ in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)]:
        arrow(ax, R(a_), Lm(b_))
    poly_arrow(ax, [T(b3, 0.75), (T(b3, 0.75)[0], 50.3), (T(b5, 0.3)[0], 50.3), T(b5, 0.3)])
    ax.text((T(b3, 0.75)[0] + T(b5, 0.3)[0]) / 2, 51.0, 'relational state $\\mathbf{h}_i$ (skip path)', ha='center', fontsize=6.3)
    ob = box(ax, 86.5, 22.5, 12.5, 7.0, 'benign | known family\nambiguous | zero-day', fs=6.2)
    arrow(ax, Bm(b6), T(ob))
    # Stage I (pretraining)
    p1 = box(ax, 17.5, 4.5, 14.5, 13.5, '35% of flows masked;\none incidence per\nmasked flow hidden', bold_title='Masking', fs=6.8)
    p2 = box(ax, 35.5, 4.5, 14.5, 13.5, 'attribute recon-\nstruction + incidence\nrecovery (InfoNCE)', bold_title='SSL decoders', fs=6.8)
    arrow(ax, T(p1, 0.5), Bm(b2, 0.5))
    arrow(ax, Bm(b3, 0.25), T(p2, 0.25))
    arrow(ax, T(p2, 0.80), Bm(b3, 0.80), ls=(0, (3, 2)))
    ax.text(48.4, 25.6, 'reverse-mode\ngradient', fontsize=6.0, ha='left', va='center', style='italic')
    ax.text(30.5, 25.6, 'embeddings', fontsize=6.0, ha='left', va='center', style='italic')
    ax.text(1, 11.25, 'Stage I:\nself-supervised\npretraining on\nunlabelled\nflows', fontsize=6.6, style='italic', va='center', ha='left')
    # Stage III (adaptation)
    c3 = box(ax, 55.5, 4.5, 12, 13.5, 'episodic replay\n+ embedding and\nprototype anchors', bold_title='Dual memory', fs=6.8)
    c2 = box(ax, 71.5, 4.5, 11, 13.5, 'singleton set\nand $p\\geq\\tau$;\nzero-day flows\nexcluded', bold_title='Trust gate', fs=6.8)
    c1 = box(ax, 87.5, 4.5, 11.5, 13.5, 'conformal\nmartingale $M_t$\non class-agnostic\np-values', bold_title='Drift monitor', fs=6.8)
    arrow(ax, Bm(ob), T(c1, (Bm(ob)[0] - c1[0]) / c1[2]))
    ax.text(Bm(ob)[0] - 0.3, 20.3, 'p-values', fontsize=6.0, ha='right', va='center', style='italic')
    arrow(ax, Lm(c1), R(c2)); ax.text((c1[0] + R(c2)[0]) / 2, 12.6, 'alarm', fontsize=6.0, ha='center', style='italic')
    arrow(ax, Lm(c2), R(c3)); ax.text((c2[0] + R(c3)[0]) / 2, 12.6, 'trusted', fontsize=6.0, ha='center', style='italic')
    arrow(ax, T(c3, 0.6), Bm(b4, (T(c3, 0.6)[0] - b4[0]) / b4[2]), ls=(0, (3, 2)))
    arrow(ax, T(c2, 0.5), Bm(b5, (T(c2, 0.5)[0] - b5[0]) / b5[2]), ls=(0, (3, 2)))
    ax.text(68.6, 25.6, 'parameter\nupdate', fontsize=6.0, style='italic', ha='center', va='center')
    ax.text(55.5, 1.6, 'Stage III: drift-triggered continual adaptation', fontsize=6.6, style='italic', va='center')
    fig.savefig(f'{OUT}/fig1_architecture.png', dpi=600); plt.close(fig)


# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig = plt.figure(figsize=(7.16, 3.05)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis('off')
    ax.text(1, 40.5, '(a) Typed incidence of flows inside one time bin', fontsize=7, fontweight='bold', va='center')
    ax.text(52, 40.5, '(b) Type-attentive update and trajectory encoding of flow $i$', fontsize=7, fontweight='bold', va='center')
    flows = ['$f_1$', '$f_2$', '$f_3$', '$f_4$', '$f_5$', '$f_6$', '$f_7$']
    types = [('source', 2), ('dst.-service', 3), ('signature', 2)]
    inc = [[0, 0, 0], [1, 1, 0], [0, 2, 1], [1, 0, 0], [0, 2, 1], [0, 1, 1], [1, 2, 0]]  # hyperedge index per type
    x0, y0, cw, rh = 9, 33.5, 4.6, 3.7
    col = 0; starts = []
    for (tn, ne) in types:
        starts.append(col)
        cx0 = x0 + col * cw; cx1 = x0 + (col + ne) * cw
        ax.plot([cx0 + 0.3, cx1 - 0.3], [y0 + 2.3, y0 + 2.3], 'k', lw=0.8)
        ax.text((cx0 + cx1) / 2, y0 + 3.4, tn, ha='center', fontsize=6.3)
        for j in range(ne):
            ax.text(cx0 + (j + 0.5) * cw, y0 + 0.6, f'$e_{{{col + j + 1}}}$', ha='center', fontsize=6.3)
        col += ne
    for r, fl in enumerate(flows):
        yy = y0 - (r + 1) * rh
        hl = (r == 4)
        if hl:
            ax.add_patch(Rectangle((x0 - 6.3, yy - 0.2), 6.3 + col * cw, rh - 0.4, fc='#e3e3e3', ec='none', zorder=0))
        ax.text(x0 - 1.2, yy + rh / 2 - 0.2, fl, ha='right', va='center', fontsize=6.8)
        for ti, (tn, ne) in enumerate(types):
            for j in range(ne):
                cx = x0 + (starts[ti] + j) * cw
                filled = inc[r][ti] == j
                ax.add_patch(Rectangle((cx + 0.6, yy + 0.25), cw - 1.2, rh - 1.0, fc='black' if filled else 'white', ec='black', lw=0.5))
    for c in [starts[1], starts[2]]:
        ax.plot([x0 + c * cw, x0 + c * cw], [y0 - 7 * rh - 0.3, y0 + 1.6], color='#7a7a7a', lw=0.6, ls='--')
    ax.text(x0 + col * cw + 1.5, y0 - 5 * rh + rh / 2, 'flow $i=f_5$:\none incidence\nper type', fontsize=6.2, va='center')
    arrow(ax, (x0 - 6.0, 4.6), (x0 + col * cw + 0.5, 4.6)); ax.text(x0 + col * cw + 1.0, 4.6, '$t$', va='center', fontsize=7)
    ax.text(x0 - 6.0, 2.0, 'all flows start in bin $[\\tau\\Delta,(\\tau+1)\\Delta)$; destination and host-pair types omitted', fontsize=6.0)
    # (b) right panel
    ax.plot([50.5, 50.5], [2, 39], color='#9e9e9e', lw=0.6)
    bh = box(ax, 53, 23, 11, 8, '$\\mathbf{h}_i^{(l)}$', fs=7.5)
    bm = [box(ax, 53, 13.25, 11, 8, 'messages\n$\\mathbf{m}_{i,\\kappa}$, $\\kappa\\in\\mathcal{K}$', fs=6.6)]
    ba = box(ax, 68, 14.5, 11, 15, 'type\nattention\n$\\beta_{i,\\kappa}$', fs=6.8)
    bu = box(ax, 68, 32, 11, 6.5, '$\\mathbf{h}_i^{(l+1)}$', fs=7.5)
    arrow(ax, R(bh), (68, R(bh)[1])); arrow(ax, R(bm[0]), (68, R(bm[0])[1]))
    ax.text(66.0, R(bh)[1] + 0.9, 'query', fontsize=5.8, ha='center', style='italic')
    ax.text(66.0, R(bm[0])[1] + 0.9, 'keys', fontsize=5.8, ha='center', style='italic')
    arrow(ax, T(ba), Bm(bu))
    poly_arrow(ax, [T(bh, 0.5), (T(bh, 0.5)[0], Lm(bu)[1]), Lm(bu)])
    ax.text(T(bh, 0.5)[0] - 0.8, 33.5, 'residual', fontsize=6.0, ha='right', style='italic')
    arrow(ax, (51.2, R(bm[0])[1]), Lm(bm[0])); ax.text(51.6, R(bm[0])[1] + 1.0, '$\\mathbf{z}_e$', fontsize=6.4)
    bs = box(ax, 84, 19.5, 14, 18.5, 'trajectory\n$(\\mathbf{v}_{1},\\ldots,\\mathbf{v}_{L})$\nselective $\\Delta_t,\\mathbf{B}_t$\nclosed-form\nfinal state $\\mathbf{s}_L$', bold_title='Selective SSM', fs=6.8)
    arrow(ax, R(bu), (84, R(bu)[1])); ax.text(81.5, R(bu)[1] + 1.0, '$\\mathbf{v}_t$', fontsize=6.4, ha='center')
    bo = box(ax, 84, 4, 14, 8, 'trajectory\nsummary $\\mathbf{g}_i$', fs=6.8)
    arrow(ax, Bm(bs), T(bo))
    ax.text(53, 6.5, 'cost per flow: $O(|\\mathcal{K}|\\,d^2)$ for the\nhyperconvolution, $O(L\\,d_s N)$ for the SSM', fontsize=6.2, va='center')
    fig.savefig(f'{OUT}/fig2_hypergraph.png', dpi=600); plt.close(fig)


if __name__ == '__main__':
    fig1(); fig2()
