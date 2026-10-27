"""AegisFlow model components (PyTorch, CPU friendly).

FlowEncoder           : per-flow attribute embedding
TypedHyperConv        : type-attentive message passing over typed temporal hyperedges
SelectiveSSM          : input-dependent (selective) diagonal state-space encoder over the
                        host trajectory of each flow, evaluated in closed form at the
                        sequence end (linear in the trajectory length L)
PrototypeHead         : distance-to-prototype logits used for open-set conformal scoring
AegisNet              : configurable assembly (also yields the neural baselines)
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def scatter_mean(h, idx, n, w=None):
    if w is None:
        w = torch.ones(h.shape[0], device=h.device)
    s = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, idx, h * w[:, None])
    c = torch.zeros(n, device=h.device).index_add_(0, idx, w)
    return s / c.clamp_min(1e-6)[:, None], c


class FlowEncoder(nn.Module):
    def __init__(self, n_num, cards, d):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(cards[i], 8) for i in range(len(cards))])
        self.lin = nn.Sequential(nn.Linear(n_num + 8 * len(cards), d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)
        self.mask_tok = nn.Parameter(torch.zeros(d))

    def forward(self, x, c, mask=None):
        e = torch.cat([x] + [m(c[:, i]) for i, m in enumerate(self.emb)], 1)
        h = self.ln(self.lin(e))
        if mask is not None:
            h = torch.where(mask[:, None], self.mask_tok.expand_as(h), h)
        return h


class TypedHyperConv(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.K = K
        self.msg = nn.ModuleList([nn.Linear(d + 1, d) for _ in range(K)])
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.miss = nn.Parameter(torch.zeros(K, d))
        self.ff = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, h, he, ne, hide=None, return_edges=False):
        n, d = h.shape
        msgs, edges = [], []
        for k in range(self.K):
            w = None
            if hide is not None:
                w = (~hide[k]).float()
            e, cnt = scatter_mean(h, he[k], int(ne[k]), w)
            edges.append(e)
            m = self.msg[k](torch.cat([e[he[k]], torch.log1p(cnt[he[k]])[:, None]], 1))
            if hide is not None:
                m = torch.where(hide[k][:, None], self.miss[k].expand_as(m), m)
            msgs.append(m)
        M = torch.stack(msgs, 1)                        # n,K,d
        a = torch.softmax((self.k(M) * self.q(h)[:, None]).sum(-1) / math.sqrt(d), 1)
        m = (a[..., None] * M).sum(1)
        out = self.ln(h + self.ff(torch.cat([h, m], 1)))
        if return_edges:
            return out, edges, a
        return out


class SelectiveSSM(nn.Module):
    """Selective diagonal SSM (Mamba-style parameterisation) whose final state is
    computed in closed form: h_L = sum_t exp(A * sum_{s>t} Delta_s) * Delta_t x_t B_t."""
    def __init__(self, d, di=32, N=8, kconv=4):
        super().__init__()
        self.inp = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, kconv, groups=di, padding=kconv - 1)
        self.dt = nn.Linear(di, di)
        self.B = nn.Linear(di, N, bias=False); self.C = nn.Linear(di, N, bias=False)
        A = torch.arange(1, N + 1).float().repeat(di, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, d)
        nn.init.constant_(self.dt.bias, math.log(math.expm1(0.05)))

    def forward(self, X, mask):
        n, L, _ = X.shape
        xz = self.inp(X); x, z = xz.chunk(2, -1)
        x = self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)
        x = F.silu(x) * mask[..., None]
        dt = F.softplus(self.dt(x)) * mask[..., None]          # n,L,di
        Bm = self.B(x); Cl = self.C(x[:, -1])                    # n,L,N ; n,N
        A = -torch.exp(self.A_log)                               # di,N
        S = torch.flip(torch.cumsum(torch.flip(dt, [1]), 1), [1]) - dt   # sum_{s>t}
        u = dt * x                                               # n,L,di
        hL = torch.stack([(torch.exp(S * A[:, k]) * u * Bm[:, :, k:k + 1]).sum(1)
                          for k in range(A.shape[1])], -1)       # n,di,N
        y = (hL * Cl[:, None, :]).sum(-1) + self.D * x[:, -1]
        y = y * F.silu(z[:, -1])
        return self.out(y)


class TemporalTransformer(nn.Module):
    def __init__(self, d, L):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(L, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)

    def forward(self, X, mask):
        H = self.enc(X + self.pos, src_key_padding_mask=~mask)
        return H[:, -1]


class GRUEnc(nn.Module):
    def __init__(self, d):
        super().__init__(); self.g = nn.GRU(d, d, batch_first=True)

    def forward(self, X, mask):
        return self.g(X * mask[..., None])[0][:, -1]


class PrototypeHead(nn.Module):
    def __init__(self, d, C):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(C, d) * 0.5)
        self.log_g = nn.Parameter(torch.zeros(()))

    def dist(self, u):
        return torch.cdist(u, self.mu).pow(2)

    def forward(self, u):
        return -torch.exp(self.log_g) * self.dist(u)


class AegisNet(nn.Module):
    def __init__(self, n_num, cards, C, d=64, L=32, hyper=True, temporal='ssm',
                 head='proto', layers=2, K=5, graph=False):
        super().__init__()
        self.hyper, self.temporal, self.headtype, self.graph = hyper, temporal, head, graph
        self.enc = FlowEncoder(n_num, cards, d)
        self.hc = nn.ModuleList([TypedHyperConv(d, K) for _ in range(layers)]) if hyper else None
        if graph:   # E-GraphSAGE-style host graph (edge = flow)
            self.sage = nn.ModuleList([nn.Linear(3 * d, d) for _ in range(2)])
            self.sage_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.tm = {'ssm': lambda: SelectiveSSM(d), 'trf': lambda: TemporalTransformer(d, L),
                   'gru': lambda: GRUEnc(d), 'none': lambda: None}[temporal]()
        din = d * (2 if temporal != 'none' else 1) + (2 * d if graph else 0)
        self.fuse = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.head = PrototypeHead(d, C) if head == 'proto' else nn.Linear(d, C)
        # self-supervised decoders
        self.dec_x = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_num))
        self.dec_c = nn.ModuleList([nn.Linear(d, cards[i]) for i in range(len(cards))])
        self.q_ctx = nn.Linear(d, d); self.k_ctx = nn.Linear(d, d)

    def embed(self, b, mask=None, hide=None, want_edges=False):
        h = self.enc(b['x'], b['c'], mask)
        edges = None
        if self.hyper:
            for i, layer in enumerate(self.hc):
                if want_edges and i == len(self.hc) - 1:
                    h, edges, _ = layer(h, b['he'], b['ne'], hide, return_edges=True)
                else:
                    h = layer(h, b['he'], b['ne'], hide)
        parts = [h]
        if self.tm is not None:
            seq = b['seq']; m = seq >= 0
            X = h[seq.clamp_min(0)] * m[..., None]
            parts.append(self.tm(X, m))
        if self.graph:
            src, dst = b['src'], b['dst']; nh = int(max(src.max(), dst.max())) + 1
            ag, _ = scatter_mean(torch.cat([h, h]), torch.cat([src, dst]), nh)
            z = ag
            for l in range(2):
                nb, _ = scatter_mean(torch.cat([z[dst], z[src]]), torch.cat([src, dst]), nh)
                z = self.sage_ln[l](z + F.gelu(self.sage[l](torch.cat([z, nb, ag], 1))))
            parts += [z[src], z[dst]]
        u = self.fuse(torch.cat(parts, 1))
        return (u, edges) if want_edges else u

    def logits(self, u):
        return self.head(u)
