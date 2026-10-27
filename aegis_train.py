"""Training, inference and conformal decision layer for AegisFlow and baselines."""
import numpy as np, torch, torch.nn.functional as F, time
from aegis_model import AegisNet, scatter_mean

import os
torch.set_num_threads(int(os.environ.get('THREADS', '2')))


def batch(sp, b, ymap=None):
    s, e = b['s'], b['e']
    out = dict(x=torch.from_numpy(sp['x'][s:e]), c=torch.from_numpy(sp['c'][s:e]),
               he=torch.from_numpy(b['he']), ne=b['ne'], seq=torch.from_numpy(b['seq']),
               src=torch.from_numpy(sp['src'][s:e]).long(), dst=torch.from_numpy(sp['dst'][s:e]).long())
    y = sp['y'][s:e]
    if ymap is not None:
        y = ymap[y]
    out['y'] = torch.from_numpy(y).long()
    return out


def ssl_step(model, bt, rho, beta, gen):
    n = bt['x'].shape[0]; K = bt['he'].shape[0]
    mask = torch.rand(n, generator=gen) < rho
    kstar = torch.randint(0, K, (n,), generator=gen)
    hide = torch.zeros(K, n, dtype=torch.bool)
    hide[kstar[mask], torch.nonzero(mask).squeeze(1)] = True
    if model.hyper:
        u, edges = model.embed(bt, mask=mask, hide=hide, want_edges=True)
    else:
        u, edges = model.embed(bt, mask=mask), None
    um = u[mask]
    l_attr = F.mse_loss(model.dec_x(um), bt['x'][mask])
    l_cat = sum(F.cross_entropy(dc(um), bt['c'][mask, i]) for i, dc in enumerate(model.dec_c)) / len(model.dec_c)
    l_ctx = torch.zeros(())
    if model.hyper:
        idx = torch.nonzero(mask).squeeze(1); ks = kstar[idx]; tot = 0; cnt = 0
        for k in range(K):
            sel = idx[ks == k]
            if len(sel) < 2:
                continue
            E = edges[k]                                     # hyperedge embeddings (visible members)
            vis = torch.zeros(E.shape[0]).index_add_(0, bt['he'][k], (~hide[k]).float()) > 0
            tgt = bt['he'][k][sel]; ok = vis[tgt]
            if ok.sum() < 2:
                continue
            q = F.normalize(model.q_ctx(u[sel[ok]]), dim=1)
            kk = F.normalize(model.k_ctx(E), dim=1)
            lg = q @ kk.T / 0.1
            lg = lg.masked_fill(~vis[None], -1e4)
            tot = tot + F.cross_entropy(lg, tgt[ok]); cnt += 1
        if cnt:
            l_ctx = tot / cnt
    return l_attr + 0.5 * l_cat + beta * l_ctx, (l_attr.item(), l_cat.item(), float(l_ctx))


def pretrain(model, sp, epochs=3, rho=0.35, beta=0.5, lr=2e-3, seed=0, log=None):
    gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        t0 = time.time(); ls = []
        for bi in rng.permutation(len(sp['blocks'])):
            bt = batch(sp, sp['blocks'][bi])
            loss, parts = ssl_step(model, bt, rho, beta, gen)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ls.append(parts)
        if log: log(f'  pretrain ep{ep} {np.mean(ls,0).round(4)} {time.time()-t0:.0f}s')


def class_weights(y, C):
    cnt = np.bincount(y, minlength=C).astype(float) + 1
    w = (cnt.max() / cnt) ** 0.5
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def finetune(model, sp, ymap, C, epochs=4, lr=2e-3, lam=0.05, seed=0, log=None, extra=None):
    rng = np.random.default_rng(seed)
    yk = ymap[sp['y']]; w = class_weights(yk[yk >= 0], C)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    nb = len(sp['blocks']); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * nb, pct_start=0.15)
    for ep in range(epochs):
        t0 = time.time(); ls = []
        for bi in rng.permutation(nb):
            bt = batch(sp, sp['blocks'][bi], ymap)
            u = model.embed(bt); y = bt['y']; ok = y >= 0
            lg = model.logits(u)
            loss = F.cross_entropy(lg[ok], y[ok], weight=w)
            if model.headtype == 'proto':
                loss = loss + lam * model.head.dist(u[ok]).gather(1, y[ok][:, None]).mean()
            if extra is not None:
                loss = loss + extra(model)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            ls.append(loss.item())
        if log: log(f'  finetune ep{ep} loss {np.mean(ls):.4f} {time.time()-t0:.0f}s')


@torch.no_grad()
def infer(model, sp):
    model.eval(); U, LG = [], []
    for b in sp['blocks']:
        bt = batch(sp, b); u = model.embed(bt)
        U.append(u.numpy()); LG.append(model.logits(u).numpy())
    model.train()
    idx = np.concatenate([np.arange(b['s'], b['e']) for b in sp['blocks']])
    return np.concatenate(U), np.concatenate(LG), sp['y'][idx]


# ---------------- conformal decision layer ----------------
def mondrian_pvalues(S_cal, y_cal, S_test, C):
    """Class-conditional conformal p-values from nonconformity S (n x C)."""
    P = np.zeros_like(S_test)
    for c in range(C):
        sc = np.sort(S_cal[y_cal == c, c]); nc = len(sc)
        if nc == 0:
            continue
        ge = nc - np.searchsorted(sc, S_test[:, c], side='left')
        P[:, c] = (1 + ge) / (nc + 1)
    return P


def resolvable(n_cal, alpha):
    """A class can be excluded from a level-alpha conformal set only if its smallest
    attainable p-value 1/(n_c+1) is <= alpha, i.e. n_c >= 1/alpha - 1."""
    return np.asarray(n_cal) + 1 >= 1.0 / alpha


def decide(P, S, normal=0, eps=0.01, alpha=0.05, n_cal=None):
    """Alert if the benign hypothesis is rejected (p_normal <= eps); among alerts,
    known-family set {c != normal : p_c > alpha}; empty set -> zero-day alert.
    The point prediction of an alerted flow is its least nonconforming attack class."""
    alert = P[:, normal] <= eps
    Pa = P.copy(); Pa[:, normal] = -1
    if n_cal is not None:
        Pa[:, ~resolvable(n_cal, alpha)] = -1
    fam_set = Pa > alpha
    novel = alert & (fam_set.sum(1) == 0)
    Sa = S.copy(); Sa[:, normal] = np.inf
    pred = np.where(alert, Sa.argmin(1), normal)
    return alert, novel, pred, fam_set
