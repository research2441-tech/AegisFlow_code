"""Cross-session deployment stream: train on the January capture (session A), deploy on the
February capture (session B) processed as 12 hourly segments with prequential evaluation.

usage: python exp_stream.py METHOD SEED
METHOD: Static | SelfTrain | GatedST | GatedST+ER | GatedST+EWC | AegisFlow-CA | Label-1%
"""
import sys, json, time, os, copy, numpy as np, torch, torch.nn.functional as F
from aegis_data import *; from aegis_train import *; from aegis_metrics import *
from aegis_model import AegisNet
from sklearn.metrics import f1_score

method, seed = sys.argv[1], int(sys.argv[2])
EPS, ALPHA, TAU, DELTA = 0.01, 0.05, 0.2, 0.01
os.makedirs(f'{WORK}/stream', exist_ok=True)
torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
df = load_all()
A = df[df.sess == 'A']; thA = (A.Stime - A.Stime.min()) / 3600
A = A[thA < 4.0].reset_index(drop=True)
B = df[df.sess == 'B'].reset_index(drop=True)
fz = Featurizer().fit(A)
SA = make_split(A, fz); SB = make_split(B, fz)
# block-level random split of the source into train / calibration
nbA = len(SA['blocks']); perm = rng.permutation(nbA); ncal = nbA // 5
cal_ids, tr_ids = sorted(perm[:ncal]), sorted(perm[ncal:])
def sub(sp, ids):
    s = dict(sp); s['blocks'] = [sp['blocks'][i] for i in ids]; return s
STr, SCa = sub(SA, tr_ids), sub(SA, cal_ids)
DEBUG = len(sys.argv) > 3
if DEBUG:
    STr, SCa = sub(SA, tr_ids[:3]), sub(SA, cal_ids[:3])
C = 10; ymap = np.arange(10)
ckpt = f'{WORK}/stream/source_s{seed}{"_dbg" if DEBUG else ""}.pt'
model = AegisNet(len(NUM), list(fz.card().values()), C)
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt))
else:
    pretrain(model, STr, epochs=2, seed=seed, log=print); finetune(model, STr, ymap, C, epochs=10, seed=seed, log=print)
    torch.save(model.state_dict(), ckpt)
old = copy.deepcopy(model).eval()
for p in old.parameters():
    p.requires_grad_(False)


def calib(m):
    U, Lc, yc = infer(m, SCa); g = float(torch.exp(m.head.log_g)); Sc = -Lc / g
    smin = np.sort(Sc.min(1))
    return Sc, yc, g, smin


Sc, yc, g, smin = calib(model)
# segments of session B (hourly)
tB = SB['t'] - SB['t'].min()
seg_of_block = np.array([int(tB[b['s']] // 3600) for b in SB['blocks']])
memory = [STr['blocks'][i] for i in rng.choice(len(STr['blocks']), min(12, len(STr['blocks'])), replace=False)]  # episodic memory
mu_old = model.head.mu.detach().clone()
fisher = None
if method == 'GatedST+EWC':
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for b in memory:
        bt = batch(STr, b); lg = model.logits(model.embed(bt))
        loss = F.cross_entropy(lg, bt['y']); model.zero_grad(); loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2 / len(memory)
    theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
logM, alarms, rows = 0.0, [], []
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
cw = class_weights(STr['y'][np.concatenate([np.arange(b['s'], b['e']) for b in STr['blocks']])], C)
for seg in range(int(seg_of_block.max()) + 1):
    ids = np.where(seg_of_block == seg)[0]
    if DEBUG:
        ids = ids[:2]
        if seg > 2:
            break
    if len(ids) == 0:
        continue
    Sseg = sub(SB, ids)
    U, Lt, yt = infer(model, Sseg); St = -Lt / g
    P = mondrian_pvalues(Sc, yc, St, C)
    ncal = np.bincount(yc, minlength=C); res_ok = resolvable(ncal, ALPHA); res_ok[0] = False
    al, nv, pr, fs = decide(P, St, 0, EPS, ALPHA, n_cal=ncal)
    # drift monitor: conformal test martingale on class-agnostic p-values (1-in-25 subsample)
    pi = (1 + len(smin) - np.searchsorted(smin, St.min(1)[::25], side='left')) / (len(smin) + 1)
    seg_alarm = False
    for p in pi:
        # simple-mixture power martingale increment: int_0^1 eps p^(eps-1) d eps
        lp = np.log(p)
        inc = (p * lp - p + 1) / (p * lp * lp) if p < 0.999 else 1.0
        logM = max(logM + np.log(max(inc, 1e-12)), 0.0)  # reset-at-zero (CUSUM-style) log martingale
        if logM > np.log(1 / DELTA):
            seg_alarm = True
    row = dict(seg=seg, n=int(len(yt)), macro_f1=float(f1_score(yt, pr, average='macro', labels=list(range(C)), zero_division=0)),
               far=float(al[yt == 0].mean()), det=float(al[yt != 0].mean()), novel_rate=float(nv.mean()),
               attack_frac=float((yt != 0).mean()), logM=float(logM), alarm=seg_alarm)
    # ---------------- adaptation ----------------
    adapt = method != 'Static' and (seg_alarm or method != 'AegisFlow-CA')
    if adapt:
        if method == 'SelfTrain':
            trust = np.ones(len(yt), bool); plab = Lt.argmax(1)
        elif method == 'Label-1%':
            trust = rng.random(len(yt)) < 0.01; plab = yt.copy()
        else:
            single = fs.sum(1) == 1
            ben = (~al) & (P[:, res_ok].max(1) <= ALPHA)
            att = al & single & (P[np.arange(len(pr)), pr] >= TAU)
            trust = ben | att; plab = np.where(ben, 0, pr)
        row['trusted_frac'] = float(trust.mean()); row['trusted_acc'] = float((plab[trust] == yt[trust]).mean()) if trust.any() else 0.0
        offs = np.cumsum([0] + [Sseg['blocks'][i]['e'] - Sseg['blocks'][i]['s'] for i in range(len(ids))])
        for ep in range(1):
            for j in rng.permutation(len(ids)):
                b = Sseg['blocks'][j]; bt = batch(Sseg, b)
                tr_ = torch.from_numpy(trust[offs[j]:offs[j + 1]]); yl = torch.from_numpy(plab[offs[j]:offs[j + 1]]).long()
                if tr_.sum() < 8:
                    continue
                u = model.embed(bt); loss = F.cross_entropy(model.logits(u)[tr_], yl[tr_], weight=cw)
                if method in ('GatedST+ER', 'AegisFlow-CA'):
                    mb = memory[rng.integers(len(memory))]; bm = batch(STr, mb); um = model.embed(bm)
                    loss = loss + F.cross_entropy(model.logits(um), bm['y'], weight=cw)
                    if method == 'AegisFlow-CA':
                        with torch.no_grad():
                            uo = old.embed(bm)
                        loss = loss + 1.0 * F.mse_loss(um, uo) + 1.0 * (model.head.mu - mu_old).pow(2).sum(1).mean()
                if fisher is not None:
                    loss = loss + 50.0 * sum((fisher[n] * (p - theta0[n]) ** 2).sum() for n, p in model.named_parameters())
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        Sc, yc, g, smin = calib(model)   # re-calibrate on the labelled source calibration blocks
        if method == 'AegisFlow-CA':
            logM = 0.0
    row['adapted'] = bool(adapt)
    rows.append(row); print(json.dumps(row), flush=True)
# forgetting: source calibration-block performance after the stream (prototype argmax)
_, Lsrc, ysrc = infer(model, SCa); _, Lsrc0, _ = infer(old, SCa)
res = dict(method=method, seed=seed, rows=rows,
           src_f1_after=float(f1_score(ysrc, Lsrc.argmax(1), average='macro')),
           src_f1_before=float(f1_score(ysrc, Lsrc0.argmax(1), average='macro')))
json.dump(res, open(f'{WORK}/stream/{method}_s{seed}{"_dbg" if DEBUG else ""}.json', 'w'), default=float)
print('forgetting', res['src_f1_before'], res['src_f1_after'])
