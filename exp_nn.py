"""Neural experiments: closed-set chronological benchmark, leave-one-attack-family-out
(LOAFO) open-world evaluation and ablations, all on UNSW-NB15 session B.

usage: python exp_nn.py VARIANT SEED PROTOCOL [key=value ...]
PROTOCOL: closed | loafo:<Family>
"""
import sys, json, time, os, numpy as np, torch
from aegis_data import *; from aegis_train import *; from aegis_metrics import *
from aegis_model import AegisNet
from sklearn.metrics import roc_auc_score, f1_score

VARIANTS = {  # hyper, temporal, head, ssl, beta(ctx), graph
    'MLP':            dict(hyper=False, temporal='none', head='linear', ssl=False),
    'E-GraphSAGE':    dict(hyper=False, temporal='none', head='linear', ssl=False, graph=True),
    'HGNN':           dict(hyper=True, temporal='none', head='linear', ssl=False),
    'GRU-Seq':        dict(hyper=False, temporal='gru', head='linear', ssl=False),
    'Transformer-Seq': dict(hyper=False, temporal='trf', head='linear', ssl=False),
    'SSM-Seq':        dict(hyper=False, temporal='ssm', head='linear', ssl=False),
    'AegisFlow':      dict(hyper=True, temporal='ssm', head='proto', ssl=True),
    'A-noSSL':        dict(hyper=True, temporal='ssm', head='proto', ssl=False),
    'A-attrSSL':      dict(hyper=True, temporal='ssm', head='proto', ssl=True, beta=0.0),
    'A-noHyper':      dict(hyper=False, temporal='ssm', head='proto', ssl=True),
    'A-noTemporal':   dict(hyper=True, temporal='none', head='proto', ssl=True),
    'A-Transformer':  dict(hyper=True, temporal='trf', head='proto', ssl=True),
    'A-GRU':          dict(hyper=True, temporal='gru', head='proto', ssl=True),
    'A-softmax':      dict(hyper=True, temporal='ssm', head='linear', ssl=True),
}

variant, seed, protocol = sys.argv[1], int(sys.argv[2]), sys.argv[3]
kw = dict(a.split('=') for a in sys.argv[4:])
cfg = dict(VARIANTS[variant]); cfg.setdefault('beta', 0.5); cfg.setdefault('graph', False)
L = int(kw.get('L', 16)); BLK = int(kw.get('block', 4096)); tbin = float(kw.get('tbin', 10)); rho = float(kw.get('rho', 0.35))
ep_pre = int(kw.get('ep_pre', 2)); ep_ft = int(kw.get('ep_ft', 10)); tag = kw.get('tag', '')
name = f"{variant}_{protocol.replace(':', '-')}_s{seed}{tag}"
os.makedirs(f'{WORK}/runs', exist_ok=True)
logf = open(f'{WORK}/runs/{name}.log', 'w')
def log(s):
    print(s, flush=True); logf.write(s + '\n'); logf.flush()

torch.manual_seed(seed); np.random.seed(seed)
df = load_all(); tr, ca, te = session_frames(df, 'B', [0, 6, 7.5, 13])
held = None
if protocol.startswith('loafo'):
    held = CLASSES.index(protocol.split(':')[1])
    tr = tr[tr.y != held]; ca = ca[ca.y != held]
known = [c for c in range(10) if c != held]
ymap = np.full(10, -1); ymap[known] = np.arange(len(known)); C = len(known)
fz = Featurizer().fit(tr)
S = {k: make_split(f, fz, block=BLK, tbin=tbin, L=L) for k, f in zip(['tr', 'ca', 'te'], [tr, ca, te])}
if 'maxblk' in kw:   # debugging only
    for k in S: S[k]['blocks'] = S[k]['blocks'][:int(kw['maxblk'])]
model = AegisNet(len(NUM), list(fz.card().values()), C, L=L, hyper=cfg['hyper'], temporal=cfg['temporal'],
                 head=cfg['head'], graph=cfg['graph'])
if int(kw.get('compile', 1)) and model.tm is not None:
    model.tm.forward = torch.compile(model.tm.forward, dynamic=True)
nparam = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith(('dec_', 'q_ctx', 'k_ctx')))
t0 = time.time()
if cfg['ssl']:
    pretrain(model, S['tr'], epochs=ep_pre, rho=rho, beta=cfg['beta'], seed=seed, log=log)
t_pre = time.time() - t0
finetune(model, S['tr'], ymap, C, epochs=ep_ft, seed=seed, log=log)
t_train = time.time() - t0
t1 = time.time(); Ut, Lt, yt = infer(model, S['te']); t_inf = time.time() - t1
Uc, Lc, yc = infer(model, S['ca'])
ykc = ymap[yc]
res = dict(block=BLK, ep_ft=ep_ft, ep_pre=ep_pre, variant=variant, seed=seed, protocol=protocol, params=nparam, train_s=t_train, pre_s=t_pre,
           infer_s=t_inf, infer_us_per_flow=1e6 * t_inf / len(yt), cfg=cfg, L=L, tbin=tbin, rho=rho)

# nonconformity scores
if cfg['head'] == 'proto':
    g = float(torch.exp(model.head.log_g)); Sc, St = -Lc / g, -Lt / g
else:
    Sc = -np.log(softmax(Lc) + 1e-12); St = -np.log(softmax(Lt) + 1e-12)
P = mondrian_pvalues(Sc, ykc, St, C)
nrm = 0  # Normal keeps index 0 in every protocol
if held is None:
    r = closed_metrics(Lt, yt, C); res.update(r)
    for eps in [0.001, 0.005, 0.01, 0.05]:
        al, nv, pr, fs = decide(P, St, nrm, eps, 0.05)
        res[f'conf_eps{eps}'] = dict(far=float(al[yt == 0].mean()), det=float(al[yt != 0].mean()),
                                     macro_f1=float(f1_score(yt, pr, average='macro')),
                                     set_size=float(fs[al].sum(1).mean()) if al.any() else 0.0,
                                     novel_known=float(nv[yt != 0].mean()))
    # empirical class-conditional coverage at alpha=0.05
    cov = [float((P[yt == c, c] > 0.05).mean()) for c in range(C) if (yt == c).any()]
    res['coverage_per_class'] = cov
    np.savez_compressed(f'{WORK}/pred_B_{variant}_s{seed}{tag}.npz', lt=Lt.astype(np.float16), lc=Lc.astype(np.float16),
                        pt=P.astype(np.float16))
else:
    unk = yt == held; yk = np.where(unk, -1, ymap[yt]); kn = ~unk; katt = kn & (yk != nrm)
    Pr = softmax(Lt)
    # novelty scores (higher = more novel)
    Ps = np.sort(P, 1)
    scores = {'Conformal credibility': -Ps[:, -1], 'Conformal confidence': Ps[:, -2], 'MSP': -Pr.max(1), 'MaxLogit': -Lt.max(1),
              'Energy': -np.log(np.exp(Lt - Lt.max(1, keepdims=True)).sum(1)) - Lt.max(1)}
    # Mahalanobis + kNN on embeddings fitted on a training subsample
    Utr, _, ytr_ = infer(model, S['tr']); ytr_k = ymap[ytr_]
    rs = np.random.default_rng(seed); sub = rs.choice(len(Utr), min(60000, len(Utr)), replace=False)
    Us, ys = Utr[sub], ytr_k[sub]
    mus = np.stack([Us[ys == c].mean(0) if (ys == c).any() else np.zeros(Us.shape[1]) for c in range(C)])
    Z = Us - mus[ys]; cov = np.cov(Z.T) + 1e-3 * np.eye(Z.shape[1]); icov = np.linalg.inv(cov)
    def maha(U):
        out = np.full(len(U), np.inf)
        for c in range(C):
            dlt = U - mus[c]; out = np.minimum(out, np.einsum('ij,jk,ik->i', dlt, icov, dlt))
        return out
    scores['Mahalanobis'] = maha(Ut)
    Un = Us / np.linalg.norm(Us, axis=1, keepdims=True)
    Un = Un[:30000].astype(np.float32)
    def knn(U):
        kd = []
        for i in range(0, len(U), 1000):
            q = U[i:i + 1000]; q = q / np.linalg.norm(q, axis=1, keepdims=True)
            kd.append(-np.partition(q @ Un.T, -10, axis=1)[:, -10])
        return np.concatenate(kd)
    scores['kNN'] = knn(Ut)
    res['auroc_unk_vs_known'] = {k: float(roc_auc_score(unk, v)) for k, v in scores.items()}
    res['auroc_unk_vs_knownattack'] = {k: float(roc_auc_score(unk[unk | katt], v[unk | katt])) for k, v in scores.items()}
    # operating points
    al, nv, pr, fs = decide(P, St, nrm, 0.01, 0.05)
    res['ours_op'] = dict(unk_alerted=float(al[unk].mean()), unk_novel=float(nv[unk].mean()),
                          unk_misattributed=float((al & ~nv)[unk].mean()), unk_missed=float((~al)[unk].mean()),
                          known_att_novel=float(nv[katt].mean()), benign_far=float(al[yk == nrm].mean()),
                          known_macro_f1=float(f1_score(yk[kn], pr[kn], average='macro')))
    osr_pred = np.where(nv, C, pr); osr_true = np.where(unk, C, yk)
    res['ours_op']['open_macro_f1'] = float(f1_score(osr_true, osr_pred, average='macro'))
    # energy instantiation of the zero-day rule: conformal p-value of the energy score against
    # known-attack calibration flows (class-agnostic); alert rule unchanged
    En_c = -np.log(np.exp(Lc - Lc.max(1, keepdims=True)).sum(1)) - Lc.max(1)
    En_t = -np.log(np.exp(Lt - Lt.max(1, keepdims=True)).sum(1)) - Lt.max(1)
    ref_e = np.sort(En_c[ykc != nrm]); n_e = len(ref_e)
    p_nov = (1 + n_e - np.searchsorted(ref_e, En_t, side='left')) / (n_e + 1)
    for a_ in [0.05, 0.1]:
        nv2 = al & (p_nov <= a_)
        res[f'energy_op_a{a_}'] = dict(unk_novel=float(nv2[unk].mean()), known_att_novel=float(nv2[katt].mean()),
                                      unk_alerted=float(al[unk].mean()), benign_far=float(al[yk == nrm].mean()),
                                      open_macro_f1=float(f1_score(osr_true, np.where(nv2, C, pr), average='macro')))
    # baselines: argmax prediction + novelty threshold at 5% of calibration known-attack flows
    Prc = softmax(Lc); att_c = (ykc != nrm)
    np.savez_compressed(f'{WORK}/runs/{name}_scores.npz', P=P.astype(np.float16), St=St.astype(np.float16), y=yt.astype(np.int8), Lt=Lt.astype(np.float16), Lc=Lc.astype(np.float16), yc=ykc.astype(np.int8))
    cal_scores = {'MSP': -Prc.max(1), 'MaxLogit': -Lc.max(1),
                  'Energy': -np.log(np.exp(Lc - Lc.max(1, keepdims=True)).sum(1)) - Lc.max(1)}
    Uc_ = Uc; cal_scores['Mahalanobis'] = maha(Uc_)
    cal_scores['kNN'] = knn(Uc_)
    pa = Pr.argmax(1); res['baseline_op'] = {}
    for k, v in cal_scores.items():
        th = np.quantile(v[att_c], 0.95); nvb = (pa != nrm) & (scores[k] > th)
        res['baseline_op'][k] = dict(unk_alerted=float((pa != nrm)[unk].mean()), unk_novel=float(nvb[unk].mean()),
                                     known_att_novel=float(nvb[katt].mean()), benign_far=float((pa != nrm)[yk == nrm].mean()),
                                     open_macro_f1=float(f1_score(osr_true, np.where(nvb, C, pa), average='macro')))
log(json.dumps({k: v for k, v in res.items() if k not in ('per_class_f1',)}, default=float)[:1500])
json.dump(res, open(f'{WORK}/runs/{name}.json', 'w'), default=float)
