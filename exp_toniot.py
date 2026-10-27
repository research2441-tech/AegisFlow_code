"""External open-set check of the prototype + Mondrian conformal layer on a ToN-IoT network
subset (93,199 de-duplicated flows, 10 classes; no endpoint or timestamp fields, so only the
flow-level part of AegisFlow is applicable). Leave-one-attack-family-out, stratified random
60/20/20 split, three seeds."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, json
from sklearn.metrics import roc_auc_score, f1_score
from aegis_train import mondrian_pvalues, decide
from aegis_metrics import softmax
torch.set_num_threads(1)
d = pd.read_parquet('/home/claude/data/tg/ToN-IoT.parquet')
fam = d['family'].astype(str).values
X = d.drop(columns=['family', 'fine', 'order']).values.astype(np.float64)
X = np.sign(X) * np.log1p(np.abs(X))
classes = ['normal'] + sorted(set(fam) - {'normal'})
y_all = np.array([classes.index(f) for f in fam])
held_list = [c for c in classes[1:] if c != 'mitm']


class Net(nn.Module):
    def __init__(self, din, C, proto):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(din, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64), nn.LayerNorm(64))
        self.proto = proto
        self.mu = nn.Parameter(torch.randn(C, 64) * 0.5); self.lg = nn.Parameter(torch.zeros(()))
        self.lin = nn.Linear(64, C)

    def forward(self, x):
        u = self.f(x)
        return (-torch.exp(self.lg) * torch.cdist(u, self.mu).pow(2), u) if self.proto else (self.lin(u), u)


out = []
for seed in [0, 1, 2]:
    rng = np.random.default_rng(seed); perm = rng.permutation(len(y_all))
    n = len(perm); itr, ica, ite = perm[:int(.6 * n)], perm[int(.6 * n):int(.8 * n)], perm[int(.8 * n):]
    mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-6; Z = ((X - mu) / sd).astype(np.float32)
    for held in held_list:
        h = classes.index(held); known = [c for c in range(len(classes)) if c != h]
        ymap = np.full(len(classes), -1); ymap[known] = np.arange(len(known)); C = len(known)
        tr = itr[y_all[itr] != h]; ca = ica[y_all[ica] != h]
        for proto in [True, False]:
            torch.manual_seed(seed)
            net = Net(Z.shape[1], C, proto); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
            xt = torch.from_numpy(Z[tr]); yt = torch.from_numpy(ymap[y_all[tr]]).long()
            for ep in range(15):
                p = torch.randperm(len(tr))
                for i in range(0, len(tr), 512):
                    b = p[i:i + 512]; lg, u = net(xt[b]); loss = F.cross_entropy(lg, yt[b])
                    if proto:
                        loss = loss + 0.05 * torch.cdist(u, net.mu).pow(2).gather(1, yt[b][:, None]).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                Lc = net(torch.from_numpy(Z[ca]))[0].numpy(); Lt = net(torch.from_numpy(Z[ite]))[0].numpy()
            yc = ymap[y_all[ca]]; ytt = y_all[ite]; unk = ytt == h; yk = np.where(unk, -1, ymap[ytt])
            if proto:
                g = float(torch.exp(net.lg)); Sc, St = -Lc / g, -Lt / g
            else:
                Sc, St = -np.log(softmax(Lc) + 1e-12), -np.log(softmax(Lt) + 1e-12)
            P = mondrian_pvalues(Sc, yc, St, C)
            al, nv, pr, fs = decide(P, St, 0, 0.01, 0.05)
            katt = (~unk) & (yk != 0)
            r = dict(seed=seed, held=held, head='prototype' if proto else 'softmax',
                     auroc_conf=roc_auc_score(unk, -P.max(1)), auroc_msp=roc_auc_score(unk, -softmax(Lt).max(1)),
                     auroc_conf_vs_att=roc_auc_score(unk[unk | katt], -P.max(1)[unk | katt]),
                     unk_novel=nv[unk].mean(), unk_alerted=al[unk].mean(), known_att_novel=nv[katt].mean(),
                     benign_far=al[yk == 0].mean(), known_f1=f1_score(yk[~unk], pr[~unk], average='macro'))
            out.append(r); print(json.dumps(r, default=float), flush=True)
json.dump(out, open('/home/claude/work/res_toniot.json', 'w'), default=float)
