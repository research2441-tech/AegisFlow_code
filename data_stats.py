import numpy as np, pandas as pd, json
from aegis_data import *
df = load_all()
trB, caB, teB = session_frames(df, 'B', [0, 6, 7.5, 13])
A = df[df.sess == 'A']; thA = (A.Stime - A.Stime.min()) / 3600; A4 = A[thA < 4]
B = df[df.sess == 'B']
out = {}
for name, f in [('B-train', trB), ('B-cal', caB), ('B-test', teB), ('A-source(0-4h)', A4), ('B-stream', B), ('A-all', A)]:
    vc = f.attack_cat.value_counts().to_dict()
    out[name] = dict(n=len(f), hours=float((f.Stime.max() - f.Stime.min()) / 3600), counts={c: int(vc.get(c, 0)) for c in CLASSES},
                     n_src=int(f.srcip.nunique()), n_dst=int(f.dstip.nunique()))
    print(name, out[name])
# label conflicts: identical feature vectors (numeric+cats) with different family labels in B-train
def key(f):
    x = f[NUM].round(4).astype(str).agg('|'.join, axis=1) + '|' + f.proto.astype(str) + f.state.astype(str) + f.service.astype(str)
    return x
att = trB[trB.y > 0]
k = key(att)
g = pd.DataFrame({'k': k.values, 'y': att.y.values})
nlab = g.groupby('k').y.nunique()
conf_keys = set(nlab[nlab > 1].index)
out['conflict_frac_train_attack'] = float(g.k.isin(conf_keys).mean())
per = g[g.k.isin(conf_keys)].y.value_counts().to_dict()
out['conflict_by_class'] = {CLASSES[c]: float(per.get(c, 0) / (g.y == c).sum()) for c in range(1, 10)}
out['dup_frac_train_attack'] = float(g.k.duplicated(keep=False).mean())
print(out['conflict_frac_train_attack'], out['conflict_by_class'], out['dup_frac_train_attack'])
json.dump(out, open('/home/claude/work/data_stats.json', 'w'), indent=1)
