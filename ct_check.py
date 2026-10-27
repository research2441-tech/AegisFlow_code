"""How much do the dataset's own context counters (ct_*) carry? RF with and without them (seed 0)."""
import numpy as np, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from aegis_data import *
df = load_all(); tr, ca, te = session_frames(df, 'B', [0, 6, 7.5, 13]); fz = Featurizer().fit(tr)
def XY(f):
    x, c = fz.transform(f.reset_index(drop=True)); return np.hstack([x, c]).astype(np.float32), f['y'].values
Xtr, ytr = XY(tr); Xte, yte = XY(te)
ct = [i for i, n in enumerate(NUM) if n.startswith('ct_')]
keep = [i for i in range(Xtr.shape[1]) if i not in ct]
out = {}
for name, cols in [('all', list(range(Xtr.shape[1]))), ('no_ct', keep)]:
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=1, random_state=0, max_features='sqrt')
    rf.fit(Xtr[:, cols], ytr); pr = rf.predict(Xte[:, cols])
    out[name] = dict(macro_f1=float(f1_score(yte, pr, average='macro')), far=float((pr[yte == 0] != 0).mean()), det=float((pr[yte != 0] != 0).mean()))
    print(name, out[name], flush=True)
out['ct_features'] = [NUM[i] for i in ct]
json.dump(out, open(f'{WORK}/res_ct.json', 'w'), indent=1)
