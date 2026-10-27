"""LightGBM baseline with configuration chosen on the calibration slice (macro-F1), then 3 seeds on test.
The first configuration (lr 0.1, 200 rounds, balanced weights) diverged on this data (test macro-F1 0.18-0.27)."""
import numpy as np, json, time, os
import lightgbm as lgb
from sklearn.metrics import f1_score
from aegis_data import *; from aegis_metrics import *
os.environ.setdefault('OMP_NUM_THREADS', '2')
df = load_all(); tr, ca, te = session_frames(df, 'B', [0, 6, 7.5, 13]); fz = Featurizer().fit(tr)
def XY(f):
    x, c = fz.transform(f.reset_index(drop=True)); return np.hstack([x, c]).astype(np.float32), f['y'].values
Xtr, ytr = XY(tr); Xca, yca = XY(ca); Xte, yte = XY(te)
cat_idx = list(range(Xtr.shape[1] - 4, Xtr.shape[1]))
cfgs = {'lr0.05_bal': dict(learning_rate=0.05, n_estimators=300, class_weight='balanced', min_child_samples=20),
        'lr0.05_none': dict(learning_rate=0.05, n_estimators=300, class_weight=None, min_child_samples=20),
        'lr0.03_sqrt': dict(learning_rate=0.03, n_estimators=400, class_weight='sqrt', min_child_samples=20)}
cnt = np.bincount(ytr, minlength=10); sq = {c: float((cnt.max() / max(cnt[c], 1)) ** 0.5) for c in range(10)}
sel = {}
for name, c in cfgs.items():
    c = dict(c)
    if c['class_weight'] == 'sqrt':
        c['class_weight'] = sq
    m = lgb.LGBMClassifier(num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=2, verbose=-1, **c)
    m.fit(Xtr, ytr, categorical_feature=cat_idx)
    sel[name] = f1_score(yca, m.predict(Xca), average='macro'); print(name, round(sel[name], 4), flush=True)
best = max(sel, key=sel.get); print('selected', best)
W = WORK
out = json.load(open(f'{W}/res_trees_B.json')) if os.path.exists(f'{W}/res_trees_B.json') else {}
out['LightGBM'] = []; out['LightGBM_selection'] = sel
for seed in [0, 1, 2]:
    c = dict(cfgs[best]); c['class_weight'] = sq if c['class_weight'] == 'sqrt' else c['class_weight']
    m = lgb.LGBMClassifier(num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=2, verbose=-1, **c)
    t = time.time(); m.fit(Xtr, ytr, categorical_feature=cat_idx); tt = time.time() - t
    t = time.time(); P = m.predict_proba(Xte); ti = time.time() - t; Pc = m.predict_proba(Xca)
    r = closed_metrics(np.log(P + 1e-9), yte, 10); r.update(train_s=tt, infer_s=ti, config=best); out['LightGBM'].append(r)
    np.savez_compressed(f'{W}/pred_B_LightGBM_s{seed}.npz', lt=np.log(P + 1e-9).astype(np.float16), lc=np.log(Pc + 1e-9).astype(np.float16))
    print('lgb', seed, round(r['macro_f1'], 4), flush=True)
json.dump(out, open(f'{W}/res_trees_B.json', 'w'), default=float)
