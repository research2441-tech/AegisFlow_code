"""Flow-level tree baselines (Random Forest, LightGBM) on the identical chronological split."""
import numpy as np, json, time, sys
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from aegis_data import *; from aegis_metrics import *

df = load_all(); tr, ca, te = session_frames(df, 'B', [0, 6, 7.5, 13]); fz = Featurizer().fit(tr)
def XY(f):
    x, c = fz.transform(f.reset_index(drop=True)); return np.hstack([x, c]).astype(np.float32), f['y'].values
Xtr, ytr = XY(tr); Xca, yca = XY(ca); Xte, yte = XY(te)
cat_idx = list(range(Xtr.shape[1] - 4, Xtr.shape[1]))
out = {}
for seed in [0, 1, 2]:
    t = time.time()
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, class_weight='balanced',
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=2, verbose=-1)
    m.fit(Xtr, ytr, categorical_feature=cat_idx)
    tt = time.time() - t; t = time.time(); P = m.predict_proba(Xte); ti = time.time() - t
    Pc = m.predict_proba(Xca)
    r = closed_metrics(np.log(P + 1e-9), yte, 10); r.update(train_s=tt, infer_s=ti)
    out.setdefault('LightGBM', []).append(r)
    np.savez_compressed(f'{WORK}/pred_B_LightGBM_s{seed}.npz', lt=np.log(P + 1e-9).astype(np.float16), lc=np.log(Pc + 1e-9).astype(np.float16))
    print('lgb', seed, round(r['macro_f1'], 4), round(tt), flush=True)
    t = time.time()
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight='balanced_subsample',
                                n_jobs=2, random_state=seed, max_features='sqrt')
    rf.fit(Xtr, ytr); tt = time.time() - t; t = time.time(); P = rf.predict_proba(Xte); ti = time.time() - t
    Pc = rf.predict_proba(Xca)
    r = closed_metrics(np.log(P + 1e-9), yte, 10); r.update(train_s=tt, infer_s=ti)
    out.setdefault('RandomForest', []).append(r)
    np.savez_compressed(f'{WORK}/pred_B_RandomForest_s{seed}.npz', lt=np.log(P + 1e-9).astype(np.float16), lc=np.log(Pc + 1e-9).astype(np.float16))
    print('rf', seed, round(r['macro_f1'], 4), round(tt), flush=True)
    json.dump(out, open(f'{WORK}/res_trees_B.json', 'w'), default=float)
np.save(f'{WORK}/y_B_te.npy', yte); np.save(f'{WORK}/y_B_ca.npy', yca)
