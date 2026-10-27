import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, balanced_accuracy_score


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(P, y, bins=15):
    conf = P.max(1); pred = P.argmax(1); acc = pred == y; e = 0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def closed_metrics(LG, y, C, normal=0):
    P = softmax(LG); pred = P.argmax(1)
    att = y != normal; patt = pred != normal
    oh = np.eye(C)[y]
    return dict(acc=(pred == y).mean(), macro_f1=f1_score(y, pred, average='macro', labels=list(range(C)), zero_division=0),
                bal_acc=balanced_accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred),
                det_rate=patt[att].mean(), far=patt[~att].mean(),
                ece=ece(P, y), brier=((P - oh) ** 2).sum(1).mean(), nll=-np.log(P[np.arange(len(y)), y] + 1e-12).mean(),
                per_class_f1=f1_score(y, pred, average=None, labels=list(range(C)), zero_division=0).tolist())
