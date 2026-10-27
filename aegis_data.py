"""Data pipeline for AegisFlow on the full UNSW-NB15 flow records.

Flows are sorted by start time, featurised, and cut into contiguous blocks.
Inside each block the temporal hypergraph is formed from five typed hyperedge
families (source host, destination host, destination service endpoint,
host pair, protocol-state-service signature), each restricted to a time bin.
Each flow also receives the indices of the previous L-1 flows of the same
source host inside the block, which feed the selective state-space encoder.
"""
import numpy as np, pandas as pd, os, pickle

CLASSES = ['Normal', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
           'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
NUM = ['dur', 'sbytes', 'dbytes', 'sloss', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts',
       'swin', 'dwin', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
       'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
       'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
       'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
# excluded on purpose: sttl, dttl, ct_state_ttl (testbed TTL artefacts), stcpb/dtcpb
# (random ISNs), IP addresses and source port (identifiers), Stime/Ltime.
CAT = ['proto', 'state', 'service', 'dport']
HE_TYPES = ['src', 'dst', 'dstsvc', 'pair', 'sig']

WORK = '/home/claude/work'


def load_all():
    slim = f'{WORK}/unsw_slim.pkl'
    if os.path.exists(slim):
        return pd.read_pickle(slim)
    df = pd.read_pickle(f'{WORK}/unsw_all.pkl')
    df['sess'] = np.where(df['day'] == '2015-02-18', 'B', 'A')
    df['y'] = df['attack_cat'].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
    keep = NUM + ['proto', 'state', 'service', 'srcip', 'dstip', 'dsport', 'Stime', 'y', 'sess', 'attack_cat']
    df = df[keep].copy()
    for c in ['proto', 'state', 'service', 'srcip', 'dstip', 'sess', 'attack_cat']:
        df[c] = df[c].astype('category')
    df['dsport'] = df['dsport'].astype(np.float32); df['y'] = df['y'].astype(np.int64)
    df.to_pickle(slim)
    return df


def port_token(p):
    p = np.nan_to_num(p, nan=-1).astype(np.int64)
    return p


class Featurizer:
    """Log-scaling + standardisation fitted on training rows only; categorical
    vocabularies are frozen on the training rows (unseen values map to OOV)."""
    def fit(self, d):
        x = np.log1p(np.clip(d[NUM].values.astype(np.float64), 0, None))
        self.mu = x.mean(0); self.sd = x.std(0) + 1e-6
        self.vocab = {}
        for c in ['proto', 'state', 'service']:
            vc = d[c].astype(str).value_counts()
            keep = vc.index[:24].tolist()
            self.vocab[c] = {v: i + 1 for i, v in enumerate(keep)}
        vc = pd.Series(port_token(d['dsport'].values)).value_counts()
        keep = [p for p in vc.index[:48] if p >= 0]
        self.vocab['dport'] = {int(p): i + 4 for i, p in enumerate(keep)}
        return self

    def card(self):
        return {c: len(v) + 4 for c, v in self.vocab.items()}

    def transform(self, d):
        x = (np.log1p(np.clip(d[NUM].values.astype(np.float64), 0, None)) - self.mu) / self.sd
        cats = []
        for c in ['proto', 'state', 'service']:
            cats.append(d[c].astype(str).map(self.vocab[c]).fillna(0).astype(np.int64).values)
        p = port_token(d['dsport'].values)
        dp = pd.Series(p).map(self.vocab['dport'])
        bucket = np.where(p < 0, 0, np.where(p < 1024, 1, np.where(p < 49152, 2, 3)))
        cats.append(np.where(dp.isna(), bucket, dp.fillna(0)).astype(np.int64))
        return x.astype(np.float32), np.stack(cats, 1)


def block_structures(d, block, tbin, L):
    """Return per-block hyperedge memberships and host-sequence indices."""
    src = pd.factorize(d['srcip'])[0]
    dst = pd.factorize(d['dstip'])[0]
    dport = port_token(d['dsport'].values)
    sig = pd.factorize(d['proto'].astype(str) + '|' + d['state'].astype(str) + '|' + d['service'].astype(str))[0]
    tb = ((d['Stime'].values - d['Stime'].values.min()) // tbin).astype(np.int64)
    keys = {
        'src': src.astype(np.int64),
        'dst': dst.astype(np.int64),
        'dstsvc': dst.astype(np.int64) * 70000 + (dport + 1),
        'pair': src.astype(np.int64) * 100000 + dst,
        'sig': sig.astype(np.int64),
    }
    n = len(d); out = []
    for s in range(0, n, block):
        e = min(n, s + block); m = e - s
        he = np.zeros((len(HE_TYPES), m), np.int64); ne = []
        for k, t in enumerate(HE_TYPES):
            kk = keys[t][s:e] * 1000 + (tb[s:e] - tb[s])
            inv = np.unique(kk, return_inverse=True)[1]
            he[k] = inv; ne.append(int(inv.max()) + 1)
        # host sequences (causal, same source host, inside the block)
        g = src[s:e]
        order = np.lexsort((np.arange(m), g))
        gs = g[order]
        seq = np.full((m, L), -1, np.int64)
        seq[order, L - 1] = order
        for k in range(1, L):
            prev = np.full(m, -1, np.int64)
            ok = np.zeros(m, bool); ok[k:] = gs[k:] == gs[:-k]
            prev[k:][ok[k:]] = order[:-k][ok[k:]]
            seq[order, L - 1 - k] = prev
        out.append(dict(s=s, e=e, he=he, ne=np.array(ne), seq=seq))
    return out


def make_split(d, fz, block=4096, tbin=10, L=16):
    """Featurise a (time-sorted) subset and build its blocks; no structure
    crosses split boundaries, so held-out families can be physically removed."""
    d = d.reset_index(drop=True)
    x, c = fz.transform(d)
    bl = block_structures(d, block, tbin, L)
    return dict(x=x, c=c, y=d['y'].values, t=d['Stime'].values, blocks=bl,
                src=pd.factorize(d['srcip'])[0], dst=pd.factorize(d['dstip'])[0], n=len(d))


def session_frames(df, sess, cuts):
    """Chronological train / calibration / test frames of one session; cuts in hours."""
    d = df[df.sess == sess]
    th = (d['Stime'] - d['Stime'].min()) / 3600.0
    return [d[(th >= a) & (th < b)] for a, b in zip(cuts[:-1], cuts[1:])]
