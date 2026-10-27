"""Forward latency of trajectory encoders versus trajectory length L (block of 4096 flows, CPU, 2 threads)."""
import torch, time, json, numpy as np
from aegis_model import SelectiveSSM, TemporalTransformer, GRUEnc
torch.set_num_threads(2); torch.manual_seed(0)
n, d = 4096, 64; out = {}
for L in [4, 8, 16, 32, 64]:
    X = torch.randn(n, L, d); M = torch.ones(n, L, dtype=torch.bool)
    for name, mod in [('Selective SSM (closed form)', SelectiveSSM(d)), ('Transformer layer', TemporalTransformer(d, L)), ('GRU', GRUEnc(d))]:
        mod.eval(); ts = []
        with torch.no_grad():
            mod(X, M)
            for _ in range(5):
                t = time.perf_counter(); mod(X, M); ts.append(time.perf_counter() - t)
        out.setdefault(name, {})[L] = 1e6 * float(np.median(ts)) / n
        print(name, L, round(out[name][L], 2), 'us/flow', flush=True)
json.dump(out, open('/home/claude/work/res_latency.json', 'w'), indent=1)
