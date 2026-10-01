"""
Runs the GluonTS MXNet REFERENCE (DeepState / DeepVAR) on one simulated dataset. Executed by run_calibration.py in the ISOLATED
virtualenv that carries mxnet 1.9.1 (it needs two shims on Python 3.12: legacy numpy aliases and setuptools' distutils); never imported by
the main environment.   usage: python reference_worker.py <in.npz> <out.npz>
"""
import sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
for a, b in (("bool", bool), ("int", int), ("float", float), ("object", object), ("complex", complex), ("str", str)):
    if not hasattr(np, a):
        setattr(np, a, b)
import pandas as pd
import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer

d = np.load(sys.argv[1], allow_pickle=True)
kind, Y, H, seed = str(d["kind"]), d["Y"], int(d["H"]), int(d["seed"])
epochs, batches, ns = int(d["epochs"]), int(d["batches"]), int(d["num_samples"])
mx.random.seed(seed); np.random.seed(seed)
start = pd.Period("2024-01-01", "D")
t0 = time.time()
trainer = Trainer(epochs=epochs, num_batches_per_epoch=batches, learning_rate=float(d["lr"]), hybridize=False)
if kind == "deepstate":
    from gluonts.mx.model.deepstate import DeepStateEstimator
    n = Y.shape[0]
    ds = ListDataset([{"start": start, "target": Y[i], "feat_static_cat": [i]} for i in range(n)], freq="D")
    est = DeepStateEstimator(freq="D", prediction_length=H, cardinality=[n], use_feat_static_cat=True, add_trend=True,
                             num_layers=1, num_cells=int(d["cells"]), trainer=trainer)
    pred = est.train(ds)
    out = np.stack([f.samples for f in pred.predict(ds, num_samples=ns)])                # (n, ns, H)
else:
    from gluonts.mx.model.deepvar import DeepVAREstimator
    dim = Y.shape[0]
    ds = ListDataset([{"start": start, "target": Y}], freq="D", one_dim_target=False)
    est = DeepVAREstimator(freq="D", prediction_length=H, target_dim=dim, context_length=int(d["context"]), num_layers=1,
                           num_cells=int(d["cells"]), rank=int(d["rank"]), trainer=trainer)
    pred = est.train(ds)
    out = next(iter(pred.predict(ds, num_samples=ns))).samples                            # (ns, H, dim)
np.savez(sys.argv[2], samples=out, seconds=time.time() - t0)
print("reference done", kind, out.shape, round(time.time() - t0, 1), "s")
