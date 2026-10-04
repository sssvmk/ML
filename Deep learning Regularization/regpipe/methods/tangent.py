"""7.14  Tangent distance, tangent propagation and the manifold tangent classifier.

Tangent prop   : penalise the derivative of the network output along tangent vectors v_i of the
                 data manifold:  Omega = sum_i || (d softmax(f(x)) / dx) . v_i ||^2  (computed
                 exactly with forward-mode autodiff, torch.func.jvp).
  source=affine   : analytic tangents of shift-x, shift-y, rotation, zoom and shear
                    (finite differences of the transformations).
  source=manifold : tangents LEARNED FROM THE DATA with local PCA: for each training image take
                    its k nearest same-class training images, and use the top singular vectors of
                    the neighbour differences as the local tangent plane.  This is the
                    non-parametric stand-in for the manifold tangent classifier of Rifai et al.
                    (which obtains tangents from a contractive autoencoder); it is NOT that
                    algorithm, only the same idea of data-driven tangents.
Tangent distance: a 1-nearest-neighbour classifier whose distance is the one-sided tangent
                 distance  min_a || q - (p + T a) ||  to each prototype's tangent plane; reported
                 next to Euclidean 1-NN on the same prototypes as a diagnostic (validation set)."""
import torch
import torch.nn.functional as F

from ..data import N_CLASSES, stratified_subset, to_float
from ..evaluate import auc_macro_ovr
from ..transforms import affine_tangents
from .base import Method, TrainSet

N_TANGENTS = 5
TP_BATCH = 32          # examples per step that receive the tangent penalty (cost control)
K_NEIGHBOURS = 10
POOL = 5000


def manifold_tangents(x, y, pool, pool_y, k=K_NEIGHBOURS, d=N_TANGENTS):
    """Local-PCA tangent vectors [B, d, 784] from the k nearest same-class pool images."""
    dist = torch.cdist(x, pool)
    dist = dist.masked_fill(pool_y[None, :] != y[:, None], float("inf"))
    idx = dist.topk(k + 1, largest=False).indices[:, 1:]          # drop the nearest (usually x itself)
    diffs = pool[idx] - x[:, None, :]
    _, _, vh = torch.linalg.svd(diffs, full_matrices=False)
    return vh[:, :d, :]


def tangent_distance_scores(q, protos, proto_y, tau_frac=0.1, chunk=500):
    """Class scores from the one-sided tangent distance of queries q [B,784] to prototypes."""
    t = affine_tangents(protos)                                    # [N,K,784]
    qm, _ = torch.linalg.qr(t.transpose(1, 2))                     # orthonormal basis, [N,784,K]
    p_proj = torch.einsum("nd,ndk->nk", protos, qm)
    d_tan, d_euc = [], []
    for i in range(0, len(q), chunk):
        qb = q[i : i + chunk]
        e2 = torch.cdist(qb, protos) ** 2
        coef = torch.einsum("bd,ndk->bnk", qb, qm) - p_proj[None]
        d_tan.append((e2 - (coef ** 2).sum(-1)).clamp_min(0))
        d_euc.append(e2)
    out = []
    for d in (torch.cat(d_tan), torch.cat(d_euc)):
        cls = torch.stack([d[:, proto_y == c].min(1).values for c in range(N_CLASSES)], dim=1)
        out.append(F.softmax(-cls / (tau_frac * cls.mean()), dim=1))
    return out


class Tangent(Method):
    name = "tangent"
    title = "Tangent prop / tangent distance / manifold tangents"
    section = "7.14"

    def defaults(self):
        return {"lr": 0.05, "lam": 0.1, "source": "affine"}

    def space(self, trial):
        return {"lam": trial.suggest_float("lam", 1e-3, 3.0, log=True),
                "source": trial.suggest_categorical("source", ["affine", "manifold"])}

    def prepare(self, data, hp, ctx):
        idx = stratified_subset(data.y_train, POOL, ctx.seed)
        self._pool = to_float(data.x_train[idx]).to(ctx.device)
        self._pool_y = data.y_train[idx].to(ctx.device)
        return super().prepare(data, hp, ctx)

    def penalty(self, model, batch, hp):
        x, y = batch["x"][:TP_BATCH], batch["y"][:TP_BATCH]
        with torch.no_grad():
            v = affine_tangents(x) if hp["source"] == "affine" else \
                manifold_tangents(x, y, self._pool, self._pool_y)
        k = v.shape[1]
        xr = x.repeat_interleave(k, dim=0)
        _, jv = torch.func.jvp(lambda z: F.softmax(model(z), dim=-1), (xr,), (v.reshape(-1, v.shape[-1]),))
        return hp["lam"] * jv.pow(2).sum(-1).mean()

    @torch.no_grad()
    def extras(self, result, data, ctx, hp):
        n = int(self.settings(ctx).get("n_prototypes", 2000))
        idx = stratified_subset(data.y_train, n, ctx.seed)
        protos, py = to_float(data.x_train[idx]), data.y_train[idx]
        q, yq = to_float(data.x_val), data.y_val
        p_tan, p_euc = tangent_distance_scores(q, protos, py)
        out = {"n_prototypes": len(py), "tangent_source": hp["source"]}
        for name, p in (("tangent_distance_1nn", p_tan), ("euclidean_1nn", p_euc)):
            out[f"{name}_val_acc"] = float((p.argmax(1) == yq).float().mean())
            out[f"{name}_val_auc"] = auc_macro_ovr(yq.numpy(), p.numpy())
        return out
