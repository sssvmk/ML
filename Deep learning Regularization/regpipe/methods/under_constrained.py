"""7.3  Regularization and under-constrained problems.

Regime: fewer training examples (default 500) than inputs (784), so X^T X is singular and the
problem is under-determined.  Regularization is what makes it well-posed: X^T X + alpha*I is
always invertible, and the Moore-Penrose pseudo-inverse is the alpha -> 0 limit of ridge.

Reported model : the MLP trained on that small set with L2 (alpha tuned).
Diagnostics    : rank(X^T X) vs 784, closed-form ridge classifier (X^T X + aI)^-1 X^T Y on pixels,
                 the pseudo-inverse solution, their agreement as alpha -> 0, and an ablation that
                 trains the same MLP on the same examples with NO regularizer.
NOTE: metrics come from a model trained on `n_train` examples, so they are NOT comparable with
the full-data methods; the report labels this regime."""
import torch

from ..data import DIM, N_CLASSES, stratified_subset, to_float
from ..evaluate import evaluate
from .base import Method, TrainSet, weights_only


def _design(x_u8):
    x = to_float(x_u8).double()
    return torch.cat([x, torch.ones(len(x), 1, dtype=torch.float64)], dim=1)


def ridge_fit(xb, y_onehot, alpha):
    a = xb.T @ xb + alpha * torch.eye(xb.shape[1], dtype=torch.float64)
    return torch.linalg.solve(a, xb.T @ y_onehot)


class UnderConstrained(Method):
    name = "under_constrained"
    title = "Regularization and under-constrained problems"
    section = "7.3"
    champion_eligible = False

    @property
    def regime(self):
        return f"{getattr(self, '_n', 500)} training examples (< {DIM} inputs)"

    def defaults(self):
        return {"lr": 0.05, "alpha": 1e-3}

    def space(self, trial):
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True)}

    def prepare(self, data, hp, ctx):
        n = int(self.settings(ctx).get("n_train", 500))
        self._n = n
        idx = stratified_subset(data.y_train, n, ctx.seed)
        return TrainSet(data.x_train[idx], data.y_train[idx])

    def penalty(self, model, batch, hp):
        return 0.5 * hp["alpha"] * sum((w ** 2).sum() for w in weights_only(model))

    def extras(self, result, data, ctx, hp):
        from ..trainer import fit_single
        ts = self.prepare(data, hp, ctx)
        xb, y = _design(ts.x), ts.y
        yo = torch.nn.functional.one_hot(y, N_CLASSES).double()
        xv, yv = _design(data.x_val), data.y_val
        rank = int(torch.linalg.matrix_rank(xb[:, :DIM].T @ xb[:, :DIM]))
        out = {"n_train_examples": len(y), "n_inputs": DIM, "rank_XtX": rank, "XtX_is_singular": rank < DIM}
        accs = {}
        for a in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            accs[a] = ((xv @ ridge_fit(xb, yo, a)).argmax(1) == yv).double().mean().item()
        best_a = max(accs, key=accs.get)
        out.update(ridge_best_alpha=best_a, ridge_val_acc=accs[best_a])
        w_pinv = torch.linalg.pinv(xb) @ yo
        out["pinv_val_acc"] = ((xv @ w_pinv).argmax(1) == yv).double().mean().item()
        w_dual = xb.T @ torch.linalg.solve(xb @ xb.T + 1e-9 * torch.eye(len(xb), dtype=torch.float64), yo)
        out["pinv_vs_ridge_alpha0_max_abs_diff"] = float((w_dual - w_pinv).abs().max())
        base = fit_single(self, data, {**hp, "alpha": 0.0}, ctx)
        out["ablation_no_regularizer_val_acc"] = base.history[-1]["val_acc"]
        out["ablation_no_regularizer_val_loss"] = base.history[-1]["val_loss"]
        return out
