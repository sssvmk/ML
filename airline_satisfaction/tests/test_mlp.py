import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airsat.algorithms.pytorch_residual_mlp import Algorithm, TorchResidualMLPClassifier  # noqa: E402


def _toy(n=800, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = (X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y


def test_initial_loss_and_tiny_batch_overfit():
    X, y = _toy()
    est = TorchResidualMLPClassifier(seed=0)
    assert 0.5 < est.initial_loss(X, y) < 1.0
    assert est.overfit_tiny_batch(X[:64], y[:64]) < 0.05


def test_fit_predict_and_pickle_roundtrip():
    import joblib, io
    X, y = _toy()
    est = TorchResidualMLPClassifier(hidden=32, n_blocks=1, max_epochs=5, batch_size=128, seed=0).fit(X[:600], y[:600], X[600:], y[600:])
    p = est.predict_proba(X[600:])
    assert p.shape == (200, 2) and np.allclose(p.sum(axis=1), 1)
    buf = io.BytesIO(); joblib.dump(est, buf); buf.seek(0)
    assert np.allclose(joblib.load(buf).predict_proba(X[600:]), p, atol=1e-6)
    curve = Algorithm().training_curve(est)
    assert {"iteration", "train_loss", "val_loss"} <= set(curve.columns)
