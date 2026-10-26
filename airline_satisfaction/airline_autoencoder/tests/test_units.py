import numpy as np
import pytest
import torch

from aeclf.config import load_config
from aeclf.encoding import TabularEncoder
from aeclf.io_utils import load_training_data
from aeclf.models import MixedAutoencoder, SatisfactionClassifier, corrupt, kl_divergence, reconstruction_loss
from aeclf.schema import infer_schema
from aeclf.splitting import make_splits
from aeclf.training import pre_training_checks, to_tensors, train_autoencoder, reconstruction_report


@pytest.fixture(scope="module")
def prepared(csv_path):
    cfg = load_config()
    X, y, _, _ = load_training_data(csv_path, cfg)
    schema = infer_schema(X, cfg)
    enc = TabularEncoder(schema).fit(X)
    return cfg, X, y, schema, enc


def test_schema_roles(prepared):
    cfg, X, y, s, enc = prepared
    assert len(s.ordinal) == 13 and len(s.nominal) == 4 and len(s.continuous) == 4


def test_encoder_shapes_unknown_category_and_log_transform(prepared):
    cfg, X, y, s, enc = prepared
    xc, xk = enc.transform(X)
    assert xc.shape == (len(X), 4) and xk.shape == (len(X), 17) and np.isfinite(xc).all()
    assert all((xk[:, j] < c).all() for j, c in enumerate(enc.cards))
    assert enc.cont_stats["Departure Delay in Minutes"]["log"]            # skewed delays are log-transformed
    row = X.iloc[[0]].copy(); row["Class"] = "Space Class"; row["Age"] = np.nan
    xc2, xk2 = enc.transform(row)
    assert xk2[0, enc.cat_cols.index("Class")] == enc.cards[enc.cat_cols.index("Class")] - 1 and np.isfinite(xc2).all()


def test_encoder_rating_zero_is_its_own_class(prepared):
    cfg, X, y, s, enc = prepared
    assert "0" in enc.levels["Inflight wifi service"] and "Inflight wifi service" in enc.ordinal_cats


def test_inverse_cont_roundtrip(prepared):
    cfg, X, y, s, enc = prepared
    xc, _ = enc.transform(X.head(200))
    back = enc.inverse_cont(xc)
    assert np.allclose(back["Age"], X["Age"].head(200), atol=1e-3)
    assert np.allclose(back["Flight Distance"], X["Flight Distance"].head(200), rtol=1e-3, atol=1e-2)


def test_split_disjoint_stratified(prepared):
    cfg, X, y, s, enc = prepared
    sp = make_splits(X, y, cfg)
    idx = np.concatenate(list(sp.values()))
    assert len(idx) == len(set(idx)) == len(X)
    assert abs(y.iloc[sp["test"]].mean() - y.mean()) < 0.03


@pytest.mark.parametrize("kind", ["ae", "dae", "vae"])
def test_models_forward_loss_and_variants(kind):
    m = MixedAutoencoder(4, [6, 6, 3, 5], kind)
    xc, xk = torch.randn(16, 4), torch.randint(0, 3, (16, 4))
    out = m(xc, xk)
    loss, c, k = reconstruction_loss(out, xc, xk)
    assert torch.isfinite(loss) and (out["logvar"] is not None) == (kind == "vae")
    xc2, xk2 = corrupt(xc, xk, 0.2, 1.0, torch.tensor([5, 5, 2, 4]))
    assert (xk2 == torch.tensor([5, 5, 2, 4])).all() and not torch.equal(xc, xc2)
    assert SatisfactionClassifier(m)(xc, xk).shape == (16,)
    if kind == "vae":
        assert kl_divergence(out["mu"], out["logvar"]) >= 0


def test_autoencoder_learns_and_beats_trivial_reconstruction(prepared):
    cfg, X, y, s, enc = prepared
    T = to_tensors(enc, X, y)
    tr, va = T.take(range(3000)), T.take(range(3000, 4000))
    arch = {"n_cont": 4, "cards": enc.cards, "emb_max": 8}
    P = dict(cfg["autoencoder"]["defaults"])
    assert not any(c["blocking"] for c in pre_training_checks("dae", arch, P, tr, cfg, 0))
    model, hist, best = train_autoencoder("dae", P, arch, tr, va, cfg, 0, "cpu", max_epochs=15)
    assert hist["val_loss"].iloc[-1] < hist["val_loss"].iloc[0]
    rep, ex = reconstruction_report(model, enc, X.iloc[3000:], "cpu", 5)
    assert rep["summary"]["mean_continuous_r2"] > 0.2 and ex.shape[0] == 5


def test_missing_lightgbm_gives_actionable_error(monkeypatch):
    import numpy as np
    from aeclf import lgbm_head
    monkeypatch.setattr(lgbm_head, "lgb", None)
    with pytest.raises(ImportError, match="pip install lightgbm"):
        lgbm_head.train_lgbm({}, np.zeros((10, 2)), np.zeros(10), 0)
