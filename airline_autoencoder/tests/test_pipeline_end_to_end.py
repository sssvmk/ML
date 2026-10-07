import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeclf.bundle import ModelBundle
from aeclf.final_evaluation import HoldoutAlreadyUsed, evaluate_on_test
from aeclf.predict import predict_file


def test_run_creates_all_artifacts(smoke_run):
    for rel in ["schema.json", "encoding.json", "selection.json", "model_comparison.md", "test_evaluation.json", "champion/weights.pt", "champion/meta.json",
                "reports/experiment_report.md", "reports/MODEL_CARD.md", "reports/label_efficiency.png", "eda/eda_summary.json", "label_efficiency.csv",
                "autoencoders/ae/reconstruction_examples.csv", "autoencoders/dae/plots/latent_pca.png", "autoencoders/ae/sanity_checks.json",
                "candidates/ae_frozen/plots/roc_pr.png", "candidates/scratch_mlp/result.json", "reports/agent_explanations/index.md"]:
        assert (smoke_run / rel).exists(), rel


def test_floor_control_and_pretraining_verdicts_present(smoke_run):
    sel = json.load(open(smoke_run / "selection.json"))
    t = {r["name"]: r for r in sel["table"]}
    assert abs(t["majority_baseline"]["val_primary"] - 0.5) < 1e-9
    assert "scratch_mlp" in t and sel["winner"] != "majority_baseline"
    assert {b["candidate"] for b in sel["pretraining_benefit"]} >= {"ae_frozen", "dae_finetune"}


def test_frozen_encoder_really_is_frozen(smoke_run):
    r = json.load(open(smoke_run / "candidates" / "ae_frozen" / "result.json"))
    f = json.load(open(smoke_run / "candidates" / "ae_finetune" / "result.json"))
    assert r["trainable_params"] < f["trainable_params"]
    b = ModelBundle.load(smoke_run / "candidates" / "ae_frozen" / "bundle")
    for k, v in b.ae_state.items():           # classifier's encoder weights == pretrained autoencoder weights
        if k.startswith(("embs", "enc", "to_mu")):
            assert torch_equal(v, b.clf_state["ae." + k]), k


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def test_test_set_is_locked_after_first_use(smoke_run, fast_cfg, csv_path):
    assert (smoke_run / "test_set_used.lock").exists()
    X = pd.read_csv(csv_path).drop(columns=["id", "satisfaction"])
    with pytest.raises(HoldoutAlreadyUsed):
        evaluate_on_test(smoke_run / "champion", X, np.zeros(len(X), dtype=int), fast_cfg, smoke_run)


def test_unseen_inference_matches_sample_format_and_extras(smoke_run, fast_cfg, csv_path, tmp_path):
    out = predict_file(str(Path(csv_path).parent / "unseen.csv"), str(smoke_run / "candidates" / "ae_finetune" / "bundle"), str(tmp_path / "p.csv"), fast_cfg, drift=True, extras=True)
    assert list(out.columns) == ["id", "satisfaction"] and len(out) == 500 and set(out["satisfaction"]) <= {"TRUE", "FALSE"}
    for suffix in (".drift.json", ".embeddings.csv", ".reconstruction_error.csv"):
        assert (tmp_path / f"p{suffix}").exists()


def test_bundle_deterministic_and_row_order_independent(smoke_run, csv_path):
    X = pd.read_csv(csv_path).drop(columns=["id", "satisfaction"]).head(200)
    b = ModelBundle.load(smoke_run / "champion")
    assert np.allclose(b.predict_proba(X), ModelBundle.load(smoke_run / "champion").predict_proba(X.iloc[::-1])[::-1], atol=1e-6)


def test_bundle_rejects_missing_columns(smoke_run, csv_path):
    from aeclf.validation import InputValidationError
    X = pd.read_csv(csv_path).drop(columns=["id", "satisfaction", "Age"]).head(5)
    with pytest.raises(InputValidationError):
        ModelBundle.load(smoke_run / "champion").predict_proba(X)


def test_lightgbm_heads_and_raw_control(smoke_run, csv_path):
    sel = json.load(open(smoke_run / "selection.json"))
    names = {r["name"] for r in sel["table"]}
    assert {"ae_lgbm_latent", "ae_lgbm_hybrid", "dae_lgbm_latent", "lgbm_raw_control"} <= names
    ben = {b["candidate"]: b["control"] for b in sel["pretraining_benefit"]}
    assert ben["ae_lgbm_latent"] == "lgbm_raw_control" and ben["ae_frozen"] == "scratch_mlp"
    X = pd.read_csv(csv_path).drop(columns=["id", "satisfaction"]).head(100)
    for n in ("ae_lgbm_latent", "ae_lgbm_hybrid", "lgbm_raw_control"):
        b = ModelBundle.load(smoke_run / "candidates" / n / "bundle")
        p = b.predict_proba(X)
        assert p.shape == (100,) and np.isfinite(p).all() and np.allclose(p, b.predict_proba(X.iloc[::-1])[::-1], atol=1e-6)
    assert (smoke_run / "candidates" / "ae_lgbm_latent" / "bundle" / "lgbm.joblib").exists()
    lat = ModelBundle.load(smoke_run / "candidates" / "ae_lgbm_latent" / "bundle")
    assert lat.embed(X).shape == (100, lat.ae_arch["latent"])
    with pytest.raises(RuntimeError):
        ModelBundle.load(smoke_run / "candidates" / "lgbm_raw_control" / "bundle").embed(X)


def test_lgbm_latent_uses_only_latent_columns(smoke_run, csv_path):
    from aeclf.lgbm_head import build_features
    from aeclf.training import to_tensors
    X = pd.read_csv(csv_path).drop(columns=["id", "satisfaction"]).head(50)
    b = ModelBundle.load(smoke_run / "candidates" / "ae_lgbm_hybrid" / "bundle")
    t = to_tensors(b.encoder, X)
    L = b.ae_arch["latent"]
    assert build_features(b.autoencoder, t, "lgbm_latent").shape[1] == L
    assert build_features(b.autoencoder, t, "lgbm_hybrid").shape[1] == L + len(b.encoder.cont_cols) + len(b.encoder.cat_cols)
    assert build_features(None, t, "lgbm_raw").shape[1] == len(b.encoder.cont_cols) + len(b.encoder.cat_cols)


def test_rerun_with_same_run_id_fails_fast_unless_overwrite(smoke_run, fast_cfg, csv_path):
    from aeclf.run_pipeline import run
    with pytest.raises(FileExistsError, match="--overwrite"):
        run(fast_cfg, str(csv_path), run_id=smoke_run.name)
