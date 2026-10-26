import os

import pandas as pd
import pytest


def test_http_service(smoke_run, csv_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from aeclf.serving import app as m

    m.load_model(str(smoke_run / "candidates" / "ae_finetune" / "bundle"))
    rows = pd.read_csv(csv_path).drop(columns=["id", "satisfaction"]).head(3).where(lambda d: d.notna(), None).to_dict("records")
    with TestClient(m.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert len(c.post("/predict", json={"records": rows}).json()["predictions"]) == 3
        z = c.post("/embed", json={"records": rows}).json()["embeddings"]
        assert len(z) == 3 and len(z[0]) == c.get("/model-info").json()["latent_dim"]
        r = c.post("/reconstruct", json={"records": rows}).json()
        assert len(r["reconstruction"]) == 3 and len(r["reconstruction_error"]) == 3
        assert c.post("/predict", json={"records": [{k: v for k, v in rows[0].items() if k != "Age"}]}).status_code == 422
    m.load_model(str(smoke_run / "candidates" / "ae_lgbm_hybrid" / "bundle"))
    with TestClient(m.app) as c:
        assert len(c.post("/predict", json={"records": rows}).json()["predictions"]) == 3
        assert c.get("/model-info").json()["head_type"] == "lgbm_hybrid" and len(c.post("/embed", json={"records": rows}).json()["embeddings"]) == 3
    m.load_model(str(smoke_run / "candidates" / "scratch_mlp" / "bundle"))
    with TestClient(m.app) as c:
        assert c.post("/embed", json={"records": rows}).status_code == 409        # control has no pretrained autoencoder


def test_agent_tools_facts(smoke_run):
    from aeclf.agent.tools import RunArtifacts

    a = RunArtifacts(smoke_run)
    assert a.autoencoder("ae")["sanity_checks"] and a.candidate("ae_frozen")["loss"]
    with pytest.raises(KeyError):
        a.candidate("nope")


def test_ag2_tool_loop_with_mock_llm(smoke_run, fast_cfg):
    pytest.importorskip("autogen")
    import mock_llm_server

    from aeclf.agent.ag2_explainer import explain_run

    server, port = mock_llm_server.start()
    os.environ.update(LLM_MODEL="mock", LLM_API_KEY="k", LLM_BASE_URL=f"http://127.0.0.1:{port}/v1")
    try:
        out = explain_run(smoke_run, fast_cfg, mode="llm")
    finally:
        server.shutdown()
    t = (out / "autoencoder_ae.md").read_text()
    assert "Independent audit" in t and "VERDICT: VERIFIED" in t and any(c.get("tools") for c in mock_llm_server.Handler.calls)


def test_agent_falls_back_offline(smoke_run, fast_cfg, monkeypatch):
    from aeclf.agent.ag2_explainer import explain_run

    for v in ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    assert "Mode: **offline**" in (explain_run(smoke_run, fast_cfg, mode="llm") / "index.md").read_text()
