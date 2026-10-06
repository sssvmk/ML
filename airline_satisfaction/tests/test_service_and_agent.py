import os

import pandas as pd
import pytest


def test_http_service(smoke_run, csv_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from airsat.serving import app as app_module

    app_module.load_model(str(smoke_run / "champion"))
    with TestClient(app_module.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        rows = pd.read_csv(csv_path).drop(columns=["id", "satisfaction"]).head(3).where(lambda d: d.notna(), None).to_dict("records")
        r = client.post("/predict", json={"records": rows})
        assert r.status_code == 200 and len(r.json()["predictions"]) == 3
        bad = [{k: v for k, v in rows[0].items() if k != "Age"}]
        assert client.post("/predict", json={"records": bad}).status_code == 422
        assert "required_columns" in client.get("/model-info").json()


def test_agent_tools_return_facts_only(smoke_run):
    from airsat.agent.tools import RunArtifacts

    art = RunArtifacts(smoke_run)
    names = [a["name"] for a in art.algorithms()]
    assert "logistic_regression_elasticnet" in names
    rep = art.algorithm("logistic_regression_elasticnet")
    assert rep["assumption_checks"] and rep["loss_function"]
    assert "unknown term" in art.tool_glossary("nonsense")
    with pytest.raises(KeyError):
        art.algorithm("does_not_exist")


def test_ag2_tool_calling_loop_with_mock_llm(smoke_run, fast_cfg):
    pytest.importorskip("autogen")
    import mock_llm_server

    from airsat.agent.ag2_explainer import explain_run

    server, port = mock_llm_server.start()
    os.environ.update(LLM_MODEL="mock", LLM_API_KEY="k", LLM_BASE_URL=f"http://127.0.0.1:{port}/v1")
    try:
        out = explain_run(smoke_run, fast_cfg, mode="llm")
    finally:
        server.shutdown()
    text = (out / "logistic_regression_elasticnet.md").read_text()
    assert "Independent audit" in text and "VERDICT: VERIFIED" in text
    assert any(c.get("tools") for c in mock_llm_server.Handler.calls)      # tools were offered to the LLM


def test_agent_falls_back_offline_without_credentials(smoke_run, fast_cfg, monkeypatch):
    from airsat.agent.ag2_explainer import explain_run

    for v in ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    out = explain_run(smoke_run, fast_cfg, mode="llm")
    assert "Mode: **offline**" in (out / "index.md").read_text()
