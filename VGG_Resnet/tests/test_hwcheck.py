import pytest
import torch

from src.hwcheck import benchmark_one, correctness_check, describe_torch, diagnose_no_gpu, gpu_check
from src.utils import ROOT, load_config, select_device


def _cfg():
    return load_config(ROOT / "config" / "train.yaml", [
        "data.batch_size=8", "model.arch=vgg11", "model.width_mult=0.125", "model.head_hidden=32"])


def test_select_device_cuda_without_gpu_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="gpu-check"):
        select_device("cuda")
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in ("cpu", "mps")


def test_diagnose_no_gpu_messages():
    cpu_only = {"gpu_available": False, "hip_build": None, "cuda_build": None}
    assert "CPU-only" in diagnose_no_gpu(cpu_only)[0]
    assert "NVIDIA" in diagnose_no_gpu({"gpu_available": False, "hip_build": None, "cuda_build": "12.8"})[0]
    assert "driver" in diagnose_no_gpu({"gpu_available": False, "hip_build": "7.1", "cuda_build": None})[0]
    assert diagnose_no_gpu({"gpu_available": True, "hip_build": "7.1", "cuda_build": None}) == []


def test_correctness_check_skips_on_cpu():
    assert "skipped" in correctness_check(torch.device("cpu"))


def test_benchmark_one_cpu_and_amp_flag_ignored_on_cpu():
    r = benchmark_one(_cfg(), torch.device("cpu"), amp=True, channels_last=True, steps=2, warmup=1)
    assert r["amp"] is False and r["channels_last"] is False
    assert r["sec_per_step"] > 0 and r["loss_finite"] and r["est_epoch_minutes"] > 0


def test_gpu_check_end_to_end_on_cpu_only_machine():
    rep = gpu_check(_cfg(), steps=2)
    assert describe_torch()["gpu_available"] is False  # sandbox has no GPU
    assert rep["advice"] and rep["benchmarks"] and rep["recommended"]["device"] == "cpu"
    assert "training.device=cpu" in rep["recommended"]["overrides"]
