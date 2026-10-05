"""Hardware check and benchmark: is the GPU visible, does it compute correctly, and which settings are fastest?"""
from __future__ import annotations

import copy
import json
import time

import torch
import torch.nn as nn

from src.model import build_model, count_params
from src.utils import ROOT, env_info, select_device


def describe_torch() -> dict:
    info = {
        "torch": torch.__version__,
        "hip_build": getattr(torch.version, "hip", None),
        "cuda_build": torch.version.cuda,
        "gpu_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu_name": p.name, "gpu_arch": getattr(p, "gcnArchName", None),
                     "gpu_memory_gb": round(p.total_memory / 1e9, 2)})
    return info


def diagnose_no_gpu(info: dict) -> list[str]:
    if info["gpu_available"]:
        return []
    msgs = []
    if info["hip_build"] is None and info["cuda_build"] is None:
        msgs.append("This is a CPU-only PyTorch build (no HIP or CUDA). Install the ROCm build for your AMD GPU "
                    "(scripts/setup_windows_rocm.ps1 on Windows).")
    elif info["cuda_build"]:
        msgs.append("This is an NVIDIA CUDA build, which cannot drive an AMD GPU. Install the ROCm build instead.")
    else:
        msgs.append("ROCm build installed, but no GPU is visible: check the AMD driver, Windows 11 25H2, "
                    "and the Windows prerequisites in the README (Application Guard / Smart App Control).")
    return msgs


@torch.no_grad()
def _flatten_grads(model):
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}


def correctness_check(device: torch.device, seed: int = 0) -> dict:
    """Same weights, same batch: forward loss and gradients on `device` must match the CPU (fp32)."""
    if device.type == "cpu":
        return {"skipped": "device is cpu"}
    mcfg = {"arch": "vgg11", "batch_norm": True, "width_mult": 0.25, "head_hidden": 64, "dropout": 0.0}
    torch.manual_seed(seed)
    cpu_model = build_model(mcfg, 100).train()
    dev_model = copy.deepcopy(cpu_model).to(device).train()
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 100, (16,))
    out = {}
    for tag, m, xx, yy in (("cpu", cpu_model, x, y), ("dev", dev_model, x.to(device), y.to(device))):
        m.zero_grad()
        loss = nn.functional.cross_entropy(m(xx), yy)
        loss.backward()
        out[tag] = (loss.item(), _flatten_grads(m))
    loss_diff = abs(out["cpu"][0] - out["dev"][0])
    worst = 0.0
    for name, g in out["cpu"][1].items():
        d = (g - out["dev"][1][name]).abs().max().item() / (g.abs().max().item() + 1e-8)
        worst = max(worst, d)
    ok = loss_diff < 1e-2 and worst < 5e-2
    return {"loss_abs_diff": loss_diff, "worst_relative_grad_diff": worst, "passed": bool(ok)}


def benchmark_one(cfg: dict, device: torch.device, amp: bool, channels_last: bool, steps: int = 6,
                  warmup: int = 3) -> dict:
    k = cfg["data"]["num_classes"]
    bs = cfg["data"]["batch_size"]
    amp = bool(amp and device.type == "cuda")
    channels_last = bool(channels_last and device.type == "cuda")
    row = {"device": device.type, "amp": amp, "channels_last": channels_last, "batch_size": bs}
    try:
        torch.manual_seed(cfg["seed"])
        model = build_model(cfg["model"], k).to(device).train()
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda") if amp else None
        x = torch.randn(bs, 3, 32, 32, device=device)
        y = torch.randint(0, k, (bs,), device=device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = nn.functional.cross_entropy(model(x), y)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            return loss

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        t0 = time.time(); step(); sync()
        row["first_step_s"] = round(time.time() - t0, 2)  # includes kernel compilation / autotuning on GPUs
        for _ in range(max(0, warmup - 1)):
            step()
        sync()
        t0 = time.time()
        for _ in range(steps):
            loss = step()
        sync()
        sec = (time.time() - t0) / steps
        row.update({"sec_per_step": round(sec, 3), "loss_finite": bool(torch.isfinite(loss).item()),
                    "params": count_params(model)})
        n_train = int(50000 * (1 - cfg["data"]["val_fraction"]))
        epoch_s = sec * (n_train // bs)
        row["est_epoch_minutes"] = round(epoch_s / 60, 1)
        row["est_full_run_hours"] = round(epoch_s * cfg["training"]["max_epochs"] / 3600, 2)
        if device.type == "cuda":
            row["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except RuntimeError as e:
        row["error"] = ("out of memory: lower data.batch_size" if "out of memory" in str(e).lower()
                        else str(e)[:200])
    return row


def gpu_check(cfg: dict, steps: int = 6, include_cpu: bool = True) -> dict:
    info = describe_torch()
    report = {"torch": info, "advice": diagnose_no_gpu(info), "benchmarks": []}
    device = select_device("cuda" if info["gpu_available"] else "cpu")
    print(json.dumps(info, indent=2))
    for m in report["advice"]:
        print("ADVICE:", m)
    if device.type == "cuda":
        report["correctness_vs_cpu"] = correctness_check(device)
        print("correctness vs CPU:", report["correctness_vs_cpu"])
        for amp, cl in ((False, False), (True, False), (False, True), (True, True)):
            r = benchmark_one(cfg, device, amp, cl, steps)
            report["benchmarks"].append(r); print(r)
    if include_cpu:
        r = benchmark_one(cfg, torch.device("cpu"), False, False, max(2, steps // 2), warmup=2)
        report["benchmarks"].append(r); print(r)
    ok = [b for b in report["benchmarks"] if "sec_per_step" in b and b.get("loss_finite")]
    if device.type == "cuda" and report["correctness_vs_cpu"].get("passed") is False:
        ok = [b for b in ok if b["device"] == "cpu"]
        report["advice"].append("GPU results differ from the CPU beyond tolerance: do NOT train on this GPU build.")
    if ok:
        best = min(ok, key=lambda b: b["sec_per_step"])
        cpu = next((b for b in ok if b["device"] == "cpu"), None)
        report["recommended"] = {
            **best,
            "overrides": f"--set training.device={best['device']} training.amp={str(best['amp']).lower()} "
                         f"training.channels_last={str(best['channels_last']).lower()}",
            "speedup_vs_cpu": round(cpu["sec_per_step"] / best["sec_per_step"], 2) if cpu and best is not cpu else None,
        }
        print("RECOMMENDED:", report["recommended"]["overrides"], "| speedup vs cpu:", report["recommended"]["speedup_vs_cpu"])
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "gpu_check.json").write_text(json.dumps(report, indent=2))
    report["env"] = env_info(device)
    return report
