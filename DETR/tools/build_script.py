#!/usr/bin/env python3
"""Generate the single-file script dist/detr_pipeline.py from the package (one source of truth)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ["config", "env", "data_download", "ops", "model", "infer", "data", "loss", "metrics", "ema", "schedule", "curves",
           "diagnose", "train", "evaluate", "serve", "pipeline"]
OUT = ROOT / "dist" / "detr_pipeline.py"


def strip_package_imports(src: str) -> str:
    tree = ast.parse(src)
    lines = src.splitlines()
    drop = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("detr_voc"):
            drop.update(range(node.lineno - 1, node.end_lineno))
    out = "\n".join(ln for i, ln in enumerate(lines) if i not in drop).strip("\n") + "\n"
    for node in ast.walk(ast.parse(out)):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and "detr_voc" in (getattr(node, "module", None) or " ".join(a.name for a in node.names)):
            raise SystemExit(f"nested package import on line {node.lineno}: move it to the top of the module")
    return out


def blocks() -> list[tuple[str, str]]:
    seen, out = {}, []
    for m in MODULES:
        src = strip_package_imports((ROOT / "detr_voc" / f"{m}.py").read_text())
        for node in ast.parse(src).body:
            names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else \
                [t.id for a in [node] if isinstance(a, ast.Assign) for t in a.targets if isinstance(t, ast.Name)]
            for n in names:
                if n in seen and n != "log":
                    raise SystemExit(f"name clash in the single-file build: {n} defined in {seen[n]} and {m}")
                seen[n] = m
        out.append((m, src))
    return out


def write_script(out: Path = OUT) -> None:
    parts = ['#!/usr/bin/env python3\n"""DETR (Carion et al., ECCV 2020) with an ImageNet-pretrained ResNet-50: the complete pipeline in one file.\n\n'
             "  python detr_pipeline.py PATH [cpu|gpu]\n\nPATH holds everything (data, caches, runs, models, MLflow). The device is asked for when omitted.\n"
             'Generated from the package by tools/build_script.py; do not edit.\n"""\n']
    for m, src in blocks():
        parts.append(f"\n# {'=' * 100}\n# {m}.py\n# {'=' * 100}\n{src}")
    parts.append('\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts))


if __name__ == "__main__":
    write_script()
    print("written:", OUT)
