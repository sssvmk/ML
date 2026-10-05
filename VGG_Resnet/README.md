# CIFAR classifiers (ResNet-20/32/56, VGG): training, MLflow tracking, registry, inference

Models: `resnet20`, `resnet32`, `resnet56` (also `resnet44`, `resnet110`; CIFAR ResNets from He et al., 2015) and
`vgg11/13/16/19`. Datasets: `cifar10` (default), `cifar100`. The family is inferred from `model.arch`.

```
pip install -r requirements.txt                    # CPU laptop: first pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python run.py benchmark --set training.device=cpu  # time a few steps; projects minutes/epoch and hours/run
python run.py train                                # default config/train.yaml = ResNet-20 on CIFAR-10
python run.py train --config config/resnet56_cifar10.yaml
python run.py train --config config/vgg16_cifar100.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db  # browse runs at http://127.0.0.1:5000
python run.py test --run-id <RUN_ID> [--config ...]      # test set, once (a second attempt is refused)
python run.py register --run-id <RUN_ID> [--config ...]  # package, register, reload in a fresh process, alias `candidate`
python run.py promote --version 1 --approver "<name>" [--config ...]   # gates, then alias `champion`
python run.py predict --model-uri models:/resnet20-cifar10@champion --image cat.png
uvicorn src.api:app --port 8000                    # POST /predict (multipart image), GET /health
```
Pass the same `--config` to `test`, `register` and `promote` that you used for `train`: it decides the registry name
(`<arch>-<dataset>`, e.g. `resnet56-cifar10`; override with `REGISTERED_MODEL_NAME`). To serve another model set
`model_name` in `config/serve.yaml` or the `REGISTERED_MODEL_NAME` / `MODEL_URI` environment variables.

Configs: `train.yaml` (= `resnet20_cifar10.yaml`), `resnet32_cifar10.yaml`, `resnet56_cifar10.yaml`, `vgg16_cifar100.yaml`.
Any value can be overridden: `--set model.arch=resnet32 training.max_epochs=60 data.dataset=cifar100`.

## What a run produces
MLflow params (whole config), per-epoch metrics, tags (git commit, data version, device), `metrics.csv`, `curves.png`,
`diagnosis.json` (underfit / overfit verdict, a hypothesis to check), `sanity.json` (initial loss ~ ln K and
tiny-batch overfit), `worst_errors/` (most confidently wrong validation images), `reference_stats.json` (for drift
monitoring), `checkpoints/best.pt`. Local copies in `reports/runs/<run_id>/`.

## Model contract
MLflow pyfunc: input `uint8` array `(N, 32, 32, 3)`, output `float32` probabilities `(N, K)` with K = 10 or 100.
Normalisation statistics are stored inside the checkpoint, so serving always matches the dataset the model was
trained on. `src/serve.py` resizes arbitrary images to 32x32 and returns top-k labels. These models only know the
classes of their training set (CIFAR-10: airplane ... truck).

## Promotion gates (`src/promote.py`)
Test set used exactly once; target met (or a written reason to accept the gap); load-and-predict verified in a fresh
process; sanity checks passed; not worse than the current champion on validation; model card exists; approver named.
Each attempt is written to `approvals/`. Rollback: move the `champion` alias back to the previous version, e.g.
`python -c "from mlflow.tracking import MlflowClient as C; C().set_registered_model_alias('resnet20-cifar10','champion','<prev_version>')"`
(set MLFLOW_TRACKING_URI first if you use a server).

## Using the AMD Radeon 780M (Ryzen 7 PRO 7840U) on Windows
1. Check the prerequisites AMD lists for the ROCm 10.0.0 PyTorch wheels: Windows 11 25H2 (`winver`), the current
   AMD Software: Adrenalin Edition driver, Python 3.11-3.14, and Microsoft Defender Application Guard and Smart App
   Control switched off. On an IT-managed laptop, get IT's approval before changing security settings or drivers.
2. In PowerShell from this folder: `powershell -ExecutionPolicy Bypass -File scripts/setup_windows_rocm.ps1`.
   Behind a TLS-inspecting proxy, if pip reports SSL errors: `pip config set global.cert <corporate-CA-bundle.pem>`.
3. `python run.py gpu-check` prints whether the GPU is visible, whether GPU math matches the CPU, and which of
   fp32 / fp16 / channels-last is fastest, plus the exact `--set ...` overrides to use. It also writes
   `reports/gpu_check.json`.
4. Train with the recommended overrides, e.g. `python run.py train --set training.device=cuda training.amp=true`.
   (ROCm builds of PyTorch use the name `cuda` for AMD GPUs.)
The first GPU steps can be slow while kernels compile (`first_step_s` in the check); judge speed by `sec_per_step`.
The 780M shares system memory and bandwidth with the CPU, so expect a moderate speedup, not discrete-GPU speed.

CPU laptop presets (override without editing files): `--set training.max_epochs=20`, `--set data.num_workers=2`,
`--set model.width_mult=0.5` (about 4x less compute for ResNets and VGG; changes the architecture, so note it when
comparing with paper numbers).

## Tests
`pytest -q` (about 1 minute on CPU; synthetic data plus a tiny fake CIFAR-10 in the on-disk format, no download).
