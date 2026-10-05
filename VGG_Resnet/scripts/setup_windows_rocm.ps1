# Windows 11 + AMD Radeon 780M (gfx1103): create a venv with the ROCm build of PyTorch, then verify the GPU.
# Run from the project folder in PowerShell. Prerequisites (see README): Windows 11 25H2, current AMD Software:
# Adrenalin Edition driver, Python 3.11-3.14, Defender Application Guard and Smart App Control turned off.
$ErrorActionPreference = "Stop"
$py = "3.12"
py -$py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# ROCm build of PyTorch for gfx1103 from AMD's index. If pip cannot find 2.13.0 for Windows, use the 2.12.0 pair
# below instead (the selector on AMD's "Install PyTorch for ROCm" page is the authority).
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1103]==2.13.0+rocm10.0.0" "torchvision[device-gfx1103]==0.28.0+rocm10.0.0"
# python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1103]==2.12.0+rocm10.0.0" "torchvision[device-gfx1103]==0.27.0+rocm10.0.0"

# Everything else from PyPI (does not touch torch)
python -m pip install -r requirements-base.txt

python -c "import torch; print('torch', torch.__version__, '| hip', torch.version.hip, '| gpu visible:', torch.cuda.is_available())"
python run.py gpu-check
