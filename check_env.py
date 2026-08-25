"""
Nelson AI — Environment Setup & Verification Script
Run this to check your environment and get fix instructions.
"""

import sys
import os
import subprocess
import platform

print("=" * 65)
print("  NELSON AI — ENVIRONMENT CHECK")
print("=" * 65)

# ── Python version ────────────────────────────────────────────────
py_ver = sys.version_info
print(f"\n  Python version : {sys.version}")
print(f"  Executable     : {sys.executable}")

if py_ver.major == 3 and py_ver.minor >= 13:
    print(f"\n  ⚠ WARNING: Python {py_ver.major}.{py_ver.minor} detected.")
    print(f"  PyTorch CUDA builds only support Python 3.9 – 3.12.")
    print(f"  You need to install Python 3.11 for GPU acceleration.")
    PYTHON_OK = False
else:
    print(f"  ✓ Python version is compatible with PyTorch CUDA.")
    PYTHON_OK = True

# ── PyTorch ───────────────────────────────────────────────────────
print(f"\n  {'─' * 40}")
try:
    import torch
    print(f"  PyTorch version : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA available  : {cuda_ok}")
    if cuda_ok:
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM            : {vram:.1f} GB")
        cap = torch.cuda.get_device_capability(0)
        print(f"  CUDA capability : {cap[0]}.{cap[1]}")
        print(f"  ✓ GPU training ready!")
    else:
        print(f"  ✗ GPU NOT available — will train on CPU (very slow)")
except ImportError:
    print(f"  ✗ PyTorch not installed")
    cuda_ok = False

# ── Other packages ────────────────────────────────────────────────
print(f"\n  {'─' * 40}")
packages = ["tokenizers", "datasets", "numpy", "requests",
            "beautifulsoup4", "tqdm", "duckduckgo_search"]
for pkg in packages:
    try:
        mod = __import__(pkg.replace("-", "_"))
        ver = getattr(mod, "__version__", "?")
        print(f"  ✓ {pkg:<25} {ver}")
    except ImportError:
        print(f"  ✗ {pkg:<25} NOT INSTALLED")

# ── Disk space ────────────────────────────────────────────────────
print(f"\n  {'─' * 40}")
try:
    import shutil
    drive = os.path.splitdrive(sys.executable)[0] or "/"
    total, used, free = shutil.disk_usage("E:\\")
    print(f"  Disk (E:) free  : {free/1e9:.1f} GB")
    if free < 20e9:
        print(f"  ⚠ Less than 20 GB free — may not be enough for training data")
    else:
        print(f"  ✓ Sufficient disk space")
except Exception as e:
    print(f"  Could not check disk: {e}")

# ── Summary & Fix Instructions ────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  DIAGNOSIS & FIX INSTRUCTIONS")
print(f"{'=' * 65}")

if not PYTHON_OK:
    print(f"""
  ❌ PROBLEM: Python {py_ver.major}.{py_ver.minor} is too new for PyTorch CUDA.

  FIX — Install Python 3.11 alongside your current Python:

  OPTION A (Recommended) — Install Python 3.11 from python.org:
    1. Download: https://www.python.org/downloads/release/python-3119/
       → Get "Windows installer (64-bit)"
    2. Install it — check "Add to PATH" during install
       OR install without PATH (to keep Python 3.14 as default)
    3. Then create a virtual environment with Python 3.11:

       py -3.11 -m venv E:\\Nelson AI\\venv
       E:\\Nelson AI\\venv\\Scripts\\activate
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
       pip install -r requirements.txt

  OPTION B — Use conda (installs its own Python):
    1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    2. Then:
       conda create -n nelson python=3.11
       conda activate nelson
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
       pip install -r requirements.txt

  After setup, ALWAYS activate the environment before running scripts:
    venv:   E:\\Nelson AI\\venv\\Scripts\\activate
    conda:  conda activate nelson
""")
elif not cuda_ok:
    print(f"""
  ❌ PROBLEM: PyTorch installed but CUDA not working.

  FIX — Reinstall PyTorch with the right CUDA version:
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

  If that doesn't work, check your NVIDIA driver:
    nvidia-smi
  Driver must be >= 520 for CUDA 12.1
""")
else:
    print(f"\n  ✅ Everything looks good! Ready to train Nelson.")
    print(f"  Run: python data_collection/download_data.py")

print(f"\n{'=' * 65}\n")
