#!/bin/bash
# One-click environment setup for SkyRL + CodeScout on Ascend NPU.
#
# Usage:
#   bash npu_support/setup_env.sh [CONDA_BASE]
#
# Arguments:
#   CONDA_BASE  Path to miniforge/miniconda base (default: ${CONDA_BASE:-$HOME/miniforge3})
#
# This script creates a conda env "codescout-npu" with:
#   - Python 3.12
#   - torch 2.8.0 + torch_npu 2.8.0
#   - vllm 0.11.0 + vllm-ascend (copied from a working 3.11 env if available)
#   - SkyRL (editable install from local repo)
#   - All codescout training dependencies

set -euo pipefail

CONDA_BASE="${1:-${CONDA_BASE:-$HOME/miniforge3}}"
ENV_NAME="codescout-npu"
ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"
PIP="$ENV_PATH/bin/pip"
PYTHON="$ENV_PATH/bin/python"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKYRL_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== SkyRL NPU Environment Setup ==="
echo "Conda base:  $CONDA_BASE"
echo "Env name:    $ENV_NAME"
echo "SkyRL root:  $SKYRL_ROOT"
echo ""

# ---------- Step 1: Create conda env ----------
if [ -d "$ENV_PATH" ]; then
    echo "[1/6] Env $ENV_NAME already exists, skipping creation."
else
    echo "[1/6] Creating conda env $ENV_NAME with Python 3.12..."
    "$CONDA_BASE/bin/conda" create -n "$ENV_NAME" python=3.12 -y
fi

# ---------- Step 2: Install torch + torch_npu ----------
echo "[2/6] Installing torch + torch_npu..."
$PIP install torch-npu==2.8.0.post2 numpy pyyaml 2>&1 | tail -3

# ---------- Step 3: Install training dependencies ----------
echo "[3/6] Installing training dependencies..."
$PIP install \
    hydra-core==1.3.2 omegaconf accelerate peft wandb datasets \
    tensordict jaxtyping polars torchdata func_timeout \
    loguru tqdm ninja tensorboard "transformers>=4.57,<5" \
    "ray==2.51.1" debugpy==1.8.0 hf_transfer \
    gcsfs litellm docker tenacity authlib \
    s3fs pybind11 setuptools \
    2>&1 | tail -3

# ---------- Step 4: Install SkyRL + skyrl-gym ----------
echo "[4/6] Installing SkyRL (editable)..."
$PIP install -e "$SKYRL_ROOT" --no-deps --no-build-isolation 2>&1 | tail -2
$PIP install -e "$SKYRL_ROOT/skyrl-gym" --no-deps --no-build-isolation 2>&1 | tail -2

# ---------- Step 5: Install openhands SDK ----------
echo "[5/6] Installing openhands SDK..."
$PIP install openhands-tools openhands-agent-server openhands-workspace openhands-sdk --no-deps 2>&1 | tail -2
$PIP install agent-client-protocol deprecation "fakeredis[lua]" fastmcp lmnr \
    python-frontmatter sqlalchemy wsproto fastapi alembic python-json-logger \
    binaryornot aiosqlite libtmux bashlex tom-swe \
    2>&1 | tail -3

# ---------- Step 6: Install vllm + vllm-ascend ----------
echo "[6/6] Installing vllm..."
$PIP install vllm==0.11.0 --no-deps 2>&1 | tail -2

# vllm-ascend can't be pip-installed easily (build deps conflict).
# Copy from an existing working env if available.
DONOR_ENV="$CONDA_BASE/envs/codescout"
if [ -d "$DONOR_ENV/lib/python3.11/site-packages/vllm_ascend" ]; then
    echo "   Copying vllm-ascend from $DONOR_ENV..."
    cp -r "$DONOR_ENV/lib/python3.11/site-packages/vllm_ascend" \
          "$ENV_PATH/lib/python3.12/site-packages/vllm_ascend"
else
    echo "   WARNING: No donor env found for vllm-ascend."
    echo "   You'll need to install vllm-ascend manually."
fi

# ---------- Verify ----------
echo ""
echo "=== Verifying installation ==="
source /usr/local/Ascend/cann-8.5.0/bin/setenv.bash 2>/dev/null || true
$PYTHON -c "
import torch_npu, torch
print(f'  torch={torch.__version__}, torch_npu={torch_npu.__version__}')
print(f'  NPU available: {torch.npu.is_available()}, count: {torch.npu.device_count()}')
import skyrl; print(f'  skyrl={skyrl.__version__}')
import vllm; print(f'  vllm={vllm.__version__}')
try:
    import vllm_ascend; print('  vllm_ascend OK')
except: print('  vllm_ascend NOT installed')
print('  All good!')
"

echo ""
echo "=== Setup complete! ==="
echo "Activate with:  source $CONDA_BASE/bin/activate $ENV_NAME"
echo "Then source CANN: source /usr/local/Ascend/cann-8.5.0/bin/setenv.bash"
