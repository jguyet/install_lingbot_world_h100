#!/usr/bin/env bash
set -euo pipefail

# ==========
# Config
# ==========
REPO_DIR="/lingbot-world"
VOL="/workspace/runpod-volume"
MODELS_DIR="$VOL/lingbot-world/models"
MODEL_REPO="robbyant/lingbot-world-base-cam"
MODEL_LOCAL_DIR="$MODELS_DIR/lingbot-world-base-cam"

# Optional: set HF_TOKEN in env if the model is gated/private
# export HF_TOKEN="hf_xxx"
# Optional: disable XET (recommended)
export HF_HUB_DISABLE_XET=1

echo "==> Sanity checks"
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found"; exit 1; }
nvidia-smi || true
python --version || true

echo "==> System deps"
apt-get update
apt-get install -y git ninja-build build-essential

echo "==> Clone repo"
if [[ ! -d "$REPO_DIR" ]]; then
  cd /
  git clone https://github.com/robbyant/lingbot-world.git
else
  echo "Repo already exists at $REPO_DIR"
fi

cd "$REPO_DIR"

echo "==> Create venv"
if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrade pip tooling"
pip install -U pip wheel setuptools packaging

echo "==> Install PyTorch (stable, compatible with flash-attn)"
pip uninstall -y torch torchvision torchaudio triton flash-attn >/dev/null 2>&1 || true
pip cache purge || true
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "==> Verify CUDA works"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(256,256, device="cuda")
print("cuda ok:", x.mean().item())
PY

echo "==> Install Python requirements (excluding flash_attn)"
# Ensure requirements won't try to reinstall flash_attn mid-way
sed -i '/flash_attn/d' requirements.txt
pip install -r requirements.txt
pip install -U einops safetensors sentencepiece huggingface_hub

echo "==> Configure persistent models directory on volume"
mkdir -p "$MODELS_DIR"
rm -rf models
ln -s "$MODELS_DIR" ./models

echo "==> Verify models symlink target FS"
readlink -f ./models
df -h ./models || true

echo "==> Install flash-attn (must match torch ABI)"
# Important: install immediately after torch in the same venv.
pip install flash-attn --no-build-isolation

echo "==> Verify flash-attn kernel"
python - <<'PY'
import torch
from flash_attn.flash_attn_interface import flash_attn_func
q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
out = flash_attn_func(q, k, v)
print("flash-attn kernel OK", out.mean().item())
PY

echo "==> Download model snapshot to persistent volume"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN", None)
snapshot_download(
    repo_id=os.environ.get("MODEL_REPO", "robbyant/lingbot-world-base-cam"),
    local_dir=os.environ.get("MODEL_LOCAL_DIR"),
    token=token,
)
print("download done")
PY

echo "==> DONE. Example run (safe preset):"
cat <<'EOF'

source /lingbot-world/.venv/bin/activate

python /lingbot-world/generate.py \
  --task i2v-A14B \
  --size 480*832 \
  --ckpt_dir /lingbot-world/models/lingbot-world-base-cam \
  --image /lingbot-world/examples/00/image.jpg \
  --frame_num 81 \
  --prompt "A cinematic forest path at sunrise, soft light, high detail"

EOF
