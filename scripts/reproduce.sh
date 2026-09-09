#!/usr/bin/env bash
# scripts/reproduce.sh
# 一键冒烟测试：在内置 test_set (3 条英文样本) 上跑完整流水线。
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

print_step() { printf "\n=== %s ===\n" "$*"; }

# 将缓存重定向到项目内，方便在只读 root 沙箱中运行
export XDG_CACHE_HOME="$PROJ_ROOT/.cache"
export HF_HOME="$PROJ_ROOT/.cache/huggingface"
export WHISPER_DOWNLOAD_ROOT="$PROJ_ROOT/.cache/whisper"
export UTMOSV2_CHACHE="$PROJ_ROOT/versa/UTMOSv2"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$WHISPER_DOWNLOAD_ROOT" "$UTMOSV2_CHACHE/models/fusion_stage3"

print_step "Step 1: python3 -m venv .venv"
if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
  echo "  .venv already exists, skipping."
else
  python3 -m venv .venv
fi
source .venv/bin/activate
PY="$(command -v python)"
echo "  python -> $PY ($("$PY" --version))"

print_step "Step 2: install requirements.txt"
if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  pip install --upgrade pip
  pip install -r requirements.txt
fi

print_step "Step 3: install versa (editable, idempotent)"
( cd versa && pip install -e . --no-build-isolation )

print_step "Step 4: install UTMOSv2"
( cd versa && bash tools/install_utmosv2.sh )

print_step "Step 5: download espnet/voxcelebs12_rawnet3 checkpoint"
"$PY" - <<'PY_EOF'
import os
os.environ.setdefault('HF_HOME', os.path.join(os.environ['XDG_CACHE_HOME'], 'huggingface'))
from huggingface_hub import snapshot_download
snapshot_download('espnet/voxcelebs12_rawnet3',
                  local_dir='checkpoints/voxcelebs12_rawnet3')
PY_EOF

print_step "Step 6: running scripts/run.sh --TTS_Models test_tts --config configs/test_set/config.yaml --stage 2-6"
bash scripts/run.sh \
  --TTS_Models test_tts \
  --config configs/test_set/config.yaml \
  --stage 2-6

print_step "Done."
SUMMARY="results/test_set/test_tts/score_summary.csv"
if [ -f "$SUMMARY" ]; then
  echo "----- $SUMMARY -----"
  cat "$SUMMARY"
fi
