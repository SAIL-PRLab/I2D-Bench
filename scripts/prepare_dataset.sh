#!/usr/bin/env bash
# scripts/prepare_dataset.sh
# 准备 I2D-Bench 三个评测数据集之一。
# 用法：bash scripts/prepare_dataset.sh <LibriTTS|Seed-Eval|CV3-Eval>
set -euo pipefail

DATASET="${1:-}"
case "$DATASET" in
  LibriTTS)  TARBALL="LibriTTS.tar.gz";  DEST_DIR="data/LibriTTS/test_clean"        ;;
  Seed-Eval) TARBALL="Seed-Eval.tar.gz";  DEST_DIR="data/Seed-Eval/test_zh"          ;;
  CV3-Eval)  TARBALL="CV3-Eval.tar.gz";   DEST_DIR="data/CV3-Eval/emotion_zeroshot"  ;;
  *) echo "用法：$0 <LibriTTS|Seed-Eval|CV3-Eval>"; exit 1 ;;
esac

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

export XDG_CACHE_HOME="$PROJ_ROOT/.cache"
export HF_HOME="$PROJ_ROOT/.cache/huggingface"
mkdir -p "$HF_HOME"

source .venv/bin/activate
PY="$(command -v python)"

REF_SCP="$DEST_DIR/ref_wav.scp"
PARENT_DIR="$(dirname "$DEST_DIR")"
print_step() { printf "\n=== %s ===\n" "$*"; }

# ---------- Step 1: 下载 tarball ----------
print_step "Step 1: download $TARBALL from huggingface (ssf1103/I2D-Bench)"
if [[ -s "$REF_SCP" && -s "$DEST_DIR/ref_text" && -s "$DEST_DIR/target_text" ]]; then
  echo "  $DEST_DIR/ already populated, skipping."
else
  "$PY" - "$TARBALL" <<'PYEOF'
import os, sys
os.environ.setdefault('HF_HOME', os.path.join(os.environ['XDG_CACHE_HOME'], 'huggingface'))
try:
    import hf_transfer
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
except ImportError:
    pass
from huggingface_hub import hf_hub_download
hf_hub_download("ssf1103/I2D-Bench", sys.argv[1], repo_type="dataset", local_dir=".")
PYEOF
fi

# ---------- Step 2: 解压 ----------
print_step "Step 2: extract -> $DEST_DIR/"
if [[ ! -s "$REF_SCP" ]]; then
  mkdir -p "$PARENT_DIR"
  tar -xzf "$TARBALL" -C "$PARENT_DIR/" --strip-components=1 --no-same-owner
fi
rm -f "$TARBALL"

# ---------- Step 3: 校验 ref_wav.scp ----------
print_step "Step 3: sanity-check $REF_SCP"
"$PY" - "$REF_SCP" <<'PYEOF'
import os, sys
scp, missing, n = sys.argv[1], 0, 0
with open(scp, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        _, path = line.split(maxsplit=1)
        n += 1
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            missing += 1
if missing:
    print(f"FAIL: {missing} missing files", file=sys.stderr)
    sys.exit(1)
print(f"OK: {n} entries")
PYEOF

# ---------- 数据集专属准备 ----------
case "$DATASET" in
  Seed-Eval)
    print_step "Step 4: clone seed-tts-eval"
    if [[ -d seed-tts-eval ]]; then
      echo "  seed-tts-eval/ already present, skipping."
    else
      git clone https://github.com/BytedanceSpeech/seed-tts-eval.git
    fi
    pip install --quiet -r seed-tts-eval/requirements.txt

    print_step "Step 5: download wavlm_large_finetune.pth"
    if [[ -s checkpoints/wavlm_large_finetune.pth ]]; then
      echo "  checkpoints/wavlm_large_finetune.pth already present, skipping."
    else
      mkdir -p checkpoints
      "$PY" - <<'PYEOF'
import os
os.environ.setdefault('HF_HOME', os.path.join(os.environ['XDG_CACHE_HOME'], 'huggingface'))
try:
    import hf_transfer
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
except ImportError:
    pass
from huggingface_hub import hf_hub_download
hf_hub_download("ssf1103/I2D-Bench", "wavlm_large_finetune.pth",
                repo_type="dataset", local_dir="checkpoints")
PYEOF
    fi
    ;;

  CV3-Eval)
    print_step "Step 4: download emotion2vec_plus_large"
    mkdir -p checkpoints/emotion2vec_plus_large
    "$PY" - <<'PYEOF'
import os, shutil
os.environ.setdefault('HF_HOME', os.path.join(os.environ['XDG_CACHE_HOME'], 'huggingface'))
from modelscope import snapshot_download
local_dir = snapshot_download(
    "iic/emotion2vec_plus_large",
    cache_dir=os.environ["HF_HOME"],
    allow_file_pattern=["model.pt", "config.yaml", "configuration.json", "tokens.txt"],
)
dst = "checkpoints/emotion2vec_plus_large"
for fname in ("model.pt", "config.yaml", "configuration.json", "tokens.txt"):
    src = os.path.join(local_dir, fname)
    if os.path.isfile(src):
        shutil.copyfile(src, os.path.join(dst, fname))
        print(f"copied {fname}")
PYEOF
    ;;
esac

print_step "Done. $DEST_DIR/ ready."
