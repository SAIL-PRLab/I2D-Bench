#/bin/bash

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ftshijt/UTMOSv2.git
( cd UTMOSv2 && pip install -e . )

REPO_ID="sarulab-speech/UTMOSv2"
FILENAME="fold0_s42_best_model.pth"
DEST_DIR="UTMOSv2/models/fusion_stage3"
mkdir -p "$DEST_DIR"

python - "$REPO_ID" "$FILENAME" "$DEST_DIR" <<'PYEOF'
import os, sys
try:
    import hf_transfer  # noqa: F401
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    print("hf_transfer enabled (fast multi-conn)")
except ImportError:
    print("hf_transfer not installed, falling back to default HTTP")
from huggingface_hub import hf_hub_download
repo_id, filename, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
path = hf_hub_download(repo_id, filename, local_dir=local_dir)
print("downloaded ->", path)
PYEOF

echo "Be aware to execute 'source activate_utmosv2.sh' to enable the checkpoint"
