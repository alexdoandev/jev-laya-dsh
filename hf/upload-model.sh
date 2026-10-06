#!/bin/bash
# Upload the exported multilingual ONNX bundle to your Hugging Face account.
#
# Prerequisites:
#   1. An HF account + a WRITE token: https://huggingface.co/settings/tokens
#   2. Export done: onnx-export/fp32/ contains laya.onnx, laya.onnx.data,
#      laya_config.json, tokenizer/
#
# Usage:
#   HF_TOKEN=hf_xxxxx bash hf/upload-model.sh [your-hf-username]
#
# Default repo id: <username>/laya-multilingual-onnx
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN=<write token from https://huggingface.co/settings/tokens>}"
USERNAME="${1:-}"
BUNDLE="$(cd "$(dirname "$0")/../onnx-export/fp32" && pwd)"
PY=~/.venvs/laya/bin/python
[ -x "$PY" ] || PY=python3

if [ -z "$USERNAME" ]; then
  USERNAME=$("$PY" - << 'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
print(api.whoami()["name"])
PY
)
fi
REPO="${USERNAME}/laya-multilingual-onnx"
echo "Uploading $BUNDLE → https://huggingface.co/$REPO"

cp "$(dirname "$0")/MODELCARD.md" "$BUNDLE/README.md"

"$PY" - "$REPO" "$BUNDLE" << 'PY'
import os
import sys
from huggingface_hub import HfApi

repo, bundle = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
api.upload_file(
    path_or_fileobj=os.path.join(os.path.dirname(bundle), "MODELCARD.md"),
    path_in_repo="README.md",
    repo_id=repo,
    repo_type="model",
)
api.upload_folder(folder_path=bundle, repo_id=repo, repo_type="model")
print("done → https://huggingface.co/" + repo)
PY
