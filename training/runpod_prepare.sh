#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/nvg-project}"
DATA_DIR="${DATA_DIR:-/workspace/nvg-data}"
ENV_DIR="${ENV_DIR:-/workspace/envs/nvg-sid}"
OFFICIAL_DIR="${OFFICIAL_DIR:-/workspace/official-sid}"

if [[ -z "${HF_TOKEN:-}" && -r /proc/1/environ ]]; then
  HF_TOKEN="$(python3 -c 'items=dict(x.split(b"=",1) for x in open("/proc/1/environ","rb").read().split(b"\0") if b"=" in x); print(items.get(b"HF_TOKEN",b"").decode())')"
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == *RUNPOD_SECRET* ]]; then
  echo "HF_TOKEN is missing or the Runpod secret did not resolve" >&2
  exit 2
fi

mkdir -p /workspace/cache /workspace/envs "${DATA_DIR}"
python3 -m venv --system-site-packages "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements-train.txt" gdown

"${ENV_DIR}/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="verysweatydog/nvg",
    repo_type="dataset",
    local_dir=os.environ.get("DATA_DIR", "/workspace/nvg-data"),
    token=os.environ["HF_TOKEN"],
)
print("DATASET_DOWNLOAD_COMPLETE")
PY

if [[ ! -d "${OFFICIAL_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/cchen156/Learning-to-See-in-the-Dark.git "${OFFICIAL_DIR}"
fi
mkdir -p "${OFFICIAL_DIR}/checkpoint/Sony"
if [[ ! -s "${OFFICIAL_DIR}/checkpoint/Sony/model.ckpt.data-00000-of-00001" ]]; then
  "${ENV_DIR}/bin/gdown" 1wmx7AM6XWHjHIvpErmIouQgbQoMxAymG -O "${OFFICIAL_DIR}/checkpoint/Sony/model.ckpt.data-00000-of-00001"
fi
if [[ ! -s "${OFFICIAL_DIR}/checkpoint/Sony/model.ckpt.meta" ]]; then
  "${ENV_DIR}/bin/gdown" 1OmrGMng1QuwUa8lf-_wBVvbRJwBr0ETr -O "${OFFICIAL_DIR}/checkpoint/Sony/model.ckpt.meta"
fi

cd "${PROJECT_DIR}"
"${ENV_DIR}/bin/python" -m tools.convert_sid \
  --checkpoint "${OFFICIAL_DIR}/checkpoint/Sony/model.ckpt" \
  --output models/sid-sony.pt
"${ENV_DIR}/bin/python" -m tools.convert_fp16 \
  --input models/sid-sony.onnx \
  --output models/sid-sony-fp16.onnx
"${ENV_DIR}/bin/python" -m tools.verify_fp16 \
  --fp32 models/sid-sony.onnx \
  --fp16 models/sid-sony-fp16.onnx \
  --channels 4 \
  --output-name output \
  --output /workspace/nvg-run/sid-fp16.json
echo "PREPARE_COMPLETE"
