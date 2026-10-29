#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/nvg-project}"
DATA_DIR="${DATA_DIR:-/workspace/nvg-data}"
ENV_DIR="${ENV_DIR:-/workspace/envs/nvg-sid}"
OFFICIAL_DIR="${OFFICIAL_DIR:-/workspace/official-sid}"
RUN_DIR="${RUN_DIR:-/workspace/nvg-run}"
SHORT_RAW="${SHORT_RAW:-${DATA_DIR}/Sony/short/10003_00_0.04s.ARW}"
LONG_RAW="${LONG_RAW:-${DATA_DIR}/Sony/long/10003_00_10s.ARW}"

if [[ ! -s "${SHORT_RAW}" || ! -s "${LONG_RAW}" ]]; then
  echo "Held-out official Sony test pair is missing: ${SHORT_RAW} / ${LONG_RAW}" >&2
  exit 3
fi

mkdir -p "${RUN_DIR}/sid-official"
cd "${PROJECT_DIR}"
ratio="$("${ENV_DIR}/bin/python" -c 'import sys; from pathlib import Path; from training.dataset import exposure; print(min(exposure(Path(sys.argv[2])) / exposure(Path(sys.argv[1])), 300.0))' "${SHORT_RAW}" "${LONG_RAW}")"
"${ENV_DIR}/bin/python" -m tools.make_sid_sample \
  --raw "${SHORT_RAW}" \
  --ratio "${ratio}" \
  --size 256 \
  --output "${RUN_DIR}/sid-official/input.npz"
TF_ENABLE_ONEDNN_OPTS=0 "${ENV_DIR}/bin/python" -m tools.export_reference_outputs \
  --checkpoint "${OFFICIAL_DIR}/checkpoint/Sony/model.ckpt" \
  --input "${RUN_DIR}/sid-official/input.npz" \
  --output "${RUN_DIR}/sid-official/reference.npz"
"${ENV_DIR}/bin/python" -m tools.verify_sid \
  --reference "${RUN_DIR}/sid-official/reference.npz" \
  --pytorch models/sid-sony.pt \
  --onnx models/sid-sony.onnx \
  --output "${RUN_DIR}/sid-official"
"${ENV_DIR}/bin/python" -m tools.export_browser_fixture \
  --reference "${RUN_DIR}/sid-official/reference.npz" \
  --output public/validation/sid
"${ENV_DIR}/bin/python" -m tools.verify_fp16 \
  --fp32 models/sid-sony.onnx \
  --fp16 models/sid-sony-fp16.onnx \
  --channels 4 \
  --output-name output \
  --sample "${RUN_DIR}/sid-official/reference.npz" \
  --output "${RUN_DIR}/sid-fp16.json"
echo "SID_OFFICIAL_VALIDATION_COMPLETE ratio=${ratio} short=${SHORT_RAW}"
