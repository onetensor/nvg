#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/nvg-project}"
DATA_DIR="${DATA_DIR:-/workspace/nvg-data}"
PATCH_DIR="${PATCH_DIR:-/workspace/nvg-patches}"
ENV_DIR="${ENV_DIR:-/workspace/envs/nvg-sid}"
RUN_DIR="${RUN_DIR:-/workspace/nvg-run}"
STEPS="${STEPS:-3000}"

if [[ -z "${HF_TOKEN:-}" && -r /proc/1/environ ]]; then
  HF_TOKEN="$(python3 -c 'items=dict(x.split(b"=",1) for x in open("/proc/1/environ","rb").read().split(b"\0") if b"=" in x); print(items.get(b"HF_TOKEN",b"").decode())')"
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == *RUNPOD_SECRET* ]]; then
  echo "HF_TOKEN is missing or the Runpod secret did not resolve" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -s "${RUN_DIR}/sid-official/metrics.json" ]]; then
  bash training/runpod_validate.sh
fi

if [[ ! -s "${PATCH_DIR}/patches/manifest.json" ]]; then
  "${ENV_DIR}/bin/python" -m training.build_patch_dataset \
    --data "${DATA_DIR}" \
    --output "${PATCH_DIR}" \
    --patch-size 512 \
    --train-patches 2 \
    --eval-patches 1 \
    --max-train-records 800 \
    --max-eval-records 100
fi

"${ENV_DIR}/bin/python" -m training.train_adapter \
  --data "${PATCH_DIR}" \
  --sid models/sid-sony.pt \
  --output "${RUN_DIR}/adapter" \
  --steps "${STEPS}" \
  --batch-size 2 \
  --patch-size 512 \
  --learning-rate 1e-5 \
  --width 64 \
  --residual-scale 0.10 \
  --raw-weight 10 \
  --validate-every 250 \
  --workers 8

"${ENV_DIR}/bin/python" -m tools.export_adapter \
  --checkpoint "${RUN_DIR}/adapter/adapter-best.pt" \
  --output models/pseudo-raw-adapter.onnx
"${ENV_DIR}/bin/python" -m tools.verify_adapter \
  --adapter-checkpoint "${RUN_DIR}/adapter/adapter-best.pt" \
  --adapter-onnx models/pseudo-raw-adapter.onnx \
  --sid-pytorch models/sid-sony.pt \
  --sid-onnx models/sid-sony.onnx \
  --output "${RUN_DIR}/adapter/equivalence.json" \
  --fixture "${RUN_DIR}/adapter/browser-reference.npz"
"${ENV_DIR}/bin/python" -m tools.export_adapter_browser_fixtures \
  --reference "${RUN_DIR}/adapter/browser-reference.npz" \
  --output public/validation
"${ENV_DIR}/bin/python" -m tools.convert_fp16 \
  --input models/pseudo-raw-adapter.onnx \
  --output models/pseudo-raw-adapter-fp16.onnx
"${ENV_DIR}/bin/python" -m tools.verify_fp16 \
  --fp32 models/pseudo-raw-adapter.onnx \
  --fp16 models/pseudo-raw-adapter-fp16.onnx \
  --channels 3 \
  --output-name pseudo_raw \
  --scale 0.025 \
  --output "${RUN_DIR}/adapter/fp16.json"

"${ENV_DIR}/bin/python" -m training.evaluate_adapter \
  --data "${PATCH_DIR}" \
  --sid models/sid-sony.pt \
  --adapter "${RUN_DIR}/adapter/adapter-best.pt" \
  --output "${RUN_DIR}/adapter/evaluation.json" \
  --require-improvement
"${ENV_DIR}/bin/python" -m tools.finalize_model_card \
  --card models/MODEL_CARD.md \
  --evaluation "${RUN_DIR}/adapter/evaluation.json"

"${ENV_DIR}/bin/python" -m tools.package_models \
  --models models \
  --adapter-checkpoint "${RUN_DIR}/adapter/adapter-best.pt" \
  --sid-validation "${RUN_DIR}/sid-official/metrics.json" \
  --sid-fp16-validation "${RUN_DIR}/sid-fp16.json" \
  --adapter-fp16-validation "${RUN_DIR}/adapter/fp16.json"

mkdir -p "${RUN_DIR}/artifacts"
cp models/sid-sony.onnx models/sid-sony-fp16.onnx \
  models/pseudo-raw-adapter.onnx models/pseudo-raw-adapter-fp16.onnx \
  models/metadata.json "${RUN_DIR}/artifacts/"
cp models/sid-sony.pt "${RUN_DIR}/artifacts/"
cp "${RUN_DIR}/adapter/adapter-best.pt" "${RUN_DIR}/artifacts/"
cp "${RUN_DIR}/adapter/adapter-latest.pt" "${RUN_DIR}/artifacts/"
cp "${RUN_DIR}/adapter/metrics.jsonl" "${RUN_DIR}/artifacts/training-metrics.jsonl"
cp models/MODEL_CARD.md "${RUN_DIR}/artifacts/README.md"
cp LICENSE "${RUN_DIR}/artifacts/LICENSE"
cp "${RUN_DIR}/adapter/evaluation.json" "${RUN_DIR}/artifacts/"
cp "${RUN_DIR}/adapter/equivalence.json" "${RUN_DIR}/artifacts/adapter-equivalence.json"
cp "${RUN_DIR}/sid-official/metrics.json" "${RUN_DIR}/artifacts/sid-validation.json"
cp "${RUN_DIR}/sid-official/comparison.png" "${RUN_DIR}/artifacts/sid-validation-comparison.png"
cp "${RUN_DIR}/sid-fp16.json" "${RUN_DIR}/artifacts/sid-fp16-validation.json"
cp "${RUN_DIR}/adapter/fp16.json" "${RUN_DIR}/artifacts/adapter-fp16-validation.json"
cp -r public/validation "${RUN_DIR}/artifacts/browser-validation"
"${ENV_DIR}/bin/python" -m tools.upload_artifacts --folder "${RUN_DIR}/artifacts" --repo verysweatydog/nvg
echo "RUN_COMPLETE"
