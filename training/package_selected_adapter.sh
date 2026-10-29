#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/nvg-project}"
ENV_DIR="${ENV_DIR:-/workspace/envs/nvg-sid}"
RUN_DIR="${RUN_DIR:-/workspace/nvg-run}"
SELECTED_CHECKPOINT="${SELECTED_CHECKPOINT:?Set SELECTED_CHECKPOINT}"
SELECTED_SID_CHECKPOINT="${SELECTED_SID_CHECKPOINT:-}"
EVALUATION="${EVALUATION:?Set EVALUATION}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${RUN_DIR}/artifacts-final}"
HF_REPO="${HF_REPO:-verysweatydog/nvg}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"

if [[ -z "${HF_TOKEN:-}" && -r /proc/1/environ ]]; then
  HF_TOKEN="$(python3 -c 'items=dict(x.split(b"=",1) for x in open("/proc/1/environ","rb").read().split(b"\0") if b"=" in x); print(items.get(b"HF_TOKEN",b"").decode())')"
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == *RUNPOD_SECRET* ]]; then
  echo "HF_TOKEN is missing or the Runpod secret did not resolve" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/final" "${ARTIFACT_DIR}"
cd "${PROJECT_DIR}"

ACTIVE_SID_PYTORCH="models/sid-sony.pt"
ACTIVE_SID_ONNX="models/sid-sony.onnx"
ACTIVE_SID_URL="../../models/sid-sony.onnx"
PACKAGE_EXTRA=()
FINALIZER_EXTRA=()

if [[ -n "${SELECTED_SID_CHECKPOINT}" ]]; then
  "${ENV_DIR}/bin/python" -m tools.export_sid \
    --checkpoint "${SELECTED_SID_CHECKPOINT}" \
    --output models/sid-sony-joint.onnx
  "${ENV_DIR}/bin/python" -m tools.convert_fp16 \
    --input models/sid-sony-joint.onnx \
    --output models/sid-sony-joint-fp16.onnx
  "${ENV_DIR}/bin/python" -m tools.verify_fp16 \
    --fp32 models/sid-sony-joint.onnx \
    --fp16 models/sid-sony-joint-fp16.onnx \
    --channels 4 \
    --output-name output \
    --scale 0.8 \
    --output "${RUN_DIR}/final/sid-joint-fp16.json"
  ACTIVE_SID_PYTORCH="${SELECTED_SID_CHECKPOINT}"
  ACTIVE_SID_ONNX="models/sid-sony-joint.onnx"
  ACTIVE_SID_URL="../../models/sid-sony-joint.onnx"
  PACKAGE_EXTRA=(--sid-joint-fp16-validation "${RUN_DIR}/final/sid-joint-fp16.json")
  FINALIZER_EXTRA=(
    --learned-label "Joint-tuned adapter + SID"
    --manual-label "Deterministic pseudo-Bayer + joint-tuned SID"
    --raw-label "True packed Bayer RAW + joint-tuned SID"
  )
fi

"${ENV_DIR}/bin/python" -m tools.export_adapter \
  --checkpoint "${SELECTED_CHECKPOINT}" \
  --output models/pseudo-raw-adapter.onnx
"${ENV_DIR}/bin/python" -m tools.export_static_models \
  --sid-checkpoint models/sid-sony.pt \
  --sid-output models/sid-sony-360p.onnx \
  --adapter-checkpoint "${SELECTED_CHECKPOINT}" \
  --adapter-output models/pseudo-raw-adapter-360p.onnx
"${ENV_DIR}/bin/python" -m tools.convert_fp16 \
  --input models/sid-sony-360p.onnx \
  --output models/sid-sony-360p-fp16.onnx
"${ENV_DIR}/bin/python" -m tools.convert_fp16 \
  --input models/pseudo-raw-adapter-360p.onnx \
  --output models/pseudo-raw-adapter-360p-fp16.onnx
"${ENV_DIR}/bin/python" -m tools.verify_fp16 \
  --fp32 models/sid-sony-360p.onnx \
  --fp16 models/sid-sony-360p-fp16.onnx \
  --channels 4 --height 192 --width 320 \
  --output-name output --scale 0.8 \
  --output "${RUN_DIR}/final/sid-static-fp16.json"
"${ENV_DIR}/bin/python" -m tools.verify_fp16 \
  --fp32 models/pseudo-raw-adapter-360p.onnx \
  --fp16 models/pseudo-raw-adapter-360p-fp16.onnx \
  --channels 3 --height 384 --width 640 \
  --output-name pseudo_raw --scale 0.025 \
  --output "${RUN_DIR}/final/adapter-static-fp16.json"
"${ENV_DIR}/bin/python" -m tools.verify_static_models \
  --adapter-dynamic models/pseudo-raw-adapter.onnx \
  --adapter-static models/pseudo-raw-adapter-360p.onnx \
  --sid-dynamic models/sid-sony.onnx \
  --sid-static models/sid-sony-360p.onnx \
  --output "${RUN_DIR}/final/static-equivalence-original.json"
PACKAGE_EXTRA+=(
  --sid-static-fp16-validation "${RUN_DIR}/final/sid-static-fp16.json"
  --adapter-static-fp16-validation "${RUN_DIR}/final/adapter-static-fp16.json"
)
if [[ -n "${SELECTED_SID_CHECKPOINT}" ]]; then
  "${ENV_DIR}/bin/python" -m tools.export_static_models \
    --sid-checkpoint "${SELECTED_SID_CHECKPOINT}" \
    --sid-output models/sid-sony-joint-360p.onnx
  "${ENV_DIR}/bin/python" -m tools.convert_fp16 \
    --input models/sid-sony-joint-360p.onnx \
    --output models/sid-sony-joint-360p-fp16.onnx
  "${ENV_DIR}/bin/python" -m tools.verify_fp16 \
    --fp32 models/sid-sony-joint-360p.onnx \
    --fp16 models/sid-sony-joint-360p-fp16.onnx \
    --channels 4 --height 192 --width 320 \
    --output-name output --scale 0.8 \
    --output "${RUN_DIR}/final/sid-joint-static-fp16.json"
  "${ENV_DIR}/bin/python" -m tools.verify_static_models \
    --adapter-dynamic models/pseudo-raw-adapter.onnx \
    --adapter-static models/pseudo-raw-adapter-360p.onnx \
    --sid-dynamic models/sid-sony-joint.onnx \
    --sid-static models/sid-sony-joint-360p.onnx \
    --output "${RUN_DIR}/final/static-equivalence-joint.json"
  PACKAGE_EXTRA+=(--sid-joint-static-fp16-validation "${RUN_DIR}/final/sid-joint-static-fp16.json")
fi
"${ENV_DIR}/bin/python" -m tools.verify_adapter \
  --adapter-checkpoint "${SELECTED_CHECKPOINT}" \
  --adapter-onnx models/pseudo-raw-adapter.onnx \
  --sid-pytorch "${ACTIVE_SID_PYTORCH}" \
  --sid-onnx "${ACTIVE_SID_ONNX}" \
  --output "${RUN_DIR}/final/equivalence.json" \
  --fixture "${RUN_DIR}/final/browser-reference.npz"
"${ENV_DIR}/bin/python" -m tools.export_adapter_browser_fixtures \
  --reference "${RUN_DIR}/final/browser-reference.npz" \
  --output public/validation \
  --sid-url "${ACTIVE_SID_URL}"
"${ENV_DIR}/bin/python" -m tools.convert_fp16 \
  --input models/pseudo-raw-adapter.onnx \
  --output models/pseudo-raw-adapter-fp16.onnx
"${ENV_DIR}/bin/python" -m tools.verify_fp16 \
  --fp32 models/pseudo-raw-adapter.onnx \
  --fp16 models/pseudo-raw-adapter-fp16.onnx \
  --channels 3 \
  --output-name pseudo_raw \
  --scale 0.025 \
  --output "${RUN_DIR}/final/fp16.json"
"${ENV_DIR}/bin/python" -m tools.finalize_model_card \
  --card models/MODEL_CARD.md \
  --evaluation "${EVALUATION}" \
  "${FINALIZER_EXTRA[@]}"
"${ENV_DIR}/bin/python" -m tools.package_models \
  --models models \
  --adapter-checkpoint "${SELECTED_CHECKPOINT}" \
  --sid-validation "${RUN_DIR}/sid-official/metrics.json" \
  --sid-fp16-validation "${RUN_DIR}/sid-fp16.json" \
  --adapter-fp16-validation "${RUN_DIR}/final/fp16.json" \
  "${PACKAGE_EXTRA[@]}"

cp models/sid-sony.onnx models/sid-sony-fp16.onnx models/sid-sony.pt \
  models/sid-sony-360p.onnx models/sid-sony-360p-fp16.onnx \
  models/pseudo-raw-adapter.onnx models/pseudo-raw-adapter-fp16.onnx \
  models/pseudo-raw-adapter-360p.onnx models/pseudo-raw-adapter-360p-fp16.onnx \
  models/metadata.json "${ARTIFACT_DIR}/"
cp "${SELECTED_CHECKPOINT}" "${ARTIFACT_DIR}/adapter-best.pt"
cp "$(dirname "${SELECTED_CHECKPOINT}")/metrics.jsonl" "${ARTIFACT_DIR}/training-metrics.jsonl"
cp "${EVALUATION}" "${ARTIFACT_DIR}/evaluation.json"
cp "${RUN_DIR}/final/equivalence.json" "${ARTIFACT_DIR}/adapter-equivalence.json"
cp "${RUN_DIR}/final/fp16.json" "${ARTIFACT_DIR}/adapter-fp16-validation.json"
cp "${RUN_DIR}/sid-official/metrics.json" "${ARTIFACT_DIR}/sid-validation.json"
cp "${RUN_DIR}/sid-official/comparison.png" "${ARTIFACT_DIR}/sid-validation-comparison.png"
cp "${RUN_DIR}/sid-fp16.json" "${ARTIFACT_DIR}/sid-fp16-validation.json"
cp "${RUN_DIR}/final/sid-static-fp16.json" "${ARTIFACT_DIR}/sid-static-fp16-validation.json"
cp "${RUN_DIR}/final/adapter-static-fp16.json" "${ARTIFACT_DIR}/adapter-static-fp16-validation.json"
cp "${RUN_DIR}/final/static-equivalence-original.json" "${ARTIFACT_DIR}/static-equivalence-original.json"
if [[ -n "${SELECTED_SID_CHECKPOINT}" ]]; then
  cp models/sid-sony-joint.onnx models/sid-sony-joint-fp16.onnx "${ARTIFACT_DIR}/"
  cp models/sid-sony-joint-360p.onnx models/sid-sony-joint-360p-fp16.onnx "${ARTIFACT_DIR}/"
  cp "${SELECTED_SID_CHECKPOINT}" "${ARTIFACT_DIR}/sid-joint-best.pt"
  cp "${RUN_DIR}/final/sid-joint-fp16.json" "${ARTIFACT_DIR}/sid-joint-fp16-validation.json"
  cp "${RUN_DIR}/final/sid-joint-static-fp16.json" "${ARTIFACT_DIR}/sid-joint-static-fp16-validation.json"
  cp "${RUN_DIR}/final/static-equivalence-joint.json" "${ARTIFACT_DIR}/static-equivalence-joint.json"
fi
cp models/MODEL_CARD.md "${ARTIFACT_DIR}/README.md"
cp LICENSE "${ARTIFACT_DIR}/LICENSE"
cp -r public/validation "${ARTIFACT_DIR}/browser-validation"
if [[ -s "${RUN_DIR}/ablations/summary.json" ]]; then
  cp "${RUN_DIR}/ablations/summary.json" "${ARTIFACT_DIR}/ablation-summary.json"
fi
if [[ -s "${RUN_DIR}/raw-ablations/summary.json" ]]; then
  cp "${RUN_DIR}/raw-ablations/summary.json" "${ARTIFACT_DIR}/raw-supervision-ablation-summary.json"
fi

"${ENV_DIR}/bin/python" -m tools.upload_artifacts \
  --folder "${ARTIFACT_DIR}" \
  --repo "${HF_REPO}" \
  --repo-type "${HF_REPO_TYPE}"
echo "SELECTED_ADAPTER_PACKAGE_COMPLETE"
