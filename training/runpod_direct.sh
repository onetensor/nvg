#!/usr/bin/env bash
# Adapter-free direct SID training under the simulated-iPhone protocol.
# Usage: STAGE=<setup|data|patches|teacher|student|evaluate|export|upload> bash training/runpod_direct.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/nvg-project}"
DATA_DIR="${DATA_DIR:-/workspace/nvg-data}"
PATCH_DIR="${PATCH_DIR:-/workspace/nvg-patches-direct}"
ENV_DIR="${ENV_DIR:-/workspace/envs/nvg-direct}"
RUN_DIR="${RUN_DIR:-/workspace/nvg-direct-run}"
WORKERS="${WORKERS:-$(( $(nproc) > 32 ? 32 : $(nproc) ))}"
TEACHER_STEPS="${TEACHER_STEPS:-30000}"
STUDENT_STEPS="${STUDENT_STEPS:-60000}"
STUDENT_WIDTH="${STUDENT_WIDTH:-16}"
PY="${ENV_DIR}/bin/python"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"

case "${STAGE:?set STAGE}" in
  setup)
    python3 -m venv --system-site-packages "${ENV_DIR}"
    "${PY}" -m pip install -q --upgrade pip
    "${PY}" -m pip install -q rawpy pillow onnx onnxruntime onnxscript onnxconverter-common huggingface_hub
    "${PY}" -c "import torch; print('TORCH', torch.__version__, torch.cuda.get_device_name(0))"
    ;;
  data)
    DATA_DIR="${DATA_DIR}" bash training/runpod_dataset.sh
    ;;
  patches)
    # Same deterministic val/test selection as the published adapter evaluation.
    "${PY}" -m training.build_patch_dataset \
      --data "${DATA_DIR}" --output "${PATCH_DIR}" --patch-size 512 \
      --train-patches 4 --max-train-records 0 --eval-patches 1 --max-eval-records 100 \
      --workers "${WORKERS}"
    ;;
  teacher)
    "${PY}" -m training.train_direct --protocol phone --data "${PATCH_DIR}" \
      --output "${RUN_DIR}/w32" --init-sid models/sid-joint-best.pt --width 32 \
      --steps "${TEACHER_STEPS}" --batch-size 16 --output-size 256 \
      --learning-rate 5e-5 --warmup-steps 500 --workers "${WORKERS}" \
      --validate-every 1000 --save-every 5000
    ;;
  student)
    "${PY}" -m training.train_direct --protocol phone --data "${PATCH_DIR}" \
      --output "${RUN_DIR}/w${STUDENT_WIDTH}" --width "${STUDENT_WIDTH}" \
      --teacher "${RUN_DIR}/w32/direct-best.pt" --teacher-weight 0.5 \
      --steps "${STUDENT_STEPS}" --batch-size 16 --output-size 256 \
      --learning-rate 5e-4 --warmup-steps 1000 --workers "${WORKERS}" \
      --validate-every 2000 --save-every 10000
    ;;
  evaluate)
    models=("w32=${RUN_DIR}/w32/direct-best.pt")
    [[ -s "${RUN_DIR}/w${STUDENT_WIDTH}/direct-best.pt" ]] && models+=("w${STUDENT_WIDTH}=${RUN_DIR}/w${STUDENT_WIDTH}/direct-best.pt")
    common=(--data "${PATCH_DIR}" --baseline-sid models/sid-joint-best.pt --baseline-adapter models/adapter-best.pt --direct "${models[@]}")
    "${PY}" -m training.evaluate_direct "${common[@]}" --protocol phone --downsample 2 --output-size 256 --output "${RUN_DIR}/eval-phone-ds2.json"
    "${PY}" -m training.evaluate_direct "${common[@]}" --protocol phone --downsample 4 --output-size 128 --output "${RUN_DIR}/eval-phone-ds4.json"
    "${PY}" -m training.evaluate_direct "${common[@]}" --protocol legacy --output "${RUN_DIR}/eval-legacy.json"
    ;;
  export)
    for width in 32 "${STUDENT_WIDTH}"; do
      checkpoint="${RUN_DIR}/w${width}/direct-best.pt"
      [[ -s "${checkpoint}" ]] || continue
      "${PY}" -m tools.export_direct --checkpoint "${checkpoint}" --output "${RUN_DIR}/artifacts/sid-direct-w${width}.onnx"
      "${PY}" -m tools.convert_fp16 --input "${RUN_DIR}/artifacts/sid-direct-w${width}.onnx" --output "${RUN_DIR}/artifacts/sid-direct-w${width}-fp16.onnx"
      "${PY}" -m tools.verify_fp16 --fp32 "${RUN_DIR}/artifacts/sid-direct-w${width}.onnx" \
        --fp16 "${RUN_DIR}/artifacts/sid-direct-w${width}-fp16.onnx" --channels 12 --output-name output \
        --output "${RUN_DIR}/artifacts/sid-direct-w${width}-fp16.json"
      cp "${checkpoint}" "${RUN_DIR}/artifacts/sid-direct-w${width}.pt"
    done
    cp "${RUN_DIR}"/eval-*.json "${RUN_DIR}/artifacts/" 2>/dev/null || true
    ;;
  upload)
    if [[ -z "${HF_TOKEN:-}" && -r /proc/1/environ ]]; then
      HF_TOKEN="$(python3 -c 'items=dict(x.split(b"=",1) for x in open("/proc/1/environ","rb").read().split(b"\0") if b"=" in x); print(items.get(b"HF_TOKEN",b"").decode())')"
      export HF_TOKEN
    fi
    "${PY}" - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
info = api.repo_info("verysweatydog/nvg", repo_type="model")
if not info.private:
    raise SystemExit("Refusing to upload to a public repository")
api.upload_folder(repo_id="verysweatydog/nvg", repo_type="model",
                  folder_path=os.environ.get("RUN_DIR", "/workspace/nvg-direct-run") + "/artifacts",
                  path_in_repo="direct", commit_message="Direct SID models (phone protocol)")
print("UPLOAD_COMPLETE")
PY
    ;;
  *)
    echo "Unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac
echo "STAGE_COMPLETE ${STAGE}"
