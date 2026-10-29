#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/nvg-project}"
ENV_DIR="${ENV_DIR:-/workspace/envs/nvg-sid}"
PATCH_DIR="${PATCH_DIR:-/workspace/nvg-patches}"
RUN_DIR="${RUN_DIR:-/workspace/nvg-run}"

cd "${PROJECT_DIR}"

run_variant() {
  local name="$1"
  local width="$2"
  local learning_rate="$3"
  local raw_weight="$4"
  echo "RAW_ABLATION_START name=${name} width=${width} lr=${learning_rate} raw_weight=${raw_weight}"
  "${ENV_DIR}/bin/python" -m training.train_adapter \
    --data "${PATCH_DIR}" \
    --sid models/sid-sony.pt \
    --output "${RUN_DIR}/raw-ablations/${name}" \
    --steps 2000 \
    --batch-size 2 \
    --patch-size 512 \
    --workers 8 \
    --validate-every 250 \
    --save-every 2000 \
    --width "${width}" \
    --residual-scale 0.10 \
    --learning-rate "${learning_rate}" \
    --raw-weight "${raw_weight}"
  echo "RAW_ABLATION_COMPLETE name=${name}"
}

run_variant w64_rw10 64 1e-5 10
run_variant w64_rw50 64 1e-5 50
run_variant w64_rw200 64 1e-5 200
run_variant w96_rw01 96 7e-6 0.1
run_variant w96_rw10 96 7e-6 10
run_variant w96_rw50 96 7e-6 50
run_variant w96_rw200 96 7e-6 200

"${ENV_DIR}/bin/python" - <<'PY'
import json
from pathlib import Path

root = Path("/workspace/nvg-run/raw-ablations")
rows = []
for path in sorted(root.glob("*/metrics.jsonl")):
    reports = [json.loads(line) for line in path.read_text().splitlines() if "learned_psnr" in line]
    best = max(reports, key=lambda row: row["learned_psnr"] + 10 * row["learned_ssim"])
    rows.append({"name": path.parent.name, **best, "score": best["learned_psnr"] + 10 * best["learned_ssim"]})
rows.sort(key=lambda row: row["score"], reverse=True)
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("RAW_ABLATION_SUMMARY " + json.dumps(rows))
PY
echo "RAW_ABLATIONS_COMPLETE"
