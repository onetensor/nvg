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
  local scale="$3"
  local learning_rate="$4"
  echo "ABLATION_START name=${name} width=${width} scale=${scale} lr=${learning_rate}"
  "${ENV_DIR}/bin/python" -m training.train_adapter \
    --data "${PATCH_DIR}" \
    --sid models/sid-sony.pt \
    --output "${RUN_DIR}/ablations/${name}" \
    --steps 3000 \
    --batch-size 2 \
    --patch-size 512 \
    --workers 8 \
    --validate-every 250 \
    --save-every 3000 \
    --width "${width}" \
    --residual-scale "${scale}" \
    --learning-rate "${learning_rate}"
  echo "ABLATION_COMPLETE name=${name}"
}

run_variant w32_s10_lr20 32 0.10 2e-5
run_variant w32_s05_lr20 32 0.05 2e-5
run_variant w48_s10_lr15 48 0.10 1.5e-5
run_variant w64_s25_lr10 64 0.25 1e-5
run_variant w64_s10_lr10 64 0.10 1e-5
run_variant w64_s05_lr10 64 0.05 1e-5
run_variant w64_s10_lr20 64 0.10 2e-5

"${ENV_DIR}/bin/python" - <<'PY'
import json
from pathlib import Path

root = Path("/workspace/nvg-run/ablations")
rows = []
for path in sorted(root.glob("*/metrics.jsonl")):
    reports = [json.loads(line) for line in path.read_text().splitlines() if "learned_psnr" in line]
    best = max(reports, key=lambda row: row["learned_psnr"] + 10 * row["learned_ssim"])
    rows.append({"name": path.parent.name, **best, "score": best["learned_psnr"] + 10 * best["learned_ssim"]})
rows.sort(key=lambda row: row["score"], reverse=True)
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("ABLATION_SUMMARY " + json.dumps(rows))
PY
echo "ABLATIONS_COMPLETE"
