#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/nvg-data}"
ARCHIVE="/workspace/Sony2025.zip"
OFFICIAL_URL="https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip"
EXPECTED_ARW=2928

current="$(find "${DATA_DIR}" -type f -iname '*.ARW' 2>/dev/null | wc -l)"
if [[ "${current}" -eq "${EXPECTED_ARW}" ]]; then
  echo "OFFICIAL_DATASET_ALREADY_PRESENT arw_files=${current}"
  exit 0
fi

echo "The private dataset repository has no samples; downloading the official Sony SID archive."
curl --fail --location --continue-at - --output "${ARCHIVE}" "${OFFICIAL_URL}"
mkdir -p "${DATA_DIR}"
unzip -oq "${ARCHIVE}" -d "${DATA_DIR}"

count="$(find "${DATA_DIR}" -type f -iname '*.ARW' | wc -l)"
if [[ "${count}" -ne "${EXPECTED_ARW}" ]]; then
  echo "Official archive integrity count failed: ${count}/${EXPECTED_ARW} ARW files" >&2
  exit 3
fi
rm -f "${ARCHIVE}"
echo "OFFICIAL_DATASET_COMPLETE arw_files=${count}"
