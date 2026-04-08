#!/bin/bash
# Build the Stage-3 training data via standardized_pipeline/stage3.

set -euo pipefail

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"
DATA_ROOT="${DATA_ROOT:-${HOME_PATH}/datasets}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

UNIVERSAL_INPUT="${STAGE3_UNIVERSAL_INPUT:-${DATA_ROOT}/stage3_universal_input.jsonl}"
STAGE3_OUTPUT="${STAGE3_OUTPUT:-${DATA_ROOT}/stage3results/qzoom_stage3.pkl}"

cd "${CODE_ROOT}"
mkdir -p "$(dirname "${STAGE3_OUTPUT}")"

if [ ! -f "${UNIVERSAL_INPUT}" ]; then
  python standardized_pipeline/stage3/build_universal_input.py \
    --manifest-file standardized_pipeline/stage3/example_manifest.json \
    --output-file "${UNIVERSAL_INPUT}"
fi

python standardized_pipeline/stage3/make_stage3_training_data.py \
  --code-root "${CODE_ROOT}" \
  --universal-input-file "${UNIVERSAL_INPUT}" \
  --image-folder "${DATA_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --model-type auto \
  --gpu-ids "${GPU_IDS}" \
  --stage3-output "${STAGE3_OUTPUT}"

echo "Stage-3 training data written to: ${STAGE3_OUTPUT}"
