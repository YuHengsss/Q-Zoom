#!/bin/bash
# Stage 1: generate pseudo-labelled ROI attention maps from the base VLM.
# Uses the standardized_pipeline/stage1 entry point so the same script
# works for both Qwen2.5-VL and Qwen3-VL.

set -euo pipefail

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"
DATA_ROOT="${DATA_ROOT:-${HOME_PATH}/datasets}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

UNIVERSAL_INPUT="${UNIVERSAL_INPUT:-${DATA_ROOT}/stage1_universal_input.jsonl}"
ANSWERS_FILE="${ANSWERS_FILE:-${DATA_ROOT}/stage1results/qzoom_stage1_pseudo.pkl}"

mkdir -p "$(dirname "${ANSWERS_FILE}")"
cd "${CODE_ROOT}"

# Step 1.a — build the universal Stage-1 input from a manifest
if [ ! -f "${UNIVERSAL_INPUT}" ]; then
  python standardized_pipeline/stage1/build_universal_input.py \
    --manifest-file standardized_pipeline/stage1/example_manifest.json \
    --output-file "${UNIVERSAL_INPUT}"
fi

# Step 1.b — generate pseudo labels (offline-friendly; pass --offline + search roots when needed)
python standardized_pipeline/stage1/make_stage1_pseudo_labels.py \
  --model-path "${MODEL_PATH}" \
  --processor-path "${MODEL_PATH}" \
  --universal-input-file "${UNIVERSAL_INPUT}" \
  --image-folder "${DATA_ROOT}" \
  --answers-file "${ANSWERS_FILE}" \
  --gpu-ids "${GPU_IDS}" \
  --workers-per-gpu "${WORKERS_PER_GPU}"

echo "Stage-1 pseudo labels written to: ${ANSWERS_FILE}"
