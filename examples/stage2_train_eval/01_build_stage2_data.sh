#!/bin/bash
# Build the Stage-2 post-SFT dataset.
#
# IMPORTANT: when training Q-Zoom on more than one backbone (e.g. both
# Qwen2.5-VL-7B and Qwen3-VL-4B), each backbone produces its OWN base /
# ROI / post-SFT files. The defaults below tag every output filename with
# ${BACKBONE_TAG} so different backbones do not overwrite each other.
# Override BACKBONE_TAG (e.g. qwen2_5vl_7b, qwen2_5vl_3b, qwen3vl_4b)
# to match the model you are running.

set -euo pipefail

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"
DATA_ROOT="${DATA_ROOT:-${HOME_PATH}/datasets}"

# Tag that distinguishes data files for different backbones. Override per run.
# Examples: qwen2_5vl_3b | qwen2_5vl_7b | qwen3vl_4b
BACKBONE_TAG="${BACKBONE_TAG:-qwen2_5vl_7b}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
STAGE1_CKPT="${STAGE1_CKPT:-${CODE_ROOT}/output/qzoom-qwen2_5vl-7b-K18T3-stage1}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${BASE_MODEL_PATH}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

UNIVERSAL_INPUT="${STAGE2_UNIVERSAL_INPUT:-${DATA_ROOT}/stage2_universal_input.jsonl}"
BASE_OUTPUT="${BASE_OUTPUT:-${DATA_ROOT}/stage2_base_${BACKBONE_TAG}.pkl}"
ROI_OUTPUT="${ROI_OUTPUT:-${DATA_ROOT}/stage2_roi_${BACKBONE_TAG}.pkl}"
POST_SFT_JSONL="${POST_SFT_JSONL:-${DATA_ROOT}/qzoom_post_sft_stage2_${BACKBONE_TAG}.jsonl}"

cd "${CODE_ROOT}"
mkdir -p "$(dirname "${POST_SFT_JSONL}")"

if [ ! -f "${UNIVERSAL_INPUT}" ]; then
  python standardized_pipeline/stage2/build_universal_input.py \
    --manifest-file standardized_pipeline/stage2/example_manifest.json \
    --output-file "${UNIVERSAL_INPUT}"
fi

python standardized_pipeline/stage2/make_stage2_training_data.py \
  --code-root "${CODE_ROOT}" \
  --universal-input-file "${UNIVERSAL_INPUT}" \
  --image-folder "${DATA_ROOT}" \
  --base-model-path "${BASE_MODEL_PATH}" \
  --roi-model-path "${STAGE1_CKPT}" \
  --model-type auto \
  --gpu-ids "${GPU_IDS}" \
  --base-output "${BASE_OUTPUT}" \
  --roi-output "${ROI_OUTPUT}" \
  --judge-model-path "${JUDGE_MODEL_PATH}" \
  --judge-model-type auto \
  --judge-gpu-ids "${GPU_IDS}" \
  --post-sft-output "${POST_SFT_JSONL}" \
  --data-root "${DATA_ROOT}" \
  --mixing-ratio 0.0

echo "Stage-2 post-SFT data written to: ${POST_SFT_JSONL}"
