#!/bin/bash
# Evaluate a Stage-1 Q-Zoom checkpoint on Doc/OCR + HR/Vision benchmarks.

set -euo pipefail

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"

export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME_PATH}/.cache/huggingface}"
export HF_HOME="${HF_HOME:-${HOME_PATH}/.cache/huggingface}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"

cd "${CODE_ROOT}"
mkdir -p ./logs

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CODE_ROOT}/output/qzoom-qwen2_5vl-7b-K18T3-stage1}"
LOG_SUFFIX_BASE="${LOG_SUFFIX_BASE:-qzoom_stage1}"
NUM_GPUS="${NUM_GPUS:-4}"

# Qwen2.5-VL uses patch_size=28
PATCH=28
MIN_DOC=$((256 * PATCH * PATCH))   # 200704
MAX_DOC=$((576 * PATCH * PATCH))   # 451584
MIN_HR=$((512 * PATCH * PATCH))    # 401408
MAX_HR=$((4096 * PATCH * PATCH))   # 3211264

run_eval () {
  local task="$1"
  local conf="$2"
  local hr="$3"
  local minp="$4"
  local maxp="$5"
  local model_args="pretrained=${CHECKPOINT_PATH},device_map=auto,two_stage_roi=True,roi_conf_thresh=${conf},high_res_thresh=${hr},min_pixels=${minp},max_pixels=${maxp},attn_implementation=flash_attention_2"
  accelerate launch --num_processes="${NUM_GPUS}" -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "${model_args}" \
    --tasks "${task}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${LOG_SUFFIX_BASE}_${task}" \
    --output_path ./logs/
}

# Doc/OCR
run_eval "textvqa_val"  "0.15"  "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "infovqa_val"  "0.15"  "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "chartqa"      "0.125" "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "ocrbench"     "0.15"  "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "docvqa_val"   "0.125" "0.10" "${MIN_DOC}" "${MAX_DOC}"

# HR / Vision-centric
run_eval "vstar_bench"        "0.05" "0.05" "${MIN_HR}" "${MAX_HR}"
run_eval "mmerealworld_lite"  "0.05" "0.05" "${MIN_HR}" "${MAX_HR}"
run_eval "hrbench"            "0.05" "0.05" "${MIN_HR}" "${MAX_HR}"
