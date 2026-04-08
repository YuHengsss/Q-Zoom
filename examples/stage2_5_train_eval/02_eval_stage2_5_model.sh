#!/bin/bash
# Evaluate a Stage-2.5 Q-Zoom checkpoint (Qwen3-VL).

set -euo pipefail

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"

export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME_PATH}/.cache/huggingface}"
export HF_HOME="${HF_HOME:-${HOME_PATH}/.cache/huggingface}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"

cd "${CODE_ROOT}"
mkdir -p ./logs

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CODE_ROOT}/output/qzoom-qwen3vl-4b-K24T3-stage2.5}"
LOG_SUFFIX_BASE="${LOG_SUFFIX_BASE:-qzoom_stage2_5}"
NUM_GPUS="${NUM_GPUS:-4}"

# Qwen3-VL: patch_size=32
PATCH=32
MIN_DOC=$((256 * PATCH * PATCH))
MAX_DOC=$((576 * PATCH * PATCH))
MIN_HR=$((512 * PATCH * PATCH))
MAX_HR=$((4096 * PATCH * PATCH))

run_eval () {
  local task="$1" conf="$2" hr="$3" minp="$4" maxp="$5"
  local model_args="pretrained=${CHECKPOINT_PATH},device_map=auto,two_stage_roi=True,roi_conf_thresh=${conf},high_res_thresh=${hr},min_pixels=${minp},max_pixels=${maxp},attn_implementation=flash_attention_2"
  accelerate launch --num_processes="${NUM_GPUS}" -m lmms_eval \
    --model qwen3_vl \
    --model_args "${model_args}" \
    --tasks "${task}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${LOG_SUFFIX_BASE}_${task}" \
    --output_path ./logs/
}

run_eval "textvqa_val"  "0.15"  "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "infovqa_val"  "0.15"  "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "chartqa"      "0.125" "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "ocrbench"     "0.15"  "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "docvqa_val"   "0.125" "0.10" "${MIN_DOC}" "${MAX_DOC}"
run_eval "vstar_bench"        "0.05" "0.05" "${MIN_HR}" "${MAX_HR}"
run_eval "mmerealworld_lite"  "0.05" "0.05" "${MIN_HR}" "${MAX_HR}"
run_eval "hrbench"            "0.05" "0.05" "${MIN_HR}" "${MAX_HR}"
