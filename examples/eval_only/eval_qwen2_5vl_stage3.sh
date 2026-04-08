#!/bin/bash
# Evaluate a Qwen2.5-VL Q-Zoom (Stage 3) checkpoint on the full
# Doc/OCR + HR/Vision benchmark suite.
#
# Hyper-parameters mirror the multi-max trade-off table from the paper:
#   - Doc/OCR (max_tokens=576): per-task `roi_conf_thresh` in {0.125, 0.15},
#     `high_res_thresh=0.05`, `min_tokens=256` (ocrbench is special: 128).
#   - HR     (max_tokens=4096): "Tab2" regime — `roi_conf_thresh` in
#     {0.03 (vstar), 0.04 (general)}, `high_res_thresh=0.02`, `min_tokens=512`.
# See examples/eval_only/README.md for the full hyper-parameter table.

set -euo pipefail

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"

export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME_PATH}/.cache/huggingface}"
export HF_HOME="${HF_HOME:-${HOME_PATH}/.cache/huggingface}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"

cd "${CODE_ROOT}"
mkdir -p ./logs

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Please set CHECKPOINT_PATH to your Q-Zoom checkpoint}"
LOG_SUFFIX_BASE="${LOG_SUFFIX_BASE:-qzoom_qwen2_5vl_eval}"
NUM_GPUS="${NUM_GPUS:-4}"
TWO_STAGE_ROI="${TWO_STAGE_ROI:-True}"

# Qwen2.5-VL: patch_size=28
PATCH=28

# Per-task token budgets (interpreted as # of vision tokens, not pixels —
# the function below converts to pixel counts via PATCH*PATCH).
DOC_MIN_TOKENS=256       # Doc/OCR default
DOC_MAX_TOKENS=576
OCR_MIN_TOKENS=128       # ocrbench special-case
HR_MIN_TOKENS=512        # HR Tab2 regime
HR_MAX_TOKENS=4096

run_eval () {
  local task="$1" conf="$2" hr="$3" min_tokens="$4" max_tokens="$5"
  local minp=$(( min_tokens * PATCH * PATCH ))
  local maxp=$(( max_tokens * PATCH * PATCH ))
  local model_args="pretrained=${CHECKPOINT_PATH}"
  model_args+=",device_map=auto"
  model_args+=",two_stage_roi=${TWO_STAGE_ROI}"
  model_args+=",roi_conf_thresh=${conf}"
  model_args+=",high_res_thresh=${hr}"
  model_args+=",min_pixels=${minp}"
  model_args+=",max_pixels=${maxp}"
  model_args+=",attn_implementation=flash_attention_2"
  accelerate launch --num_processes="${NUM_GPUS}" -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "${model_args}" \
    --tasks "${task}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${LOG_SUFFIX_BASE}_${task}" \
    --output_path ./logs/
}

# ----------------------------------------------------------------------
# Doc/OCR (default regime, max_tokens=576)
# ----------------------------------------------------------------------
run_eval textvqa_val  0.15  0.05 "${DOC_MIN_TOKENS}" "${DOC_MAX_TOKENS}"
run_eval infovqa_val  0.15  0.05 "${DOC_MIN_TOKENS}" "${DOC_MAX_TOKENS}"
run_eval chartqa      0.125 0.05 "${DOC_MIN_TOKENS}" "${DOC_MAX_TOKENS}"
run_eval ocrbench     0.15  0.05 "${OCR_MIN_TOKENS}" "${DOC_MAX_TOKENS}"
run_eval docvqa_val   0.125 0.05 "${DOC_MIN_TOKENS}" "${DOC_MAX_TOKENS}"

# ----------------------------------------------------------------------
# HR / Vision-centric (Tab2 regime, max_tokens=4096, min_tokens=512)
# ----------------------------------------------------------------------
run_eval vstar_bench       0.03 0.02 "${HR_MIN_TOKENS}" "${HR_MAX_TOKENS}"
run_eval mmerealworld_lite 0.04 0.02 "${HR_MIN_TOKENS}" "${HR_MAX_TOKENS}"
run_eval hrbench           0.04 0.02 "${HR_MIN_TOKENS}" "${HR_MAX_TOKENS}"
