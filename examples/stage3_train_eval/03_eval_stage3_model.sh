#!/bin/bash
# Evaluate a Stage-3 Q-Zoom checkpoint on the full Doc/OCR + HR/Vision
# benchmark suite. Set MODEL_FAMILY to switch between Qwen2.5-VL and
# Qwen3-VL — each family has its own per-task threshold table mirroring
# the multi-max trade-off setting from the paper.

set -euo pipefail

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"

export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME_PATH}/.cache/huggingface}"
export HF_HOME="${HF_HOME:-${HOME_PATH}/.cache/huggingface}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"

cd "${CODE_ROOT}"
mkdir -p ./logs

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CODE_ROOT}/output/qzoom-qwen2_5vl-7b-K18T3-stage3}"
LOG_SUFFIX_BASE="${LOG_SUFFIX_BASE:-qzoom_stage3}"
NUM_GPUS="${NUM_GPUS:-4}"
TWO_STAGE_ROI="${TWO_STAGE_ROI:-True}"

# qwen2_5vl | qwen3vl. Determines patch_size, model class, and the
# per-task threshold table.
MODEL_FAMILY="${MODEL_FAMILY:-qwen2_5vl}"

case "${MODEL_FAMILY}" in
  qwen2_5vl|q25)
    MODEL_NAME="qwen2_5_vl"
    PATCH=28
    # Doc/OCR thresholds
    TEXTVQA_CONF=0.15  ; TEXTVQA_HR=0.05
    INFOVQA_CONF=0.15  ; INFOVQA_HR=0.05
    CHARTQA_CONF=0.125 ; CHARTQA_HR=0.05
    OCRBENCH_CONF=0.15 ; OCRBENCH_HR=0.05
    DOCVQA_CONF=0.125  ; DOCVQA_HR=0.05
    # HR Tab2 thresholds (max_tokens >= 2048)
    VSTAR_CONF=0.03    ; VSTAR_HR=0.02
    MMERW_CONF=0.04    ; MMERW_HR=0.02
    HRBENCH_CONF=0.04  ; HRBENCH_HR=0.02
    ;;
  qwen3vl|q3)
    MODEL_NAME="qwen3_vl"
    PATCH=32
    # Doc/OCR thresholds
    TEXTVQA_CONF=0.15  ; TEXTVQA_HR=0.10
    INFOVQA_CONF=0.15  ; INFOVQA_HR=0.10
    CHARTQA_CONF=0.15  ; CHARTQA_HR=0.10
    OCRBENCH_CONF=0.125; OCRBENCH_HR=0.10
    DOCVQA_CONF=0.15   ; DOCVQA_HR=0.10
    # HR Tab2 thresholds
    VSTAR_CONF=0.10    ; VSTAR_HR=0.025
    MMERW_CONF=0.10    ; MMERW_HR=0.025
    HRBENCH_CONF=0.10  ; HRBENCH_HR=0.025
    ;;
  *)
    echo "Unknown MODEL_FAMILY: ${MODEL_FAMILY} (valid: qwen2_5vl | qwen3vl)" >&2
    exit 1
    ;;
esac

DOC_MIN_TOKENS=256
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
    --model "${MODEL_NAME}" \
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
run_eval textvqa_val  "${TEXTVQA_CONF}"  "${TEXTVQA_HR}"  "${DOC_MIN_TOKENS}" "${DOC_MAX_TOKENS}"
run_eval infovqa_val  "${INFOVQA_CONF}"  "${INFOVQA_HR}"  "${DOC_MIN_TOKENS}" "${DOC_MAX_TOKENS}"
run_eval chartqa      "${CHARTQA_CONF}"  "${CHARTQA_HR}"  "${DOC_MIN_TOKENS}" "${DOC_MAX_TOKENS}"
run_eval ocrbench     "${OCRBENCH_CONF}" "${OCRBENCH_HR}" "${OCR_MIN_TOKENS}" "${DOC_MAX_TOKENS}"
run_eval docvqa_val   "${DOCVQA_CONF}"   "${DOCVQA_HR}"   "${DOC_MIN_TOKENS}" "${DOC_MAX_TOKENS}"

# ----------------------------------------------------------------------
# HR / Vision-centric (Tab2 regime, max_tokens=4096, min_tokens=512)
# ----------------------------------------------------------------------
run_eval vstar_bench       "${VSTAR_CONF}"   "${VSTAR_HR}"   "${HR_MIN_TOKENS}" "${HR_MAX_TOKENS}"
run_eval mmerealworld_lite "${MMERW_CONF}"   "${MMERW_HR}"   "${HR_MIN_TOKENS}" "${HR_MAX_TOKENS}"
run_eval hrbench           "${HRBENCH_CONF}" "${HRBENCH_HR}" "${HR_MIN_TOKENS}" "${HR_MAX_TOKENS}"
