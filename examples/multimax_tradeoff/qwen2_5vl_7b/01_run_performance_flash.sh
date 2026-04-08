#!/bin/bash
# Multi-max tradeoff — performance pass (full samples, flash attention)
# for Qwen2.5-VL-7B baseline vs Q-Zoom (Stage 3).
#
# Usage:
#   bash 01_run_performance_flash.sh \
#     CODE_ROOT=/path/to/Q-Zoom \
#     STAGE3_CKPT=/path/to/qzoom-qwen2_5vl-7b \
#     GPU_IDS=0,1,2,3 NUM_PROCESSES=4 SKIP_EXISTING=1
#
# Required env: conda env qzoom-q25 (transformers==4.51.3) — see ../../../README.md.

set -euo pipefail

for kv in "$@"; do
  if [[ "${kv}" == *=* ]]; then
    export "${kv}"
  else
    echo "[TradeoffPerf][Error] Unsupported argument: ${kv}. Use KEY=VALUE format."
    exit 1
  fi
done

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CODE_ROOT}/logs/qwen2_5vl_7b_multimax_tradeoff/performance}"
RUN_TAG="${RUN_TAG:-perf_$(date +%Y%m%d_%H%M%S)}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_ROOT}/${RUN_TAG}_manifest.jsonl}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PERFORMANCE_LIMIT="${PERFORMANCE_LIMIT:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

BASELINE_CKPT="${BASELINE_CKPT:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_ROOT="${MODEL_ROOT:-${CODE_ROOT}/output}"
STAGE3_CKPT="${STAGE3_CKPT:-${MODEL_ROOT}/qzoom-qwen2_5vl-7b}"

DOC_TASKS="${DOC_TASKS:-docvqa_val,chartqa,ocrbench,infovqa_val,textvqa_val}"
HR_TASKS="${HR_TASKS:-vstar_bench,mmerealworld_lite,hrbench}"
IFS=',' read -r -a DOC_ARR <<< "${DOC_TASKS}"
IFS=',' read -r -a HR_ARR <<< "${HR_TASKS}"

# Baseline uses min=256 by default; min is auto-adjusted to 128 when max=256.
BASELINE_MAX_TOKENS="${BASELINE_MAX_TOKENS:-576,1024,2048,4096}"
# Q-Zoom (stage3) sweep over the full max-token grid.
ROI_MAX_TOKENS="${ROI_MAX_TOKENS:-256,384,576,1024,2048,4096}"
IFS=',' read -r -a BASE_MAX_ARR <<< "${BASELINE_MAX_TOKENS}"
IFS=',' read -r -a ROI_MAX_ARR <<< "${ROI_MAX_TOKENS}"

mkdir -p "${OUTPUT_ROOT}"
rm -f "${MANIFEST_PATH}"

export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME_PATH}/.cache/huggingface}"
export HF_HOME="${HF_HOME:-${HOME_PATH}/.cache/huggingface}"
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/lmms-eval:${PYTHONPATH:-}"

# Qwen2.5-VL: patch_size=28
PATCH_SIZE=28
tokens_to_pixels() { echo $(( $1 * ${PATCH_SIZE} * ${PATCH_SIZE} )); }

get_group() {
  local t="$1"
  case "${t}" in
    docvqa_val|chartqa|ocrbench|infovqa_val|textvqa_val) echo "doc_ocr" ;;
    *) echo "hr" ;;
  esac
}

# Per-task hyperparameters for Q-Zoom (Stage 3).
#
# Two regimes:
#   * max_tokens <  2048  -> per-task tuned thresholds (low-res sweep)
#   * max_tokens >= 2048  -> Tab2 thresholds matched to the main paper setting
get_stage_thresholds() {
  local variant="$1"
  local task="$2"
  local max_tokens="${3:-0}"
  local roi="0.00"
  local high="0.00"
  local two_stage="False"
  if [[ "${variant}" == "baseline" ]]; then
    echo "${roi} ${high} ${two_stage}"
    return
  fi
  two_stage="True"

  if [[ "${max_tokens}" -ge 2048 ]]; then
    roi="0.04"
    high="0.02"
    if [[ "${task}" == "vstar_bench" ]]; then
      roi="0.03"
    fi
    echo "${roi} ${high} ${two_stage}"
    return
  fi

  case "${task}" in
    textvqa_val|infovqa_val) roi="0.15" ;;
    chartqa|docvqa_val)      roi="0.125" ;;
    ocrbench)                roi="0.15" ;;
    vstar_bench|mmerealworld_lite) roi="0.05" ;;
    hrbench) roi="0.03" ;;
    *) roi="0.05" ;;
  esac

  case "${task}" in
    vstar_bench|mmerealworld_lite|hrbench) high="0.025" ;;
    *) high="0.05" ;;
  esac
  echo "${roi} ${high} ${two_stage}"
}

run_one() {
  local variant="$1"
  local ckpt="$2"
  local min_tokens="$3"
  local max_tokens="$4"
  local task="$5"

  local min_t="${min_tokens}"
  # rule: when max tokens is 256, use min tokens 128.
  if [[ "${max_tokens}" == "256" ]]; then
    min_t=128
  fi
  # rule: when max_tokens >= 2048 and not baseline, use min=512 for HR tasks
  # only (Tab2 setting). Doc/OCR tasks keep min=256 at all resolutions.
  local group
  group="$(get_group "${task}")"
  if [[ "${max_tokens}" -ge 2048 ]] && [[ "${variant}" != "baseline" ]] && [[ "${group}" == "hr" ]]; then
    min_t=512
  fi
  # rule: Q-Zoom + OCRBench always uses min=128 tokens.
  if [[ "${variant}" == "stage3" && "${task}" == "ocrbench" ]]; then
    min_t=128
  fi

  local min_pixels max_pixels
  min_pixels="$(tokens_to_pixels "${min_t}")"
  max_pixels="$(tokens_to_pixels "${max_tokens}")"

  local setting_id="min${min_t}_max${max_tokens}"
  local task_dir="${OUTPUT_ROOT}/${variant}/${setting_id}/${task}"
  mkdir -p "${task_dir}"

  local existing=""
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    existing="$(find "${task_dir}" -type f -name '*_results.json' | sort | tail -n 1 || true)"
  fi

  local run_suffix="${RUN_TAG}_${variant}_${setting_id}_${task}"
  if [[ -n "${existing}" ]]; then
    echo "[TradeoffPerf][SkipTask] ${variant}/${setting_id}/${task} -> ${existing}"
  else
    read -r roi_conf high_res two_stage <<< "$(get_stage_thresholds "${variant}" "${task}" "${max_tokens}")"
    local model_args="pretrained=${ckpt},device_map=auto,two_stage_roi=${two_stage},roi_conf_thresh=${roi_conf},high_res_thresh=${high_res},min_pixels=${min_pixels},max_pixels=${max_pixels},attn_implementation=${ATTN_IMPLEMENTATION}"
    local limit_args=()
    if [[ "${PERFORMANCE_LIMIT}" =~ ^[0-9]+$ ]] && [[ "${PERFORMANCE_LIMIT}" -gt 0 ]]; then
      limit_args=(--limit "${PERFORMANCE_LIMIT}")
    fi

    echo "[TradeoffPerf] run ${variant} ${setting_id} ${task} min=${min_pixels} max=${max_pixels}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch --num_processes="${NUM_PROCESSES}" -m lmms_eval \
      --model qwen2_5_vl \
      --model_args "${model_args}" \
      --tasks "${task}" \
      --batch_size "${BATCH_SIZE}" \
      "${limit_args[@]}" \
      --log_samples \
      --log_samples_suffix "${run_suffix}" \
      --output_path "${task_dir}"
  fi

  local result_file
  result_file="$(find "${task_dir}" -type f -name '*_results.json' | sort | tail -n 1 || true)"
  if [[ -z "${result_file}" ]]; then
    echo "[TradeoffPerf][Error] result json not found for ${variant}/${setting_id}/${task}"
    exit 2
  fi

  VARIANT="${variant}" CKPT="${ckpt}" TASK="${task}" GROUP="${group}" MIN_TOKENS="${min_t}" MAX_TOKENS="${max_tokens}" MIN_PIXELS="${min_pixels}" MAX_PIXELS="${max_pixels}" RESULT_FILE="${result_file}" MANIFEST_PATH="${MANIFEST_PATH}" python - <<'PY'
import json
import os

row = {
    "variant": os.environ["VARIANT"],
    "checkpoint": os.environ["CKPT"],
    "task": os.environ["TASK"],
    "group": os.environ["GROUP"],
    "min_tokens": int(os.environ["MIN_TOKENS"]),
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "min_pixels": int(os.environ["MIN_PIXELS"]),
    "max_pixels": int(os.environ["MAX_PIXELS"]),
    "results_file": os.environ["RESULT_FILE"],
}
with open(os.environ["MANIFEST_PATH"], "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("[TradeoffPerf][Manifest+]", json.dumps(row, ensure_ascii=False))
PY
}

for t in "${DOC_ARR[@]}" "${HR_ARR[@]}"; do
  t="$(echo "${t}" | xargs)"
  [[ -z "${t}" ]] && continue
  for m in "${BASE_MAX_ARR[@]}"; do
    run_one "baseline" "${BASELINE_CKPT}" 256 "${m}" "${t}"
  done
  for m in "${ROI_MAX_ARR[@]}"; do
    run_one "stage3" "${STAGE3_CKPT}" 256 "${m}" "${t}"
  done
done

echo "[TradeoffPerf] Done. Manifest: ${MANIFEST_PATH}"
