#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PERF="${SCRIPT_DIR}/01_run_performance_flash.sh"
RUN_TP="${SCRIPT_DIR}/02_run_throughput_sdpa.sh"
RUN_SUM="${SCRIPT_DIR}/03_summarize_tradeoff.py"
RUN_PLOT="${SCRIPT_DIR}/04_plot_tradeoff.py"

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"
LOG_ROOT="${LOG_ROOT:-${CODE_ROOT}/logs/qwen2_5vl_7b_multimax_tradeoff}"

RUN_ID="${RUN_ID:-tradeoff_$(date +%Y%m%d_%H%M%S)}"
PERF_MANIFEST="${PERF_MANIFEST:-${LOG_ROOT}/performance/${RUN_ID}_manifest.jsonl}"
TP_MANIFEST="${TP_MANIFEST:-${LOG_ROOT}/throughput/${RUN_ID}_manifest.jsonl}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/${RUN_ID}}"

ARGS=()
for kv in "$@"; do
  if [[ "${kv}" == *=* ]]; then
    ARGS+=("${kv}")
  else
    echo "[TradeoffRunAll][Error] Unsupported argument: ${kv}. Use KEY=VALUE format."
    exit 1
  fi
done

echo "[TradeoffRunAll] Step 1/4 — performance (flash, full samples)"
bash "${RUN_PERF}" "${ARGS[@]}" "RUN_TAG=${RUN_ID}" "MANIFEST_PATH=${PERF_MANIFEST}"

echo "[TradeoffRunAll] Step 2/4 — throughput (sdpa, 100 samples)"
bash "${RUN_TP}" "${ARGS[@]}" "RUN_TAG=${RUN_ID}" "MANIFEST_PATH=${TP_MANIFEST}"

echo "[TradeoffRunAll] Step 3/4 — summarize"
python "${RUN_SUM}" --perf-manifest "${PERF_MANIFEST}" --tp-manifest "${TP_MANIFEST}" --output-dir "${RESULT_DIR}"

echo "[TradeoffRunAll] Step 4/4 — plot"
python "${RUN_PLOT}" --global-csv "${RESULT_DIR}/global_tradeoff.csv" --output-dir "${RESULT_DIR}"

echo "[TradeoffRunAll] Done. Results: ${RESULT_DIR}"
