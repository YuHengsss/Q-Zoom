#!/bin/bash
set -euo pipefail

# Print benchmark-wise performance for each resolution setting.
#
# Usage:
#   bash experiments/qwen2_5vl_7b_multimax_doc_hr_tradeoff/05_print_performance_matrix.sh
#   bash experiments/qwen2_5vl_7b_multimax_doc_hr_tradeoff/05_print_performance_matrix.sh RESULT_DIR=... GROUP=all
#
# Optional args:
#   RESULT_DIR=<dir>               # defaults to latest results/* in this experiment dir
#   DETAILED_CSV=<path>            # defaults to ${RESULT_DIR}/detailed_tradeoff.csv
#   GROUP=all|doc_ocr|hr           # default all
#   VARIANT_ORDER=baseline,stage1,stage3

for kv in "$@"; do
  if [[ "${kv}" == *=* ]]; then
    export "${kv}"
  else
    echo "[TradeoffPerfPrint][Error] Unsupported argument: ${kv}. Use KEY=VALUE format."
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${GROUP:-all}"
VARIANT_ORDER="${VARIANT_ORDER:-baseline,stage1,stage3}"

if [[ -z "${RESULT_DIR:-}" ]]; then
  RESULT_DIR="$(ls -1dt "${SCRIPT_DIR}"/results/* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${RESULT_DIR:-}" ]]; then
  echo "[TradeoffPerfPrint][Error] RESULT_DIR not found. Please pass RESULT_DIR=..."
  exit 1
fi

DETAILED_CSV="${DETAILED_CSV:-${RESULT_DIR}/detailed_tradeoff.csv}"
if [[ ! -f "${DETAILED_CSV}" ]]; then
  echo "[TradeoffPerfPrint][Error] Missing detailed csv: ${DETAILED_CSV}"
  exit 1
fi

DETAILED_CSV="${DETAILED_CSV}" GROUP="${GROUP}" VARIANT_ORDER="${VARIANT_ORDER}" python - <<'PY'
import csv
import os
import sys
from collections import defaultdict

detailed_csv = os.environ["DETAILED_CSV"]
group = os.environ.get("GROUP", "all").strip().lower()
variant_order = [x.strip() for x in os.environ.get("VARIANT_ORDER", "baseline,stage1,stage3").split(",") if x.strip()]

if group not in {"all", "doc_ocr", "hr"}:
    print(f"[TradeoffPerfPrint][Error] GROUP must be one of all|doc_ocr|hr, got: {group}")
    sys.exit(1)

doc_tasks = ["docvqa_val", "chartqa", "ocrbench", "infovqa_val", "textvqa_val"]
hr_tasks = ["vstar_bench", "mmerealworld_lite", "hrbench4k", "hrbench8k"]
if group == "doc_ocr":
    tasks = doc_tasks
elif group == "hr":
    tasks = hr_tasks
else:
    tasks = doc_tasks + hr_tasks

rows = []
with open(detailed_csv, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            r["min_tokens"] = int(float(r["min_tokens"]))
            r["max_tokens"] = int(float(r["max_tokens"]))
            r["accuracy"] = float(r["accuracy"]) if r["accuracy"] not in ("", None) else None
        except Exception:
            continue
        rows.append(r)

if not rows:
    print(f"[TradeoffPerfPrint][Error] No usable rows in {detailed_csv}")
    sys.exit(1)

# settings[(variant, min_t, max_t)][task] = acc
settings = defaultdict(dict)
for r in rows:
    task = r.get("task")
    if task not in tasks:
        continue
    k = (r.get("variant"), r.get("min_tokens"), r.get("max_tokens"))
    settings[k][task] = r.get("accuracy")

if not settings:
    print("[TradeoffPerfPrint][Error] No matching rows for selected GROUP/tasks.")
    sys.exit(1)

variant_rank = {v: i for i, v in enumerate(variant_order)}
sorted_keys = sorted(
    settings.keys(),
    key=lambda x: (variant_rank.get(x[0], 999), x[0], int(x[2]), int(x[1])),
)

def fmt(v):
    if v is None:
        return "NA"
    return f"{v:.2f}"

cols = ["variant", "min_t", "max_t"] + tasks
width = {c: len(c) for c in cols}
for k in sorted_keys:
    variant, min_t, max_t = k
    width["variant"] = max(width["variant"], len(str(variant)))
    width["min_t"] = max(width["min_t"], len(str(min_t)))
    width["max_t"] = max(width["max_t"], len(str(max_t)))
    for t in tasks:
        width[t] = max(width[t], len(fmt(settings[k].get(t))))

header = " ".join(
    [
        f"{'variant':<{width['variant']}}",
        f"{'min_t':>{width['min_t']}}",
        f"{'max_t':>{width['max_t']}}",
    ]
    + [f"{t:>{width[t]}}" for t in tasks]
)
print(header)
print("-" * len(header))
for k in sorted_keys:
    variant, min_t, max_t = k
    line = " ".join(
        [
            f"{variant:<{width['variant']}}",
            f"{min_t:>{width['min_t']}}",
            f"{max_t:>{width['max_t']}}",
        ]
        + [f"{fmt(settings[k].get(t)):>{width[t]}}" for t in tasks]
    )
    print(line)

print(f"\n[TradeoffPerfPrint] source={detailed_csv}")
print(f"[TradeoffPerfPrint] rows={len(sorted_keys)} settings, group={group}")
PY

