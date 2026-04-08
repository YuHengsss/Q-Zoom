#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Q-Zoom Stage1] Step 1/3: Generate pseudo labels"
bash "${SCRIPT_DIR}/01_make_pseudo_labels.sh"

echo "[Q-Zoom Stage1] Step 2/3: Train Stage-1 model"
bash "${SCRIPT_DIR}/02_train_stage1_model.sh"

echo "[Q-Zoom Stage1] Step 3/3: Evaluate Stage-1 model"
bash "${SCRIPT_DIR}/03_eval_stage1_model.sh"

echo "[Q-Zoom Stage1] All steps completed."
