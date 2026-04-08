#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Q-Zoom Stage3] Step 1/3: Build Stage-3 data"
bash "${SCRIPT_DIR}/01_build_stage3_data.sh"

echo "[Q-Zoom Stage3] Step 2/3: Train Stage-3 model"
bash "${SCRIPT_DIR}/02_train_stage3_model.sh"

echo "[Q-Zoom Stage3] Step 3/3: Evaluate Stage-3 model"
bash "${SCRIPT_DIR}/03_eval_stage3_model.sh"

echo "[Q-Zoom Stage3] All steps completed."
