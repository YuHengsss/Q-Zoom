#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Q-Zoom Stage2] Step 1/3: Build post-SFT data"
bash "${SCRIPT_DIR}/01_build_stage2_data.sh"

echo "[Q-Zoom Stage2] Step 2/3: Train Stage-2 model"
bash "${SCRIPT_DIR}/02_train_stage2_model.sh"

echo "[Q-Zoom Stage2] Step 3/3: Evaluate Stage-2 model"
bash "${SCRIPT_DIR}/03_eval_stage2_model.sh"

echo "[Q-Zoom Stage2] All steps completed."
