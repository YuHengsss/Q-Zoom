#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Q-Zoom Stage2.5] Step 1/2: Re-fit TWIG branch on Stage-2 LLM"
bash "${SCRIPT_DIR}/01_train_stage2_5_model.sh"

echo "[Q-Zoom Stage2.5] Step 2/2: Evaluate Stage-2.5 model"
bash "${SCRIPT_DIR}/02_eval_stage2_5_model.sh"

echo "[Q-Zoom Stage2.5] All steps completed."
