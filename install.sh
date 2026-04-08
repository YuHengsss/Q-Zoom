#!/bin/bash
# Q-Zoom installation script.
#
# Q-Zoom requires DIFFERENT transformers versions depending on the backbone:
#
#   Qwen2.5-VL  ->  transformers==4.51.3
#   Qwen3-VL    ->  transformers==4.57.1   (4.57.0 is yanked on PyPI)
#
# These two versions are NOT mutually compatible. We recommend two
# independent conda environments — one per backbone — and pass the family
# you want as the first argument to this script.
#
# Usage:
#   bash install.sh qwen2_5vl   # creates env "qzoom-q25", pins transformers==4.51.3
#   bash install.sh qwen3vl     # creates env "qzoom-q3",  pins transformers==4.57.1
#
# Without an argument, the script defaults to qwen3vl (the version pinned
# in requirements_roi_training.txt).

set -euo pipefail

FAMILY="${1:-qwen3vl}"

case "${FAMILY}" in
  qwen2_5vl|q25|qwen2.5vl)
    ENV_NAME="qzoom-q25"
    TRANSFORMERS_VERSION="4.51.3"
    ;;
  qwen3vl|q3)
    ENV_NAME="qzoom-q3"
    TRANSFORMERS_VERSION="4.57.1"
    ;;
  *)
    echo "Unknown model family: ${FAMILY}"
    echo "Valid choices: qwen2_5vl | qwen3vl"
    exit 1
    ;;
esac

# CUDA toolkit. flash-attn build needs a CUDA toolchain >= 11.7; the
# default `/usr/bin/nvcc` on many distros is too old. Override with a
# local CUDA install that matches your PyTorch build (PyTorch 2.4 wheels
# from PyPI ship cu121). Adjust if your PyTorch wheel is built against a
# different CUDA major version.
CUDA_HOME_DEFAULT="/usr/local/cuda-12.1"
export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DEFAULT}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "============================================"
echo "Q-Zoom installer"
echo "  Family:                ${FAMILY}"
echo "  Conda env:             ${ENV_NAME}"
echo "  transformers version:  ${TRANSFORMERS_VERSION}"
echo "  CUDA_HOME:             ${CUDA_HOME}"
echo "============================================"

# 1. Create / activate the conda environment
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 2. Install pinned dependencies. We deliberately install flash-attn
#    BEFORE the rest, using a pre-built upstream wheel — pip's source
#    build of flash-attn requires a working nvcc and is slow / fragile.
#    Skip the flash-attn entry in requirements_roi_training.txt for the
#    main install pass, then install it from the wheel below.
PY_TAG="cp310"
TORCH_TAG="torch2.4"
CU_TAG="cu12"
ABI_TAG="cxx11abiFALSE"
FLASH_WHEEL="flash_attn-2.7.3+${CU_TAG}${TORCH_TAG}${ABI_TAG}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/${FLASH_WHEEL}"

# Strip the flash-attn line so the requirements pass does not try to
# build it from source.
TMP_REQ="$(mktemp -t qzoom-req-XXXX.txt)"
trap 'rm -f "${TMP_REQ}"' EXIT
grep -vE '^[[:space:]]*flash-attn' requirements_roi_training.txt > "${TMP_REQ}"
pip install -r "${TMP_REQ}"

# 3. Force the correct transformers version (overrides whatever the
#    requirements file pinned).
pip install --force-reinstall --no-deps "transformers==${TRANSFORMERS_VERSION}"

# 4. Editable installs of the local lmms-eval and qwen-vl-utils
pip install -e ./lmms-eval
pip install -e ./qwen-vl-utils

# 5. flash-attn 2.7.3 from upstream pre-built wheel.
echo
echo "Installing flash-attn 2.7.3 from upstream pre-built wheel..."
WHEEL_DIR="$(mktemp -d -t qzoom-flash-XXXX)"
trap 'rm -rf "${TMP_REQ}" "${WHEEL_DIR}"' EXIT
if ! curl -fL --retry 3 -o "${WHEEL_DIR}/${FLASH_WHEEL}" "${FLASH_URL}"; then
  echo "Failed to download ${FLASH_URL}"
  echo "Falling back to source build (requires CUDA_HOME=${CUDA_HOME})..."
  pip install "flash-attn==2.7.3" --no-build-isolation
else
  pip install "${WHEEL_DIR}/${FLASH_WHEEL}"
fi

echo
echo "Done. Activate this env later with:  conda activate ${ENV_NAME}"
echo "Quick sanity check:"
echo "  python -c 'import torch, transformers, deepspeed, flash_attn; \\"
echo "             print(torch.__version__, transformers.__version__, deepspeed.__version__, flash_attn.__version__)'"
