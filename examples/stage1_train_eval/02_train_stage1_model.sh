#!/bin/bash
# Stage 1 training: TWIG-only fine-tuning on pseudo-labelled ROI maps.

set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"
DATA_ROOT="${DATA_ROOT:-${HOME_PATH}/datasets}"

deepspeed="${CODE_ROOT}/qwen-vl-finetune/scripts/zero2.json"
entry_file="${CODE_ROOT}/qwen-vl-finetune/qwenvl/train/train_qwen.py"

llm="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
roi_data_path="${ROI_DATA_PATH:-${DATA_ROOT}/stage1results/qzoom_stage1_pseudo.pkl}"

lr="${LR:-1e-4}"
batch_size="${BATCH_SIZE:-16}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-2}"
twig_K="${TWIG_K:-18}"
twig_T="${TWIG_T:-3}"
nproc="${NUM_GPUS:-4}"
datasets="${DATASET_USE:-my_roi_dataset}"

run_name="${RUN_NAME:-qzoom-qwen2_5vl-7b-K${twig_K}T${twig_T}-stage1}"
output_dir="${CODE_ROOT}/output/${run_name}"

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 451584 \
    --min_pixels 451584 \
    --eval_strategy no \
    --save_strategy no \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard \
    --enable_twig True \
    --twig_K ${twig_K} \
    --twig_T ${twig_T} \
    --roi_branch True \
    --twig_freeze 0 \
    --roi_source qk \
    --roi_loss bce \
    --roi_super_type v1 \
    --roi_multi_head True \
    --twig_init True \
    --roi_data_path ${roi_data_path} \
    --fix_res True \
    --multi_scale_training False \
    --roi_samples -1 \
    --bg_coff 0.05 \
    --roi_binary_coeff 0.25 \
    --enable_high_res False \
    --reuse_src_pos False
"

cd "${CODE_ROOT}"
torchrun --nproc_per_node="${nproc}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  ${entry_file} ${args}
