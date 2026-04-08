#!/bin/bash
# Stage 2 training: LLM post-SFT (all decoder layers + lm_head).
# The TWIG branch from Stage 1 is carried over and stays FROZEN.

set -euo pipefail

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"

HOME_PATH="${HOME_PATH:-/path/to/home}"
CODE_ROOT="${CODE_ROOT:-${HOME_PATH}/Q-Zoom}"
DATA_ROOT="${DATA_ROOT:-${HOME_PATH}/datasets}"

deepspeed="${CODE_ROOT}/qwen-vl-finetune/scripts/zero2.json"
entry_file="${CODE_ROOT}/qwen-vl-finetune/qwenvl/train/train_qwen.py"

# Stage-1 checkpoint serves as the starting point for Stage-2
llm="${STAGE1_CKPT:-${CODE_ROOT}/output/qzoom-qwen2_5vl-7b-K18T3-stage1}"

lr="${LR:-1e-6}"
batch_size="${BATCH_SIZE:-1}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-16}"
twig_K="${TWIG_K:-18}"
twig_T="${TWIG_T:-3}"
nproc="${NUM_GPUS:-4}"

# Stage-2 uses the post-SFT JSONL produced by 01_build_stage2_data.sh.
# The dataset name (DATASET_USE) must match a registry entry in
# qwen-vl-finetune/qwenvl/data/__init__.py that points at this file.
datasets="${DATASET_USE:-qzoom_stage2_post_sft}"

run_name="${RUN_NAME:-qzoom-qwen2_5vl-7b-K${twig_K}T${twig_T}-stage2}"
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
    --fix_res False \
    --roi_samples -1 \
    --transition_mode False \
    --enable_high_res False \
    --roi_post_training True \
    --reuse_src_pos True \
    --add_noise_to_roi True
"

cd "${CODE_ROOT}"
torchrun --nproc_per_node="${nproc}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  ${entry_file} ${args}
