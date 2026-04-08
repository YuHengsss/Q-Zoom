# Stage 3 — High-resolution gating refinement

Stage 3 starts from the Stage-2 checkpoint and trains **only the
high-resolution gating network** — the `high_res_layers` and the
`high_res_head` linear classifier that predicts whether a sample needs
the high-resolution ROI refinement pass.

**Frozen** in Stage 3: the base LLM (all decoder layers), `lm_head`,
the TWIG branch (carried over from Stage 1), and the vision encoder.
Only the high-res gating module is updated.

This is the `--enable_high_res True` + `--roi_post_training False`
code path in `train_qwen.py`. The `--tune_mm_llm True` shell flag is a
no-op in this stage — `train_qwen.py` overrides it with the global
freeze + selective unfreeze of the gating module only.

```bash
bash 01_build_stage3_data.sh
bash 02_train_stage3_model.sh
bash 03_eval_stage3_model.sh
```

Set `STAGE2_CKPT` to point at the Stage-2 output you want to refine, and
`ROI_DATA_PATH` to the pkl produced by `01_build_stage3_data.sh`.

> **🚀 Or skip training entirely — use our released Stage-3 checkpoint.**
> The final Stage-3 weights for all three supported backbones are on
> Hugging Face. To just *evaluate* them with `03_eval_stage3_model.sh`:
>
> | Backbone | HF model id |
> |---|---|
> | Qwen2.5-VL-3B | [`YuhengSSS/Q-Zoom-Qwen2.5VL-3B`](https://huggingface.co/YuhengSSS/Q-Zoom-Qwen2.5VL-3B) |
> | Qwen2.5-VL-7B | [`YuhengSSS/Q-Zoom-Qwen2.5VL-7B`](https://huggingface.co/YuhengSSS/Q-Zoom-Qwen2.5VL-7B) |
> | Qwen3-VL-4B   | [`YuhengSSS/Q-Zoom-Qwen3VL-4B`](https://huggingface.co/YuhengSSS/Q-Zoom-Qwen3VL-4B)   |
>
> ```bash
> huggingface-cli download YuhengSSS/Q-Zoom-Qwen2.5VL-7B \
>   --local-dir ./checkpoints/Q-Zoom-Qwen2.5VL-7B \
>   --local-dir-use-symlinks False
>
> CHECKPOINT_PATH=./checkpoints/Q-Zoom-Qwen2.5VL-7B \
>   bash 03_eval_stage3_model.sh
> ```

> **🚀 Skip step 1 — download the prepared Stage-3 ROI pkl.** Building
> the Stage-3 mixture from scratch requires running the Stage-2 LLM
> over a fresh question pool, which is slow. We host the prepared
> per-backbone Stage-3 pkls at
> <https://huggingface.co/datasets/YuhengSSS/Q-Zoom-Training>:
>
> | Backbone | File |
> |---|---|
> | Qwen2.5-VL-3B | `qwen2_5vl_3b_stage3.pkl` |
> | Qwen2.5-VL-7B | `qwen2_5vl_7b_stage3.pkl` |
> | Qwen3-VL-4B   | `qwen3vl_4b_stage3.pkl` |
>
> Download and skip straight to step 2:
>
> ```bash
> huggingface-cli download YuhengSSS/Q-Zoom-Training \
>   --repo-type dataset --local-dir "${DATA_ROOT}" --local-dir-use-symlinks False \
>   --include "qwen2_5vl_7b_stage3.pkl"
>
> ROI_DATA_PATH=${DATA_ROOT}/qwen2_5vl_7b_stage3.pkl \
>   bash 02_train_stage3_model.sh
> ```
