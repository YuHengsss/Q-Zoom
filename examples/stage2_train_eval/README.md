# Stage 2 — LLM post-SFT on judged base/ROI data

Stage 2 takes the Stage-1 checkpoint and fine-tunes **the LLM only** (all
decoder layers + `lm_head`) on a post-SFT dataset built by judging the
base model and the Stage-1 ROI model against each other on a fresh
question pool. The TWIG branch from Stage 1 stays **frozen** and is
carried over to subsequent stages unchanged.

This is the `--roi_post_training True` code path in `train_qwen.py`.

## Steps

1. **Build the post-SFT data** with `standardized_pipeline/stage2`. This
   runs both the base and the Stage-1 model, judges where ROI helps/hurts,
   and writes a single SFT JSONL.
2. **Train** Stage-2 starting from the Stage-1 checkpoint.
3. **Evaluate** the resulting checkpoint.

```bash
bash 01_build_stage2_data.sh   # writes ${DATA_ROOT}/<post_sft>.jsonl
bash 02_train_stage2_model.sh
bash 03_eval_stage2_model.sh
```

> **🚀 Skip step 1 — download the prepared post-SFT JSONL.** Building the
> Stage-2 mixture requires running both the base and the Stage-1 ROI
> models on a fresh question pool and judging them against each other,
> which is slow. We host the prepared per-backbone post-SFT JSONLs at
> <https://huggingface.co/datasets/YuhengSSS/Q-Zoom-Training>:
>
> | Backbone | File |
> |---|---|
> | Qwen2.5-VL-3B | `qwen2_5vl_3b_stage2.jsonl` |
> | Qwen2.5-VL-7B | `qwen2_5vl_7b_stage2.jsonl` |
> | Qwen3-VL-4B   | `qwen3vl_4b_stage2.jsonl` |
>
> Download the one that matches your backbone, save it as the
> `POST_SFT_JSONL` path expected by `02_train_stage2_model.sh`, and skip
> straight to step 2:
>
> ```bash
> huggingface-cli download YuhengSSS/Q-Zoom-Training \
>   --repo-type dataset --local-dir "${DATA_ROOT}" --local-dir-use-symlinks False \
>   --include "qwen2_5vl_7b_stage2.jsonl"
>
> POST_SFT_JSONL=${DATA_ROOT}/qwen2_5vl_7b_stage2.jsonl \
> BACKBONE_TAG=qwen2_5vl_7b \
>   bash 02_train_stage2_model.sh
> ```

## Inputs you must set

| Variable | Required | Meaning |
|---|---|---|
| `STAGE1_CKPT` | yes | Path to the Stage-1 checkpoint to bootstrap from. |
| `BASE_MODEL_PATH` | yes | Base VLM used as the "without-ROI" judge baseline. |
| `BACKBONE_TAG` | yes | Short tag identifying the backbone (e.g. `qwen2_5vl_3b`, `qwen2_5vl_7b`, `qwen3vl_4b`). Used to namespace the data files so different backbones do not overwrite each other. |
| `POST_SFT_JSONL` | recommended | Output of `01_build_stage2_data.sh`; consumed by training. |

### Why `BACKBONE_TAG` matters

The Stage-2 data builder runs the base VLM and the Stage-1 ROI model
over the same question pool, judges them against each other, and writes
**three** files:

| Output | Default filename |
|---|---|
| `BASE_OUTPUT`     | `${DATA_ROOT}/stage2_base_${BACKBONE_TAG}.pkl` |
| `ROI_OUTPUT`      | `${DATA_ROOT}/stage2_roi_${BACKBONE_TAG}.pkl` |
| `POST_SFT_JSONL`  | `${DATA_ROOT}/qzoom_post_sft_stage2_${BACKBONE_TAG}.jsonl` |

Each backbone produces its **own** judging — Qwen2.5-VL-7B base
responses are very different from Qwen3-VL-4B base responses, and the
ROI/base agreement signal that drives the post-SFT mixture is
backbone-specific. Always run `01_build_stage2_data.sh` once per
backbone with a different `BACKBONE_TAG`, so the three files above are
not silently overwritten by a later run.

Example:

```bash
# Build Stage-2 data for Qwen2.5-VL-7B
BACKBONE_TAG=qwen2_5vl_7b \
BASE_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
STAGE1_CKPT=${CODE_ROOT}/output/qzoom-qwen2_5vl-7b-K18T3-stage1 \
bash 01_build_stage2_data.sh

# Build Stage-2 data for Qwen3-VL-4B (do NOT reuse the 7B files)
BACKBONE_TAG=qwen3vl_4b \
BASE_MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
STAGE1_CKPT=${CODE_ROOT}/output/qzoom-qwen3vl-4b-K24T3-stage1 \
bash 01_build_stage2_data.sh
```
