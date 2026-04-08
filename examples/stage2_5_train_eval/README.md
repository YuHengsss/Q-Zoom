# Stage 2.5 — Re-fit the TWIG branch on the Stage-2 backbone (Qwen3-VL only)

> **Use this stage only for Qwen3-VL backbones.**
> Qwen2.5-VL backbones go straight from Stage 2 → Stage 3.

## Why Stage 2.5 exists

Stage 2 is an LLM-only post-SFT pass (`--roi_post_training True` — see
`../stage2_train_eval/`). For Qwen3-VL backbones we observed that this
SFT pass shifts the LLM's intermediate hidden states enough that the
**TWIG gating branch** trained in Stage 1 is no longer well-aligned with
the new feature distribution. The base layers under `twig_K` look
slightly different to the twig layers than they did at Stage 1 time, so
the gating predictions get noisy and downstream Stage 3 training is
harder to stabilize.

The fix is a short, cheap **re-fit** of the TWIG branch only:

1. Take the Stage-2 LLM checkpoint as the new backbone.
2. Generate fresh pseudo ROI labels by running this Stage-2 backbone over
   a small fixed-resolution question pool.
3. Train **only the `twig_T` twig layers** for a few iterations on those
   pseudo labels, with a small `roi_samples` budget. The LLM stays
   frozen — this is the same code path as Stage 1
   (`--enable_twig=True`, `--roi_post_training=False`,
   `--enable_high_res=False`), with the additional flag
   `--is_2_5_stage True` so the data loader picks the small fixed-res
   subset.

The output is a checkpoint that has the **Stage-2 LLM** + a **re-aligned
TWIG branch**, ready to feed into Stage 3.

For Qwen2.5-VL backbones we did **not** observe a meaningful drift after
Stage 2, so this re-fit is unnecessary and the canonical pipeline is
just Stage 1 → Stage 2 → Stage 3.

## What gets trained

| Trainable | Frozen |
|---|---|
| `twig_T` twig layers (the gating branch) | Base LLM (carried over from Stage 2), `lm_head`, vision encoder, high-res gating |

This is the **same** trainable set as Stage 1 — we simply re-run Stage-1
training on top of the Stage-2 LLM.

## Steps

```bash
bash 01_train_stage2_5_model.sh    # writes ${CODE_ROOT}/output/<run_name>
bash 02_eval_stage2_5_model.sh
```

(The pseudo-label generation is folded into the training script via
`--roi_data_path`. Generate the small fixed-resolution pkl ahead of time
with `standardized_pipeline/stage1` against the Stage-2 LLM.)

## Inputs you must set

| Variable | Default | Meaning |
|---|---|---|
| `STAGE2_CKPT` | `${CODE_ROOT}/output/qzoom-qwen3vl-4b-K24T3-stage2` | Stage-2 LLM checkpoint |
| `ROI_DATA_PATH` | `${DATA_ROOT}/stage1_pseudo_qwen3vl_4b_stage2_576res.pkl` | Fixed-res pseudo-label pkl generated against the Stage-2 LLM |
| `RUN_NAME` | `qzoom-qwen3vl-4b-K24T3-stage2.5` | Output checkpoint dir |
| `TWIG_K` / `TWIG_T` | `24` / `3` | Must match the Stage-2 checkpoint |
| `LR` | `1e-4` | Learning rate (same as Stage 1) |
| `BATCH_SIZE` | `16` | Per-device batch |
| `GRAD_ACCUM_STEPS` | `4` | |
| `ROI_SAMPLES` | `20000` | Number of ROI samples to fine-tune on (small budget) |
| `NUM_GPUS` | `4` | |

## Conda environment

This stage trains a Qwen3-VL backbone, so use the `qzoom-q3` env
(`transformers==4.57.0`):

```bash
conda activate qzoom-q3
```

## Where Stage 2.5 fits in the pipeline

```
Stage 1  (TWIG init on base LLM, qzoom-qwen3vl-4b-K24T3-stage1)
   │
   ▼
Stage 2  (LLM post-SFT, qzoom-qwen3vl-4b-K24T3-stage2)
   │
   ▼
Stage 2.5  ← THIS STAGE (re-fit TWIG on the Stage-2 LLM, Qwen3-VL only)
   │
   ▼
Stage 3  (high-res gating refinement, qzoom-qwen3vl-4b)
```
