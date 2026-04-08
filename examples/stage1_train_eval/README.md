# Stage 1 — Pseudo-labelled TWIG initialization

Stage 1 freezes **the entire base VLM** (vision encoder, LLM, lm_head) and
trains **only the `twig_T` twig layers** — the gating branch grafted on top
of layer `twig_K`. Supervision comes from pseudo ROI attention maps that
the base model itself produces over a question pool.

`twig_K` is the layer index where the TWIG branch reads from; the first
`twig_K` base layers stay frozen and are *not* part of the trainable set.

## Steps

1. **Generate pseudo labels** — run the base VLM over a dataset, save its
   ROI attention maps as supervision targets.
2. **Build a Stage-1 training pkl** from the pseudo labels.
3. **Train** the TWIG layers.
4. **Evaluate** the gated model.

```bash
bash 01_make_pseudo_labels.sh    # writes ${DATA_ROOT}/stage1results/<...>.pkl
bash 02_train_stage1_model.sh    # writes ${CODE_ROOT}/output/<run_name>
bash 03_eval_stage1_model.sh
```

Or run all three at once:

```bash
bash run_all.sh
```

> **🚀 Skip step 1 — download the prepared pseudo labels.** Step 1 takes
> hours on a multi-GPU box. We host the prepared pseudo-label pickles
> at <https://huggingface.co/datasets/YuhengSSS/Q-Zoom-Training>.
> Download the matching `qwen{2_5vl_3b,2_5vl_7b,3vl_4b}_pseudo_*_576res_185k.pkl`
> and point training straight at it:
>
> ```bash
> huggingface-cli download YuhengSSS/Q-Zoom-Training \
>   --repo-type dataset --local-dir "${DATA_ROOT}" --local-dir-use-symlinks False \
>   --include "qwen2_5vl_pseudo_7b_576res_185k.pkl"
>
> ROI_DATA_PATH=${DATA_ROOT}/qwen2_5vl_pseudo_7b_576res_185k.pkl \
>   bash 02_train_stage1_model.sh
> ```

## Key knobs

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen2.5-VL-7B-Instruct` | Base VLM (HF id or local path). |
| `TWIG_K` | `18` | Number of frozen base layers before the TWIG branch. |
| `TWIG_T` | `3` | Number of twig layers to train. |
| `BATCH_SIZE` | `16` | Per-device batch size. |
| `LR` | `1e-4` | Learning rate. |
| `NUM_GPUS` | `4` | torchrun nproc_per_node. |

K/T defaults match the configuration reported in the paper for Qwen2.5-VL-7B
(K=18, T=3). For Qwen2.5-VL-3B use K=12,T=3; for Qwen3-VL-4B use K=24,T=3.
