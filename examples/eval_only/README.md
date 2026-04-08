# Eval-only

Evaluate any Q-Zoom checkpoint (Stage 1, 2, or 3) with `lmms-eval`.

> **Activate the matching conda env first** — Q-Zoom needs different
> `transformers` versions per backbone:
>
> | Checkpoint backbone | Conda env | `transformers` |
> |---|---|---|
> | Qwen2.5-VL-3B / 7B | `qzoom-q25` | `4.51.3` |
> | Qwen3-VL-4B | `qzoom-q3` | `4.57.1` |

## Released Q-Zoom checkpoints

The final Stage-3 checkpoints for all three supported backbones are
hosted on Hugging Face. Pull whichever one matches your env:

| Backbone | HF model id |
|---|---|
| Qwen2.5-VL-3B | [`YuhengSSS/Q-Zoom-Qwen2.5VL-3B`](https://huggingface.co/YuhengSSS/Q-Zoom-Qwen2.5VL-3B) |
| Qwen2.5-VL-7B | [`YuhengSSS/Q-Zoom-Qwen2.5VL-7B`](https://huggingface.co/YuhengSSS/Q-Zoom-Qwen2.5VL-7B) |
| Qwen3-VL-4B   | [`YuhengSSS/Q-Zoom-Qwen3VL-4B`](https://huggingface.co/YuhengSSS/Q-Zoom-Qwen3VL-4B) |

```bash
huggingface-cli download YuhengSSS/Q-Zoom-Qwen2.5VL-7B \
  --local-dir ./checkpoints/Q-Zoom-Qwen2.5VL-7B \
  --local-dir-use-symlinks False
```

```bash
# Qwen2.5-VL-7B (substitute the 3B repo for the 3B backbone)
conda activate qzoom-q25
CHECKPOINT_PATH=./checkpoints/Q-Zoom-Qwen2.5VL-7B NUM_GPUS=4 \
bash eval_qwen2_5vl_stage3.sh

# Qwen3-VL-4B
conda activate qzoom-q3
CHECKPOINT_PATH=./checkpoints/Q-Zoom-Qwen3VL-4B NUM_GPUS=4 \
bash eval_qwen3vl_stage3.sh
```

The benchmarks reported in the paper:

| Family | Tasks | Resolution range |
|---|---|---|
| Doc/OCR | textvqa_val, infovqa_val, chartqa, ocrbench, docvqa_val | 256–576 visual tokens |
| HR/Vision | vstar_bench, mmerealworld_lite, hrbench | 512–4096 visual tokens |

You can also evaluate a base (non-Q-Zoom) model by setting
`TWO_STAGE_ROI=False` to skip the gating and run vanilla decoding.

## Per-task hyper-parameter tables

The eval scripts use a different threshold pair for each task. The
numbers come from the paper's multi-max trade-off table — Doc/OCR uses
the **default** regime (max_tokens=576), and HR uses the **Tab2** regime
(max_tokens=4096, min_tokens=512).

### Qwen2.5-VL-7B (`eval_qwen2_5vl_stage3.sh`)

| Task | `roi_conf_thresh` | `high_res_thresh` | `min_tokens` | `max_tokens` |
|------|:-:|:-:|:-:|:-:|
| textvqa_val       | 0.15  | 0.05 | 256 | 576 |
| infovqa_val       | 0.15  | 0.05 | 256 | 576 |
| chartqa           | 0.125 | 0.05 | 256 | 576 |
| ocrbench          | 0.15  | 0.05 | **128** | 576 |
| docvqa_val        | 0.125 | 0.05 | 256 | 576 |
| vstar_bench       | 0.03  | 0.02 | 512 | 4096 |
| mmerealworld_lite | 0.04  | 0.02 | 512 | 4096 |
| hrbench           | 0.04  | 0.02 | 512 | 4096 |

### Qwen3-VL-4B (`eval_qwen3vl_stage3.sh`)

| Task | `roi_conf_thresh` | `high_res_thresh` | `min_tokens` | `max_tokens` |
|------|:-:|:-:|:-:|:-:|
| textvqa_val       | 0.15  | 0.10  | 256 | 576 |
| infovqa_val       | 0.15  | 0.10  | 256 | 576 |
| chartqa           | 0.15  | 0.10  | 256 | 576 |
| ocrbench          | 0.125 | 0.10  | **128** | 576 |
| docvqa_val        | 0.15  | 0.10  | 256 | 576 |
| vstar_bench       | 0.10  | 0.025 | 512 | 4096 |
| mmerealworld_lite | 0.10  | 0.025 | 512 | 4096 |
| hrbench           | 0.10  | 0.025 | 512 | 4096 |

`min_pixels` / `max_pixels` are computed from the token counts above by
multiplying with `patch_size**2` (28² for Qwen2.5-VL, 32² for Qwen3-VL).
The same per-task table is used by `examples/stage3_train_eval/03_eval_stage3_model.sh`,
selectable via the `MODEL_FAMILY={qwen2_5vl,qwen3vl}` env var.
