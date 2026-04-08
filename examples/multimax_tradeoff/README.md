# Multi-Max Tradeoff Experiments

These experiments produce the **accuracy vs. throughput** Pareto curves
that appear in the Q-Zoom paper. For each backbone, the pipeline sweeps a
grid of `max_visual_tokens` settings, evaluates both the **baseline** and
the **Q-Zoom Stage 3** checkpoint on Doc/OCR + HR benchmarks, and emits
two PNGs (`doc_ocr_tradeoff.png`, `hr_tradeoff.png`) plus the underlying
CSVs.

| Backbone | Folder | Conda env | `transformers` |
|---|---|---|---|
| Qwen2.5-VL-7B | `qwen2_5vl_7b/` | `qzoom-q25` | `4.51.3` |
| Qwen3-VL-4B   | `qwen3vl_4b/`   | `qzoom-q3`  | `4.57.0` |

> Activate the matching conda env **before** running either experiment —
> the two transformers versions are not interchangeable. See the
> top-level `README.md` for the install matrix.

## Common pipeline (per backbone)

```
01_run_performance_flash.sh    # full-sample accuracy pass (flash_attn_2)
02_run_throughput_sdpa.sh      # 100-sample throughput pass (sdpa)
03_summarize_tradeoff.py       # merge the two manifests into CSVs
04_plot_tradeoff.py            # render Pareto plots
05_print_performance_matrix.sh # print benchmark-wise accuracy table
run_all.sh                     # convenience driver for steps 1–4
```

Each step can be run independently — see the per-backbone README for the
exact CLI and the supported environment variables.

## Quick start

```bash
# Qwen2.5-VL-7B
conda activate qzoom-q25
bash examples/multimax_tradeoff/qwen2_5vl_7b/run_all.sh \
  CODE_ROOT=/path/to/Q-Zoom \
  STAGE3_CKPT=/path/to/qzoom-qwen2_5vl-7b \
  GPU_IDS=0,1,2,3 NUM_PROCESSES=4

# Qwen3-VL-4B
conda activate qzoom-q3
bash examples/multimax_tradeoff/qwen3vl_4b/run_all.sh \
  CODE_ROOT=/path/to/Q-Zoom \
  STAGE3_CKPT=/path/to/qzoom-qwen3vl-4b \
  GPU_IDS=0,1 NUM_PROCESSES=2
```

## Hyperparameter regimes

For both backbones, the scripts apply **two** sets of Q-Zoom thresholds
depending on the `max_tokens` setting:

- **Per-task tuned thresholds** when `max_tokens < 2048` — the
  per-benchmark grid that gives the best accuracy at low resolution.
- **Tab2 thresholds** when `max_tokens >= 2048` — the single setting that
  matches the main-paper Table 2 numbers.

The exact threshold tables for each backbone live in the per-backbone
READMEs.
