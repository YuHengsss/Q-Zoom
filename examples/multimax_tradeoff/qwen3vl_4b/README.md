# Qwen3-VL-4B — Multi-Max Tradeoff (Doc/OCR + HR)

Sweep accuracy, throughput, and visual-token cost for the **baseline** and
**Q-Zoom (Stage 3)** Qwen3-VL-4B models across a grid of `max_visual_tokens`
settings, then plot the resulting Pareto frontier. This is the cross-
architecture twin of `examples/multimax_tradeoff/qwen2_5vl_7b/`.

## What this experiment produces

Same outputs as the Qwen2.5-VL-7B variant — see the sibling README for
the full file list.

## Pipeline

1. **Performance pass** (full samples, `flash_attention_2`) — `01_run_performance_flash.sh`
2. **Throughput pass** (100 samples per task, `sdpa`) — `02_run_throughput_sdpa.sh`
3. **Summarize** — `03_summarize_tradeoff.py`
4. **Plot** — `04_plot_tradeoff.py`

`run_all.sh` runs all four steps end-to-end.

## Token grid

| Variant | `max_tokens` |
|---|---|
| Baseline | 576, 1024, 2048, 4096 |
| Q-Zoom (Stage 3) | 256, 384, 576, 1024, 2048, 4096 |

> Qwen3-VL uses **`patch_size = 32`** (vs Qwen2.5-VL's 28), so the actual
> pixel budget is `max_tokens × 32 × 32`.

Min-token rules (auto-applied):

- `max_tokens == 256` → `min_tokens = 128`
- `max_tokens >= 2048` & HR task & Q-Zoom → `min_tokens = 512` (Tab2 setting)
- Q-Zoom + OCRBench → `min_tokens = 128` (any max)

## Q-Zoom hyperparameters

### Per-task tuned thresholds (`max_tokens < 2048`)

| Task | `roi_conf_thresh` | `high_res_thresh` |
|---|:-:|:-:|
| textvqa_val, infovqa_val, docvqa_val | 0.15  | 0.10 |
| chartqa     | 0.15  | 0.10 |
| ocrbench    | 0.125 | 0.10 |
| vstar_bench | 0.10  | 0.05 |
| mmerealworld_lite | 0.05 | 0.05 |
| hrbench     | 0.10  | 0.05 |

### Tab2 thresholds (`max_tokens >= 2048`, matched to the main paper)

| Task group | `roi_conf_thresh` | `high_res_thresh` |
|---|:-:|:-:|
| All tasks (Doc/OCR + HR) | 0.10 | 0.025 |

## Conda environment

This experiment evaluates Qwen3-VL-4B and therefore requires
**`transformers==4.57.0`**. Activate the corresponding env first:

```bash
conda activate qzoom-q3
```

See the top-level `README.md` for the install matrix.

## Run

```bash
# Whole pipeline (perf + tp + summarize + plot)
bash examples/multimax_tradeoff/qwen3vl_4b/run_all.sh \
  CODE_ROOT=/path/to/Q-Zoom \
  STAGE3_CKPT=/path/to/qzoom-qwen3vl-4b \
  GPU_IDS=0,1 \
  NUM_PROCESSES=2 \
  SKIP_EXISTING=1
```

```bash
# Performance only (full samples, 2 GPUs)
bash examples/multimax_tradeoff/qwen3vl_4b/01_run_performance_flash.sh \
  CODE_ROOT=/path/to/Q-Zoom \
  STAGE3_CKPT=/path/to/qzoom-qwen3vl-4b \
  GPU_IDS=0,1 NUM_PROCESSES=2 SKIP_EXISTING=1
```

```bash
# Throughput only (100 samples, 1 GPU)
bash examples/multimax_tradeoff/qwen3vl_4b/02_run_throughput_sdpa.sh \
  CODE_ROOT=/path/to/Q-Zoom \
  STAGE3_CKPT=/path/to/qzoom-qwen3vl-4b \
  GPU_IDS=0 NUM_PROCESSES=1 SKIP_EXISTING=1
```

## Inspect benchmark-wise accuracy

```bash
bash examples/multimax_tradeoff/qwen3vl_4b/05_print_performance_matrix.sh \
  RESULT_DIR=examples/multimax_tradeoff/qwen3vl_4b/results/<RUN_ID>
```

## Configurable env vars

| Variable | Default | Purpose |
|---|---|---|
| `CODE_ROOT` | `${HOME_PATH}/Q-Zoom` | Repo root, used to set `PYTHONPATH` |
| `BASELINE_CKPT` | `Qwen/Qwen3-VL-4B-Instruct` | HF model id or local path |
| `STAGE3_CKPT` | `${CODE_ROOT}/output/qzoom-qwen3vl-4b` | Q-Zoom checkpoint |
| `MODEL_TYPE` | `qwen3_vl` | lmms-eval model class name |
| `GPU_IDS` | `0,1` | Comma list passed to `CUDA_VISIBLE_DEVICES` |
| `NUM_PROCESSES` | `2` | `accelerate launch --num_processes` |
| `SKIP_EXISTING` | `1` | Skip a (variant, setting, task) cell if a previous result exists |
| `BASELINE_MAX_TOKENS` | `576,1024,2048,4096` | Baseline sweep grid |
| `ROI_MAX_TOKENS` | `256,384,576,1024,2048,4096` | Q-Zoom sweep grid |
| `PERFORMANCE_LIMIT` | `0` | If `>0`, cap each task's sample count (smoke testing) |
| `OUTPUT_ROOT` | `${CODE_ROOT}/logs/qwen3vl_4b_multimax_tradeoff/...` | Where raw eval logs land |
