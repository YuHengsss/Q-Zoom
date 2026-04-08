# Qwen2.5-VL-7B — Multi-Max Tradeoff (Doc/OCR + HR)

Sweep accuracy, throughput, and visual-token cost for the **baseline** and
**Q-Zoom (Stage 3)** Qwen2.5-VL-7B models across a grid of `max_visual_tokens`
settings, then plot the resulting Pareto frontier.

## What this experiment produces

| File | Source |
|---|---|
| `detailed_tradeoff.csv` / `.json` | per-(variant, setting, task) accuracy + throughput |
| `global_tradeoff.csv` / `.json` | (variant, max_tokens) averaged across the doc/OCR and HR groups |
| `doc_ocr_tradeoff.png` | Pareto curve for doc/OCR (uses rectified throughput) |
| `hr_tradeoff.png` | Pareto curve for HR/vision (uses rectified throughput) |

## Pipeline

1. **Performance pass** (full samples, `flash_attention_2`) — `01_run_performance_flash.sh`
2. **Throughput pass** (100 samples per task, `sdpa`) — `02_run_throughput_sdpa.sh`
3. **Summarize** the two manifests into CSVs — `03_summarize_tradeoff.py`
4. **Plot** the doc/OCR and HR Pareto curves — `04_plot_tradeoff.py`

The convenience driver `run_all.sh` runs all four steps end-to-end.

## Token grid

| Variant | `max_tokens` |
|---|---|
| Baseline | 576, 1024, 2048, 4096 |
| Q-Zoom (Stage 3) | 256, 384, 576, 1024, 2048, 4096 |

Special min-token rules (auto-applied):

- `max_tokens == 256` → `min_tokens = 128`
- `max_tokens >= 2048` & HR task & Q-Zoom → `min_tokens = 512` (Tab2 setting)
- Q-Zoom + OCRBench → `min_tokens = 128` (any max)
- Doc/OCR tasks always keep `min_tokens = 256` at high resolution

## Q-Zoom hyperparameters

There are **two** hyperparameter regimes — the script picks the right one
automatically based on `max_tokens`.

### Per-task tuned thresholds (`max_tokens < 2048`)

| Task | `roi_conf_thresh` | `high_res_thresh` |
|---|:-:|:-:|
| textvqa_val | 0.15  | 0.05 |
| infovqa_val | 0.15  | 0.05 |
| chartqa     | 0.125 | 0.05 |
| docvqa_val  | 0.125 | 0.05 |
| ocrbench    | 0.15  | 0.05 |
| vstar_bench | 0.05  | 0.025 |
| mmerealworld_lite | 0.05 | 0.025 |
| hrbench     | 0.03  | 0.025 |

### Tab2 thresholds (`max_tokens >= 2048`, matched to the main paper)

| Task group | `roi_conf_thresh` | `high_res_thresh` |
|---|:-:|:-:|
| Doc/OCR (general) | 0.04 | 0.02 |
| HR (general) | 0.04 | 0.02 |
| HR (vstar_bench) | 0.03 | 0.02 |

## Conda environment

This experiment evaluates Qwen2.5-VL-7B and therefore requires
**`transformers==4.51.3`**. Activate the corresponding env first:

```bash
conda activate qzoom-q25
```

See the top-level `README.md` for the install matrix.

## Run

```bash
# Whole pipeline (perf + tp + summarize + plot)
bash examples/multimax_tradeoff/qwen2_5vl_7b/run_all.sh \
  CODE_ROOT=/path/to/Q-Zoom \
  STAGE3_CKPT=/path/to/qzoom-qwen2_5vl-7b \
  GPU_IDS=0,1,2,3 \
  NUM_PROCESSES=4 \
  SKIP_EXISTING=1
```

```bash
# Performance only (full samples) — useful when iterating on the model
bash examples/multimax_tradeoff/qwen2_5vl_7b/01_run_performance_flash.sh \
  CODE_ROOT=/path/to/Q-Zoom \
  STAGE3_CKPT=/path/to/qzoom-qwen2_5vl-7b \
  GPU_IDS=0,1,2,3 NUM_PROCESSES=4 SKIP_EXISTING=1
```

```bash
# Throughput only (100 samples)
bash examples/multimax_tradeoff/qwen2_5vl_7b/02_run_throughput_sdpa.sh \
  CODE_ROOT=/path/to/Q-Zoom \
  STAGE3_CKPT=/path/to/qzoom-qwen2_5vl-7b \
  GPU_IDS=0 NUM_PROCESSES=1 SKIP_EXISTING=1
```

## Inspect benchmark-wise accuracy

```bash
bash examples/multimax_tradeoff/qwen2_5vl_7b/05_print_performance_matrix.sh \
  RESULT_DIR=examples/multimax_tradeoff/qwen2_5vl_7b/results/<RUN_ID>
```

## Configurable env vars

| Variable | Default | Purpose |
|---|---|---|
| `CODE_ROOT` | `${HOME_PATH}/Q-Zoom` | Repo root, used to set `PYTHONPATH` |
| `BASELINE_CKPT` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model id or local path |
| `STAGE3_CKPT` | `${CODE_ROOT}/output/qzoom-qwen2_5vl-7b` | Q-Zoom checkpoint to evaluate |
| `GPU_IDS` | `0,1,2,3` | Comma list passed to `CUDA_VISIBLE_DEVICES` |
| `NUM_PROCESSES` | `4` | `accelerate launch --num_processes` |
| `SKIP_EXISTING` | `1` | Skip a (variant, setting, task) cell if a previous `_results.json` already exists |
| `BASELINE_MAX_TOKENS` | `576,1024,2048,4096` | Baseline sweep grid |
| `ROI_MAX_TOKENS` | `256,384,576,1024,2048,4096` | Q-Zoom sweep grid |
| `PERFORMANCE_LIMIT` | `0` | If `>0`, cap each task's sample count (for smoke testing) |
| `OUTPUT_ROOT` | `${CODE_ROOT}/logs/qwen2_5vl_7b_multimax_tradeoff/...` | Where raw eval logs land |
