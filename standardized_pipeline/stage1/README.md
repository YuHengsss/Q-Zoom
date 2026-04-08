# Standardized Stage1 Pipeline

This folder contains the new unified Stage1 data flow:

1. Build one universal Stage1 input file (JSONL) with dataset-level defaults and optional example-level overrides.
2. Run one unified pseudo-label generator for both natural and textual settings.

## Files

- `build_universal_input.py`: builds the universal Stage1 input JSONL from one manifest.
- `make_stage1_pseudo_labels.py`: unified Stage1 pseudo-label generation (replaces separate natural/textual scripts).
- `example_manifest.json`: reference manifest template.

## Universal Input Record

Each line in the universal input JSONL is one sample:

```json
{
  "uid": "gqa:gqa_source:120396:0",
  "question_id": "120396",
  "dataset": "gqa",
  "image": "coco/train2017/000000189951.jpg",
  "text": "What concerns could the owners...",
  "mode": "natural",
  "short_prompt": true,
  "use_system_prompt": true,
  "max_new_tokens": 128,
  "source": "gqa_source"
}
```

## Build Universal Input

```bash
python standardized_pipeline/stage1/build_universal_input.py \
  --manifest-file standardized_pipeline/stage1/example_manifest.json \
  --output-file ${DATA_ROOT}/stage1_universal_input.jsonl
```

## Default Runtime Mode: Offline

`make_stage1_pseudo_labels.py` now supports offline-first loading.

- Use `--offline` by default on servers without internet.
- Use `--model-search-roots` to map repo-style names like `Qwen/...` to local directories.
- Optionally set `--processor-path` separately; if omitted, it follows `--model-path`.

## Generate Stage1 Pseudo Labels

```bash
python standardized_pipeline/stage1/make_stage1_pseudo_labels.py \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --processor-path Qwen/Qwen3-VL-4B-Instruct \
  --model-search-roots ${MODEL_ROOTS} \
  --offline \
  --universal-input-file ${DATA_ROOT}/stage1_universal_input.jsonl \
  --image-folder ${DATA_ROOT} \
  --answers-file ${DATA_ROOT}/stage1results/qwen3vl_4b_stage1_unified.pkl \
  --gpu-ids 0,1 \
  --workers-per-gpu 2
```
