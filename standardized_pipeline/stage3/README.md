# Standardized Stage3 Pipeline

This Stage3 pipeline is also decoupled from model checkpoints in config files.

## Design

1. Build one merged universal input JSONL from source question files.
2. Pass model/output paths through runtime args.
3. Generate one final Stage3 pkl directly from the universal input.

## Files

- `build_universal_input.py`: merge multi-source json/jsonl into one Stage3 universal input JSONL.
- `make_stage3_training_data.py`: run Stage3 generation from that universal JSONL.
- `example_manifest.json`: source-only manifest template (no model path).

## Step 1: Build Universal Input

```bash
python standardized_pipeline/stage3/build_universal_input.py \
  --manifest-file standardized_pipeline/stage3/example_manifest.json \
  --output-file ${DATA_ROOT}/stage3_universal_input.jsonl
```

## Step 2: Run Stage3 Pipeline (Args Control Models/Outputs)

```bash
python standardized_pipeline/stage3/make_stage3_training_data.py \
  --code-root ${CODE_ROOT} \
  --offline \
  --model-search-roots ${MODEL_ROOTS} \
  --universal-input-file ${DATA_ROOT}/stage3_universal_input.jsonl \
  --image-folder ${DATA_ROOT} \
  --model-path ${MODEL_ROOTS}/Qwen3-VL-4B-Instruct \
  --model-type qwen3 \
  --gpu-ids 0,1 \
  --stage3-output ${DATA_ROOT}/stage3results/stage3_qwen3vl-4b.pkl
```

If `--stage3-output` is omitted, the script auto-generates a model-dependent output path:
`${DATA_ROOT}/stage3results/stage3_<model_tag>.pkl`.
Offline mode is recommended by default for cluster runs.

## Skip Example

- Skip generation (for path/debug checks only):

```bash
python standardized_pipeline/stage3/make_stage3_training_data.py \
  --code-root ${CODE_ROOT} \
  --offline \
  --model-search-roots ${MODEL_ROOTS} \
  --universal-input-file ${DATA_ROOT}/stage3_universal_input.jsonl \
  --skip-generate
```
