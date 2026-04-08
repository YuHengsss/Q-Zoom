# Standardized Stage2 Pipeline

This Stage2 pipeline is decoupled from model checkpoints in config files.

## Design

1. Build one merged universal input JSONL from multiple source question files.
2. Pass model paths and output paths through runtime args.
3. Judge directly from one base pkl + one roi pkl (no file-pattern pairing).

## Files

- `build_universal_input.py`: merge multi-source json/jsonl into one Stage2 universal input JSONL.
- `make_stage2_training_data.py`: run Stage2 generation/judge/convert from that universal JSONL.
- `example_manifest.json`: source-only manifest template (no model path).

## Step 1: Build Universal Input

```bash
python standardized_pipeline/stage2/build_universal_input.py \
  --manifest-file standardized_pipeline/stage2/example_manifest.json \
  --output-file ${DATA_ROOT}/stage2_universal_input.jsonl
```

## Step 2: Run Stage2 Pipeline (Args Control Models/Outputs)

```bash
python standardized_pipeline/stage2/make_stage2_training_data.py \
  --code-root ${CODE_ROOT} \
  --offline \
  --model-search-roots ${MODEL_ROOTS} \
  --universal-input-file ${DATA_ROOT}/stage2_universal_input.jsonl \
  --image-folder ${DATA_ROOT} \
  --base-model-path ${CODE_ROOT}/output/qwen3vl-4b-roi-K24T3-stage1 \
  --roi-model-path ${CODE_ROOT}/output/qwen3vl-4b-roi-K24T3-stage1 \
  --model-type qwen3 \
  --gpu-ids 0,1 \
  --base-output ${DATA_ROOT}/SFT_SDRPN_PostTraining/stage2_base_qwen3vl-4b-roi-K24T3-stage1.pkl \
  --roi-output ${DATA_ROOT}/SFT_SDRPN_PostTraining/stage2_roi_qwen3vl-4b-roi-K24T3-stage1.pkl \
  --judge-model-path ${MODEL_ROOTS}/Qwen3-VL-8B-Instruct \
  --judge-model-type qwen3 \
  --judge-gpu-ids 0,1 \
  --reg-output ${DATA_ROOT}/SFT_SDRPN_PostTraining/regressions-stage2-qwen3vl-4b-roi-K24T3-stage1.json \
  --imp-output ${DATA_ROOT}/SFT_SDRPN_PostTraining/improvements-stage2-qwen3vl-4b-roi-K24T3-stage1.json \
  --post-sft-output ${DATA_ROOT}/roi-post-sft-qwen3-4b-stage2-qwen3vl-4b-roi-K24T3-stage1.jsonl \
  --data-root ${DATA_ROOT} \
  --mixing-ratio 0.0
```

If outputs are omitted, the script now auto-generates model-dependent names using the model tag (derived from `--roi-model-path`/`--base-model-path`) to avoid overwrite.
Offline mode is recommended by default for cluster runs.

## Skip Examples

- Only generate base/roi responses:

```bash
python standardized_pipeline/stage2/make_stage2_training_data.py \
  --code-root ${CODE_ROOT} \
  --offline \
  --model-search-roots ${MODEL_ROOTS} \
  --universal-input-file ${DATA_ROOT}/stage2_universal_input.jsonl \
  --image-folder ${DATA_ROOT} \
  --base-model-path /path/to/base/model \
  --roi-model-path /path/to/roi/model \
  --model-type auto \
  --base-output /tmp/stage2_base.pkl \
  --roi-output /tmp/stage2_roi.pkl \
  --skip-filter --skip-convert
```
