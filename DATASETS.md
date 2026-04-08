# Q-Zoom — Dataset Setup

All Q-Zoom training and data-generation scripts read from a single root
directory pointed to by the `DATA_ROOT` environment variable. Set it
once and the rest of the scripts pick up everything from there:

```bash
export DATA_ROOT=/path/to/your/datasets
```

This document explains:

1. Which **image archives** are mirrored on Hugging Face today.
2. How `${DATA_ROOT}` should be laid out on disk.
3. Which files each stage of the pipeline actually needs.
4. The status of the question JSONLs and ROI pickles (curated separately).

---

## 1. Image archives on Hugging Face

The image archives used by every Q-Zoom training stage are mirrored at:

> **<https://huggingface.co/datasets/YuhengSSS/RoITraining/tree/main>**

Search the repo for **`.tar`** to see every archive that is currently
hosted. At the time of writing the available archives are:

| Archive | Extracts to | Source dataset |
|---|---|---|
| `train2017.zip` | `coco/train2017/` | COCO 2017 train images |
| `gqa.tar` | `gqa/` | GQA images |
| `ocr_vqa.tar` | `ocr_vqa/` | OCR-VQA images |
| `spdocvqa_images.tar.gz` | `DocVQA/` | DocVQA single-page images |
| `textvqa.tar` | `textvqa/train_images/` | TextVQA train images |
| `ChartQA.tar` | `ChartQA/` | ChartQA images |
| `infographicsvqa.tar` | `infographicsvqa/` | InfographicsVQA images |

Download **only the archives the stages you plan to run actually need**
(see section 3 for the per-stage requirement table). The repo is large,
so do not bulk-download everything unless you really intend to run the
full pipeline.

### Per-archive download example

```bash
pip install -U "huggingface_hub[cli]"

# Download just the archives you need into ${DATA_ROOT}
huggingface-cli download YuhengSSS/RoITraining \
  --repo-type dataset \
  --local-dir "${DATA_ROOT}" --local-dir-use-symlinks False \
  --include "ChartQA.tar" \
  --include "spdocvqa_images.tar.gz" \
  --include "train2017.zip"
```

You can repeat the `--include` flag for additional archives, or call
`huggingface-cli download` once per file.

---

## 2. Question JSONLs and ROI pickles on Hugging Face

The question JSONLs (Stage 1/2/3 input pools) and the ROI training
pickles (`*_576res_185k.pkl`, `*_stage3.pkl`) are hosted at:

> **<https://huggingface.co/datasets/YuhengSSS/Q-Zoom-Training>**

This is a **separate** repo from `YuhengSSS/RoITraining` (which only
holds the image archives in section 1). The split keeps the image
archives, which can be redistributed under their upstream licenses,
isolated from the model-derived pseudo labels and SD-RPN universal
inputs.

### What's in `Q-Zoom-Training`

| Type | Filename |
|---|---|
| Stage-1 question pool (GQA + OCR-VQA) | `llava_v1_5_mix665k_selected_qa.jsonl` |
| VCoT-DocVQA (33K) | `visual_cot_docvqa_subset33k.jsonl` |
| VCoT-InfoVQA (15K) | `visual_cot_infovqa_subset15k.jsonl` |
| VCoT-TextVQA + VCoT-GQA merged (68K) | `visual_cot_llava_subset68k.jsonl` |
| ChartQA<sub>train</sub> (28K) | `chartqa_28k_qa.jsonl` |
| V\*-COCO (44K) | `vstar_coco_spatial_relation_data.jsonl` |
| TextVQA train annotations (34K) | `textvqa/converted_llava_style_train.jsonl` |
| Stage-1 universal input | `stage1_universal_input.jsonl` |
| Stage-2 universal input | `stage2_universal_input.jsonl` |
| Stage-3 universal input | `stage3_universal_input.jsonl` |
| Stage-1 pseudo labels (Qwen2.5-VL-3B) | `qwen2_5vl_pseudo_3b_576res_185k.pkl` |
| Stage-1 pseudo labels (Qwen2.5-VL-7B) | `qwen2_5vl_pseudo_7b_576res_185k.pkl` |
| Stage-1 pseudo labels (Qwen3-VL-4B) | `qwen3vl_pseudo_4b_576res_185k.pkl` |
| Stage-3 ROI training (Qwen2.5-VL-3B) | `qwen2_5vl_3b_stage3.pkl` |
| Stage-3 ROI training (Qwen2.5-VL-7B) | `qwen2_5vl_7b_stage3.pkl` |
| Stage-3 ROI training (Qwen3-VL-4B) | `qwen3vl_4b_stage3.pkl` |

### Download

```bash
# Pull everything into ${DATA_ROOT}
huggingface-cli download YuhengSSS/Q-Zoom-Training \
  --repo-type dataset \
  --local-dir "${DATA_ROOT}" --local-dir-use-symlinks False
```

```bash
# Or download just one backbone's training data
huggingface-cli download YuhengSSS/Q-Zoom-Training \
  --repo-type dataset \
  --local-dir "${DATA_ROOT}" --local-dir-use-symlinks False \
  --include "qwen2_5vl_pseudo_7b_576res_185k.pkl" \
  --include "qwen2_5vl_7b_stage3.pkl" \
  --include "*.jsonl"
```

### Skip data generation entirely

If you only want to **train** Q-Zoom (no need to regenerate Stage-1
pseudo labels or Stage-3 ROI data from scratch), you can point
`ROI_DATA_PATH` straight at the prepared `.pkl`:

```bash
# Skip examples/stage1_train_eval/01_make_pseudo_labels.sh
export ROI_DATA_PATH=${DATA_ROOT}/qwen2_5vl_pseudo_7b_576res_185k.pkl
bash examples/stage1_train_eval/02_train_stage1_model.sh

# Skip examples/stage3_train_eval/01_build_stage3_data.sh
export ROI_DATA_PATH=${DATA_ROOT}/qwen2_5vl_7b_stage3.pkl
bash examples/stage3_train_eval/02_train_stage3_model.sh
```

### Re-generate locally instead

If you'd rather build everything from scratch (e.g. to retarget a
different backbone or produce fresh pseudo labels under a new TWIG
config), use the data pipelines shipped with the release:
`standardized_pipeline/{stage1,stage2,stage3}/` and the matching
`examples/<stage>_train_eval/` walkthroughs. Eval-only users
(`examples/eval_only/`) do not need any of these files.

---

## 3. Extract the image archives

The image archives ship as `.tar` / `.tar.gz` / `.zip`. Extract them
**in place** under `${DATA_ROOT}` so the on-disk paths match the ones
embedded in the question JSONLs:

```bash
cd "${DATA_ROOT}"

# COCO 2017 train images (used by GQA-style records that reference coco/train2017/...)
unzip train2017.zip -d coco/

# OCR / Doc / Chart / TextVQA / GQA / Info benchmarks
mkdir -p DocVQA
tar -xvf spdocvqa_images.tar.gz -C DocVQA/
tar -xvf gqa.tar
tar -xvf ocr_vqa.tar
tar -xvf textvqa.tar
tar -xvf ChartQA.tar
tar -xvf infographicsvqa.tar
```

After extraction, `${DATA_ROOT}` should look approximately like the
layout below.

---

## 4. Expected `${DATA_ROOT}` layout

```
${DATA_ROOT}/
│
├── coco/
│   └── train2017/                                  # COCO 2017 train images
│       └── 000000189951.jpg, ...
├── gqa/                                            # extracted from gqa.tar
├── ocr_vqa/                                        # extracted from ocr_vqa.tar
├── DocVQA/                                         # extracted from spdocvqa_images.tar.gz
├── textvqa/
│   ├── train_images/                               # extracted from textvqa.tar
│   └── converted_llava_style_train.jsonl           # see section 2
├── ChartQA/                                        # extracted from ChartQA.tar
├── infographicsvqa/                                # extracted from infographicsvqa.tar
│
├── llava_v1_5_mix665k_selected_qa.jsonl            # see section 2
├── visual_cot_docvqa_subset33k.jsonl               # see section 2
├── visual_cot_infovqa_subset15k.jsonl              # see section 2
├── visual_cot_llava_subset68k.jsonl                # see section 2
├── chartqa_28k_qa.jsonl                            # see section 2
└── vstar_coco_spatial_relation_data.jsonl          # see section 2
```

> **Tip:** If you store data on a separate filesystem (e.g. a scratch
> mount), keep `${DATA_ROOT}` outside the repository directory. Every
> Q-Zoom example script reads paths through env vars, so nothing in
> `Q-Zoom/` itself needs to be edited.

---

## 5. Training data per Q-Zoom component

This table reproduces the canonical dataset usage from the Q-Zoom
paper. Each Q-Zoom component (SD-RPN / Post-SFT / Dynamic Gate) maps
1:1 to one of the training stages in `examples/`:

- **SD-RPN** ⇄ Stage 1 (`examples/stage1_train_eval/`)
- **Post-SFT** ⇄ Stage 2 (`examples/stage2_train_eval/`)
- **Dynamic Gate** ⇄ Stage 3 (`examples/stage3_train_eval/`)

| Component | Model family | Training source | Samples |
|---|---|---|---:|
| **SD-RPN** | Qwen-series | GQA | 72K |
| | Qwen-series | OCR-VQA | 80K |
| | Qwen-series | VCoT-DocVQA | 33K |
| | Qwen-series | *Total* | *185K* |
| | LLaVA-series | GQA | 72K |
| | LLaVA-series | OCR-VQA | 80K |
| | LLaVA-series | *Total* | *152K* |
| **Post-SFT** | Qwen-series | TextVQA<sub>train</sub> | 34K |
| | Qwen-series | ChartQA<sub>train</sub> | 28K |
| | Qwen-series | VCoT-InfoVQA | 15K |
| | Qwen-series | VCoT-DocVQA | 33K |
| | Qwen-series | V\*-COCO | 44K |
| | Qwen-series | *Mined hard samples (output)* | *~7K* |
| **Dynamic Gate** | All models | VCoT-TextVQA | 18K |
| | All models | VCoT-GQA | 50K |
| | All models | VCoT-DocVQA | 33K |
| | All models | ChartQA<sub>train</sub> | 28K |
| | All models | *Filtered training set (output)* | *40K–60K* |

Italicized rows are **outputs** of the stage (the result of mining hard
samples in Post-SFT, or the result of the gating-data filter in Dynamic
Gate), not additional inputs you need to download.

> **Note for the release templates:** the example scripts default to
> the **Qwen-series** numbers above. Q-Zoom does support LLaVA-series
> backbones (SD-RPN row with 152K samples — no `VCoT-DocVQA`), but the
> `examples/` walkthroughs in this release ship with Qwen2.5-VL /
> Qwen3-VL defaults only.

### Mapping paper-name → release filename

| Paper name | Release filename | Image archive(s) |
|---|---|---|
| GQA (72K) | `llava_v1_5_mix665k_selected_qa.jsonl` (filtered to `gqa` records) | `gqa.tar`, `train2017.zip` |
| OCR-VQA (80K) | `llava_v1_5_mix665k_selected_qa.jsonl` (filtered to `ocr_vqa` records) | `ocr_vqa.tar` |
| VCoT-DocVQA (33K) | `visual_cot_docvqa_subset33k.jsonl` | `spdocvqa_images.tar.gz` |
| TextVQA<sub>train</sub> (34K) | `textvqa/converted_llava_style_train.jsonl` | `textvqa.tar` |
| ChartQA<sub>train</sub> (28K) | `chartqa_28k_qa.jsonl` | `ChartQA.tar` |
| VCoT-InfoVQA (15K) | `visual_cot_infovqa_subset15k.jsonl` | `infographicsvqa.tar` |
| V\*-COCO (44K) | `vstar_coco_spatial_relation_data.jsonl` | `train2017.zip` |
| VCoT-TextVQA (18K) + VCoT-GQA (50K) | `visual_cot_llava_subset68k.jsonl` (the two splits are merged into one 68K JSONL) | `textvqa.tar`, `gqa.tar`, `train2017.zip` |

### Per-stage download checklist

Pick the rows that match the stage you are running — you do **not**
need every archive listed in section 1.

| Stage | Image archives to download | JSONL question files |
|---|---|---|
| **Stage 1 — SD-RPN** (`examples/stage1_train_eval/`) | `gqa.tar`, `ocr_vqa.tar`, `spdocvqa_images.tar.gz`, `train2017.zip` | `llava_v1_5_mix665k_selected_qa.jsonl`, `visual_cot_docvqa_subset33k.jsonl` |
| **Stage 2 — Post-SFT** (`examples/stage2_train_eval/`) | `textvqa.tar`, `ChartQA.tar`, `infographicsvqa.tar`, `spdocvqa_images.tar.gz`, `train2017.zip` | `textvqa/converted_llava_style_train.jsonl`, `chartqa_28k_qa.jsonl`, `visual_cot_infovqa_subset15k.jsonl`, `visual_cot_docvqa_subset33k.jsonl`, `vstar_coco_spatial_relation_data.jsonl` |
| **Stage 2.5** *(Qwen3-VL only)* (`examples/stage2_5_train_eval/`) | Same image archives as Stage 1 | A small fixed-resolution pseudo-label `.pkl` regenerated against the Stage-2 LLM (point `ROI_DATA_PATH` at it) |
| **Stage 3 — Dynamic Gate** (`examples/stage3_train_eval/`) | `textvqa.tar`, `gqa.tar`, `ChartQA.tar`, `spdocvqa_images.tar.gz`, `train2017.zip` | `chartqa_28k_qa.jsonl`, `visual_cot_docvqa_subset33k.jsonl`, `visual_cot_llava_subset68k.jsonl` |
| **Eval-only** (`examples/eval_only/`) | None — `lmms-eval` downloads the eval benchmarks itself | None — only the Q-Zoom checkpoint you want to evaluate |

---

## 6. Disk space (rough)

| Item | Approx. size |
|---|---|
| `coco/train2017/` (118k images) | ~19 GB |
| `DocVQA/` images | ~6 GB |
| `gqa/` images | ~20 GB |
| `ocr_vqa/` images | ~5 GB |
| `textvqa/` images | ~7 GB |
| `ChartQA/` + `infographicsvqa/` | ~3 GB |
| Question JSONLs | <100 MB total |

**Plan for ~60 GB of free space** if you intend to download every image
archive, or ~10 GB if you only want to evaluate a downloaded checkpoint
with `examples/eval_only/`.

---

## 7. Verifying your layout

A quick sanity check before kicking off training:

```bash
# Question files
test -f "${DATA_ROOT}/chartqa_28k_qa.jsonl" && echo "ok chartqa qa"
test -f "${DATA_ROOT}/visual_cot_docvqa_subset33k.jsonl" && echo "ok docvqa qa"
test -f "${DATA_ROOT}/visual_cot_infovqa_subset15k.jsonl" && echo "ok infovqa qa"

# Image dirs
test -d "${DATA_ROOT}/coco/train2017"  && echo "ok coco"
test -d "${DATA_ROOT}/DocVQA"          && echo "ok docvqa images"
test -d "${DATA_ROOT}/textvqa/train_images" && echo "ok textvqa images"
test -d "${DATA_ROOT}/ChartQA"         && echo "ok chartqa images"
test -d "${DATA_ROOT}/infographicsvqa" && echo "ok infovqa images"
```

If any of those `test` lines stay silent, head back to the HF repo and
make sure you downloaded + extracted the matching archive.

---

## 8. License & citation

Please respect the **upstream licenses** of every dataset you download:
COCO, GQA, OCR-VQA, TextVQA, ChartQA, DocVQA, InfographicsVQA, V*Bench,
and the Visual-CoT subsets are released under their own terms. The
Hugging Face repo above only redistributes the image archives where the
upstream license permits redistribution. If a particular archive is
missing or removed from the HF repo, please obtain it from the
corresponding official source and place it under `${DATA_ROOT}` matching
the layout in section 4.
