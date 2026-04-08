# Q-Zoom: Query-Aware Adaptive Perception for MLLMs

This document describes the three-stage training pipeline, datasets, prompts, and hyperparameters for the Q-Zoom framework. The naming conventions align with the TPAMI submission.

## Pipeline Overview

| Stage | Paper Name | Code Name | Purpose |
|:---:|---|---|---|
| 1 | Self-Distilled Region Proposal Network (SD-RPN) | `stage1` | Train TWIG layers to predict query-relevant RoI heatmaps via self-distilled pseudo-labels |
| 2 | Targeted Supervised Fine-Tuning (Targeted-SFT) | `stage2` | Fine-tune LLM backbone on hard samples where RoI insertion caused regressions |
| 3 | Dynamic Gating Network | `stage3` | Train binary gate to decide per-sample whether high-res RoI refinement is needed |

### Architecture: TWIG (Two-headed Inference Gateway)

Both the SD-RPN (Stage 1) and Dynamic Gate (Stage 3) are instantiated as lightweight branches ("twigs") sharing the same architecture: `T` transformer blocks initialized from layers `K+1` to `K+T` of the frozen LLM backbone. They branch from layer `K` and operate on the intermediate hidden states `H^K_context`.

- **SD-RPN branch**: Predicts a dense spatial heatmap `M_RoI` via Q-K inner product at the final layer.
- **Gating branch**: Predicts a binary refinement probability `Y^pred` via a linear head on the last user-query token.

---

## Stage 1: SD-RPN Training

### Concept

The SD-RPN learns to localize query-relevant image regions without any human-annotated bounding boxes. Instead, it uses **self-distilled pseudo-labels** derived from the MLLM's own cross-modal attention maps. The training pipeline:

1. Run the base MLLM on multi-turn QA data and extract cross-attention maps from a designated middle layer.
2. Denoise the attention maps by filtering sink tokens (high L2-norm visual tokens that attract disproportionate attention).
3. Apply tri-state label assignment: high-confidence foreground (FG), background (BG), and ignored (ambiguous) regions.
4. Train the TWIG branch with selective BCE loss on valid FG/BG tokens only.

### Prompt Design

Two prompt modes are used during pseudo-label generation:

**Textual mode** (OCR/Document datasets): The original question is used directly. If `short_prompt=True`, trailing instructions like "Answer the question using a single word or phrase." are stripped.

**Natural mode** (Visual QA datasets): The question is augmented with a grounding instruction to elicit spatially-aware attention:

```
{original_question} Output the grounding bounding box of Region of Interest
for the question. IMPORTANT: The output MUST be raw text, one box per line.
DO NOT use JSON. Follow this exact format: x_min y_min x_max y_max {detail_label}.
```

A system prompt ("You are a helpful assistant.") is optionally prepended for natural-mode samples.

### Datasets

| Source | Dataset | Samples | Mode |
|--------|---------|:-------:|------|
| GQA | GQA (from LLaVA-v1.5 mix) | 72K | natural |
| OCR-VQA | OCR-VQA (from LLaVA-v1.5 mix) | 80K | textual |
| DocVQA | Visual CoT DocVQA subset | 33K | textual |
| **Total** | | **185K** | |

> **Note:** LLaVA-series models use only GQA + OCR-VQA (152K) due to instability with DocVQA at extreme resolutions.

### Training Hyperparameters

| Config | Setting |
|--------|---------|
| Optimizer | AdamW |
| Weight decay | 0.0 |
| Optimizer momentum | beta_1=0.9, beta_2=0.98 |
| Per-device batch size | 16 |
| Gradient accumulation steps | 2 |
| Number of GPUs | 4 |
| Effective batch size | 128 (16 x 2 x 4) |
| Learning rate schedule | Cosine decay |
| Peak learning rate | 1e-4 |
| Warm-up strategy | Linear |
| Warm-up ratio | 0.03 |
| Max gradient norm | 1.0 |
| Training epochs | 1 |
| Precision | BF16 |
| DeepSpeed | ZeRO Stage 2 |
| Resolution | Fixed (min_pixels = max_pixels = 451584) |
| Trainable parameters | TWIG layers only (K+1 to K+T) + prediction head |
| Frozen parameters | Vision encoder, projector, all backbone LLM layers |
| Loss function | Selective BCE (ignoring ambiguous tokens) |
| Supervision source | Self-distilled cross-attention pseudo-labels |
| Additional loss coefficients | bg_coeff=0.05, roi_binary_coeff=0.25 |

### TWIG Configuration

| Model | K (branch point) | T (TWIG depth) |
|-------|:-:|:-:|
| Qwen2.5-VL-3B | 24 | 3 |
| Qwen2.5-VL-7B | 18 | 3 |
| Qwen3-VL-4B | 24 | 3 |

---

## Stage 2: Targeted Supervised Fine-Tuning (Targeted-SFT)

### Concept

After Stage 1, the SD-RPN can localize RoIs, but the LLM backbone has never seen dual-stream inputs (coarse global + dense local RoI tokens). The RoI tokens use continuous spatio-temporal positional encoding (MRoPE) to maintain spatial alignment, but the backbone needs fine-tuning to fuse these streams effectively.

Rather than fine-tuning on generic data (risking catastrophic forgetting), Stage 2 performs **contrastive hard-sample mining**:

1. Run the **Base Model** (source image only) and the **un-finetuned RoI Model** (source + RoI) on a large QA pool.
2. Use an **LLM-as-a-Judge** to evaluate both responses against ground truth.
3. Isolate **regression samples**: cases where the Base Model is correct but the RoI Model fails (indicating spatial misalignment or contextual distraction).
4. Fine-tune the LLM backbone on these ~7K hard samples to teach robust dual-stream fusion.

> **Note:** This stage only applies to Qwen-series models (which use MRoPE). LLaVA models skip this stage.

### LLM-as-a-Judge

The judge evaluates whether each model's prediction (Base Model = Model A, RoI Model = Model B) is correct against the ground truth. The judge uses a text-only prompt (no image) with the following structure:

**System prompt:**
```
You are a fair and robust evaluator. You focus on semantic correctness over strict string matching.
```

**User prompt:**
```
You are an intelligent evaluator for a Visual Question Answering task.
Compare the Model Predictions against the Ground Truth.

--- DATA ---
Question: {question}
Ground Truth: {ground_truth}
Model A Prediction: {base_response}
Model B Prediction: {roi_response}

--- CRITERIA ---
1. **Containment**: If the Ground Truth is a short value (e.g., '6000') and the
   Model Prediction contains it correctly within a sentence (e.g., 'The value is
   6000'), mark it as **Yes**.
2. **Synonyms**: Accept standard synonyms (e.g., 'Yes' = 'True', '10%' = '0.1').
3. **Formatting**: Ignore punctuation or capitalization differences (e.g., 'april' = 'April').
4. **Contradiction**: Only mark 'No' if the prediction is factually wrong or
   missing the key information.

--- TASK ---
Does Model A provide the correct answer? (Yes/No)
Does Model B provide the correct answer? (Yes/No)

Respond in this exact format:
A: [Yes/No]
B: [Yes/No]
```

Regression samples are those where `A: Yes` and `B: No` (Base correct, RoI fails).

### Judge Model Selection

| Target model family | Judge model |
|---|---|
| Qwen2.5-VL-3B | Qwen2.5-VL-7B-Instruct |
| Qwen2.5-VL-7B | Qwen2.5-VL-7B-Instruct |
| Qwen3-VL-4B | Qwen3-VL-8B-Instruct |

The judge is always a same-family model of equal or larger size to ensure evaluation quality.

### Datasets

**Evaluation pool for mining (~154K QA pairs):**

| Source | Samples |
|--------|:-------:|
| TextVQA training split | 34K |
| ChartQA training split | 28K |
| Visual CoT (InfoVQA subset) | 15K |
| Visual CoT (DocVQA subset) | 33K |
| V*Bench COCO spatial relations | 44K |
| **Total evaluation pool** | **~154K** |

**Mined training set:** ~7K hard samples (Base correct, RoI fails)

### Training Hyperparameters

| Config | Setting |
|--------|---------|
| Optimizer | AdamW |
| Weight decay | 0.0 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 16 |
| Number of GPUs | 4 |
| Effective batch size | 64 (1 x 16 x 4) |
| Learning rate schedule | Cosine decay |
| Peak learning rate | 1e-6 |
| Warm-up strategy | Linear |
| Warm-up ratio | 0.03 |
| Max gradient norm | 1.0 |
| Training epochs | 1 |
| Precision | BF16 |
| DeepSpeed | ZeRO Stage 2 |
| Resolution | Fixed (min_pixels = max_pixels = 451584) |
| Trainable parameters | Full LLM backbone + TWIG layers |
| Frozen parameters | Vision encoder, projector |
| Loss function | Standard language modeling (cross-entropy on response tokens) |
| RoI post-training mode | Enabled (reuse_src_pos=True) |

---

## Stage 3: Dynamic Gating Network

### Concept

The Dynamic Gate learns a binary routing decision: given a (query, image) pair, should the system trigger the expensive high-res RoI branch, or can the coarse-resolution input suffice?

Training uses **consistency-aware sample generation**:

1. For each query, run the base model at multiple resolutions `R = {r_1, r_2, ..., r_k}` in ascending order.
2. Accept only **monotonic transition** cases: model fails at lower resolutions but succeeds at higher ones (approximating a Heaviside step function).
3. Reject **unstable** cases: correct at low-res but wrong at high-res (attributed to ambiguity or hallucination).
4. For accepted samples, randomly select a resolution `r` and assign binary label: `Y=1` (Need-Refine) if wrong at `r`, `Y=0` (No-Refine) if correct.

### Datasets

**Source pool for consistency filtering (~129K):**

| Source | Samples |
|--------|:-------:|
| Visual CoT (TextVQA subset) | 18K |
| Visual CoT (GQA subset) | 50K |
| Visual CoT (DocVQA subset) | 33K |
| ChartQA training split | 28K |
| **Total source pool** | **~129K** |

After consistency-aware filtering, the final gated training set is substantially smaller (~40-60K valid transition samples, model-dependent).

### Training Hyperparameters

| Config | Setting |
|--------|---------|
| Optimizer | AdamW |
| Weight decay | 0.0 |
| Per-device batch size | 32 |
| Gradient accumulation steps | 2 |
| Number of GPUs | 2 |
| Effective batch size | 128 (32 x 2 x 2) |
| Learning rate schedule | Cosine decay |
| Peak learning rate | 1e-4 |
| Warm-up strategy | Linear |
| Warm-up ratio | 0.03 |
| Max gradient norm | 1.0 |
| Training epochs | 1 |
| Precision | BF16 |
| DeepSpeed | ZeRO Stage 2 |
| Resolution | Dynamic (min_pixels=50176, max_pixels=802816) |
| Trainable parameters | TWIG layers only (K+1 to K+T) + gate linear head |
| Frozen parameters | Vision encoder, projector, all backbone LLM layers, SD-RPN branch |
| Loss function | Binary Cross-Entropy (BCE) |
| High-res prediction | Enabled (enable_high_res=True) |
| Backbone | Stage 2.5 checkpoint (stage2 + merged SD-RPN weights) |

---

## Inference Configuration

At inference, two thresholds control the adaptive pipeline:

| Threshold | Symbol | Purpose |
|-----------|:------:|---------|
| Gate confidence | tau_gate | Controls routing: Y^pred >= tau_gate triggers RoI branch |
| RoI confidence | tau_roi | Binarizes the predicted heatmap to extract foreground bbox |

Higher `tau_gate` = more samples skip RoI (faster, lower accuracy on hard samples).
Higher `tau_roi` = tighter RoI crops (more focused but may miss peripheral evidence).

### Default Visual Token Limits

| Setting | Max tokens | Use case |
|---------|:-:|---|
| Standard benchmarks | 576 | Fair comparison with fixed-res baselines |
| High-res benchmarks | 4096 | Fair comparison with SoTA high-res methods |
| Minimum tokens | 128 | Prevents distortion from strict min=max equality |

---

## Code References

| Component | Path |
|-----------|------|
| Stage 1 pseudo-label generation | `standardized_pipeline/stage1/` |
| Stage 2 targeted SFT data mining | `standardized_pipeline/stage2/` |
| Stage 3 gated data generation | `standardized_pipeline/stage3/` |
| Training framework | `qwen-vl-finetune/` |
| Training scripts (CityU) | `qwen-vl-finetune/scripts/A6000/` |
| TWIG model code (Qwen2.5-VL) | `qwen_src/modeling_qwen2_5_vl.py` |
| TWIG model code (Qwen3-VL) | `qwen_src/qwen3_vl/modeling_qwen3_vl.py` |
| RoI utilities | `qwen_src/mm_utils.py` |
| Evaluation framework | `lmms-eval/` |
