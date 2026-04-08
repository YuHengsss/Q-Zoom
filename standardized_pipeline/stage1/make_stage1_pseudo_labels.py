import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Allow running this script from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qwen_src.ana_utils import get_attn_weights_map, register_hooks
from qwen_src.mm_utils import expand2square
from qwen_src.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

try:
    from qwen_src.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


ANSWER_SUFFIX = "Answer the question using a single word or phrase."
GROUNDING_INSTRUCTION = (
    "Output the grounding bounding box of Region of Interest for the question. "
    "IMPORTANT: The output MUST be raw text, one box per line. DO NOT use JSON. "
    "Follow this exact format: x_min y_min x_max y_max {detail_label}."
)


def _expand(path):
    return os.path.expanduser(os.path.expandvars(path))


def resolve_local_model_path(model_path, search_roots):
    model_path = _expand(model_path)
    if os.path.exists(model_path):
        return model_path

    tail = model_path.rstrip("/\\").split("/")[-1]
    for root in search_roots:
        if not root:
            continue
        candidate = os.path.join(_expand(root), tail)
        if os.path.exists(candidate):
            print(f"[PathResolve] Resolved '{model_path}' -> '{candidate}'")
            return candidate
    return model_path


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def normalize_mode(mode, default="textual"):
    if mode is None:
        return default
    text = str(mode).strip().lower()
    if text in {"natural", "nat"}:
        return "natural"
    if text in {"textual", "text"}:
        return "textual"
    return default


def infer_model_family(model_path):
    lowered = model_path.lower().replace("-", "_")
    if "qwen3" in lowered and "8b" in lowered:
        return "qwen3_8b"
    if "qwen3" in lowered and "4b" in lowered:
        return "qwen3_4b"
    if ("qwen2.5" in model_path.lower() or "qwen2_5" in lowered) and "7b" in lowered:
        return "qwen2_5_7b"
    if ("qwen2.5" in model_path.lower() or "qwen2_5" in lowered) and "3b" in lowered:
        return "qwen2_5_3b"
    raise ValueError(f"Cannot infer model family from model path: {model_path}")


def get_head_config(model_family, mode):
    textual_heads = {
        "qwen2_5_3b": ({"20": [1, 4], "22": [14]}, {}),
        "qwen2_5_7b": ({"22": [1, 4], "23": [6, 11]}, {}),
        "qwen3_4b": ({"19": [24, 27], "24": [21, 23]}, {"24": [6, 10]}),
        "qwen3_8b": ({"19": [24, 26, 27], "24": [21, 22, 23]}, {"24": [1, 2]}),
    }
    natural_heads = {
        "qwen2_5_3b": ({"21": [9, 11], "22": [0, 7]}, {"8": [1, 13]}),
        "qwen2_5_7b": ({"16": [1, 7, 17], "19": [17, 20]}, {"9": [4, 22]}),
        "qwen3_4b": ({"20": [15, 21, 23], "21": [8, 10, 11]}, {"20": [1, 24]}),
        "qwen3_8b": ({"20": [15, 21, 23], "21": [8, 10, 11]}, {"21": [7]}),
    }

    mode = normalize_mode(mode)
    mapping = natural_heads if mode == "natural" else textual_heads
    if model_family not in mapping:
        raise ValueError(f"No head config for model_family={model_family}, mode={mode}")

    required_heads, sink_heads = mapping[model_family]
    required_heads = {int(k): v for k, v in required_heads.items()}
    sink_heads = {int(k): v for k, v in sink_heads.items()}
    return required_heads, sink_heads


def build_prompt(raw_prompt, mode, short_prompt):
    mode = normalize_mode(mode)
    raw_prompt = str(raw_prompt)

    if mode == "natural":
        if ANSWER_SUFFIX in raw_prompt:
            return raw_prompt.replace(ANSWER_SUFFIX, GROUNDING_INSTRUCTION).strip()
        return f"{raw_prompt.strip()} {GROUNDING_INSTRUCTION}".strip()

    if not short_prompt and ANSWER_SUFFIX in raw_prompt:
        return raw_prompt.replace(ANSWER_SUFFIX, "").strip()
    return raw_prompt


def build_inputs(processor, image, prepared_prompt, is_qwen3_8b, use_system_prompt):
    if is_qwen3_8b:
        messages = []
        if use_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prepared_prompt},
                    {"type": "image", "image": image},
                ],
            }
        )
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    msg = [{"role": "user", "content": [{"type": "text", "text": "<image>\n" + prepared_prompt}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "").replace(
        "<image>", "<image><|vision_start|><|image_pad|><|vision_end|>"
    )
    return processor(text=[text], images=[image], padding=True, return_tensors="pt")


def ensure_visual_mask(inputs, activations, processor, device):
    if activations.get("visual_mask", None) is not None:
        return

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    input_ids_list = inputs["input_ids"].tolist()[0]
    try:
        pos = input_ids_list.index(vision_start_id) + 1
        pos_end = input_ids_list.index(vision_end_id)
    except ValueError:
        return

    visual_mask = torch.zeros((1, len(input_ids_list)), dtype=torch.bool, device=device)
    visual_mask[0, pos:pos_end] = True
    activations["visual_mask"] = visual_mask


def aggregate_attention(attn_weights_map, output_shape, head_map):
    if not head_map:
        return np.zeros(output_shape, dtype=np.float32)

    total_heads = sum(len(v) for v in head_map.values())
    if total_heads <= 0:
        return np.zeros(output_shape, dtype=np.float32)

    output = np.zeros(output_shape, dtype=np.float32)
    for layer, heads in head_map.items():
        layer_key = f"layer{layer}"
        if layer_key not in attn_weights_map:
            continue
        layer_attn = attn_weights_map[layer_key]["output2images"].reshape(-1, output_shape[0], output_shape[1])
        for head in heads:
            output += layer_attn[head]

    return output / float(total_heads)


def load_universal_input(path):
    path = os.path.expanduser(os.path.expandvars(path))
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def worker_process(chunk, gpu_id, args, return_dict, worker_id):
    device = f"cuda:{gpu_id}"
    results = []

    is_qwen3 = "qwen3" in args.model_path.lower()
    is_qwen3_8b = "qwen3" in args.model_path.lower() and "8b" in args.model_path.lower()
    patch_size = 32 if is_qwen3 else 28
    default_pixels = 576 * patch_size * patch_size

    try:
        local_only = args.local_files_only or args.offline
        if is_qwen3:
            if Qwen3VLForConditionalGeneration is None:
                raise RuntimeError("Qwen3VLForConditionalGeneration is unavailable in this environment.")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
                local_files_only=local_only,
            ).eval().to(device)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
                local_files_only=local_only,
            ).eval().to(device)

        processor = AutoProcessor.from_pretrained(
            args.processor_path,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True,
            local_files_only=local_only,
        )
        processor.image_processor.max_pixels = default_pixels
        processor.image_processor.min_pixels = default_pixels

        activations, _ = register_hooks(model, is_qwen3vl=is_qwen3)
        model_family = infer_model_family(args.model_path)

        pbar_pos = (int(gpu_id) * args.workers_per_gpu) + worker_id
        for sample in tqdm(chunk, position=pbar_pos, desc=f"GPU{gpu_id}-W{worker_id}"):
            try:
                image_rel = sample.get("image")
                if image_rel is None:
                    continue
                image_path = image_rel if os.path.isabs(str(image_rel)) else os.path.join(args.image_folder, image_rel)

                image = Image.open(image_path).convert("RGB")
                image = expand2square(image, (127, 127, 127))

                raw_prompt = sample.get("text", sample.get("prompt"))
                if raw_prompt is None:
                    continue

                mode = normalize_mode(sample.get("mode", "textual"))
                short_prompt = parse_bool(sample.get("short_prompt", True), default=True)
                use_system_prompt = parse_bool(sample.get("use_system_prompt", mode == "natural"), default=False)
                max_new_tokens = int(sample.get("max_new_tokens", args.max_new_tokens))

                prepared_prompt = build_prompt(raw_prompt, mode=mode, short_prompt=short_prompt)
                inputs = build_inputs(
                    processor=processor,
                    image=image,
                    prepared_prompt=prepared_prompt,
                    is_qwen3_8b=is_qwen3_8b,
                    use_system_prompt=use_system_prompt,
                )
                inputs = inputs.to(device)

                activations.clear()
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, output_attentions=True)

                input_ids_len = inputs.input_ids.shape[1]
                generated_ids = output_ids[:, input_ids_len:]
                output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0]

                ensure_visual_mask(inputs=inputs, activations=activations, processor=processor, device=device)

                output_shape = inputs["image_grid_thw"].cpu().numpy().squeeze(0)[1:] // 2
                output_shape = np.array(output_shape, dtype=np.int32)

                required_heads, sink_heads = get_head_config(model_family=model_family, mode=mode)
                required_layers = list(required_heads.keys()) + list(sink_heads.keys())
                layers_max = max(required_layers) + 1 if required_layers else 0
                attn_weights_map = get_attn_weights_map(
                    activations, layers=layers_max, required_layers=required_layers
                )

                grounding_attn_o2i = aggregate_attention(attn_weights_map, output_shape, required_heads)
                sink_attn = aggregate_attention(attn_weights_map, output_shape, sink_heads)

                result = {
                    "uid": sample.get("uid"),
                    "question_id": str(sample.get("question_id", "")),
                    "dataset": sample.get("dataset"),
                    "mode": mode,
                    "prompt": str(raw_prompt),
                    "text": output_text,
                    "image": str(image_rel),
                    "sink_attn": sink_attn,
                    "grounding_attn_o2i": grounding_attn_o2i,
                }
                results.append(result)
            except Exception as inner_e:
                print(f"[GPU {gpu_id}-W{worker_id}] sample failed: {inner_e}")
                continue
    except Exception as e:
        print(f"[GPU {gpu_id}-W{worker_id}] critical error: {e}")
        import traceback

        traceback.print_exc()

    return_dict[f"{gpu_id}-{worker_id}"] = results


def main():
    parser = argparse.ArgumentParser(description="Unified Stage1 pseudo-label generation from universal input.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--universal-input-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="/")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--cur-dataset", type=str, default=None, help="Optional exact dataset filter.")
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument(
        "--model-search-roots",
        type=str,
        default="",
        help="Comma-separated local roots to resolve repo-style model names (e.g., Qwen/...).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    search_roots = [x.strip() for x in str(args.model_search_roots).split(",") if x.strip()]
    args.model_path = resolve_local_model_path(args.model_path, search_roots)
    args.processor_path = args.processor_path if args.processor_path else args.model_path
    args.processor_path = resolve_local_model_path(args.processor_path, search_roots)

    if args.offline or args.local_files_only:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(
                f"Offline/local mode requires a local model path, but not found: {args.model_path}. "
                "Pass an absolute local --model-path or add proper --model-search-roots."
            )
        if not os.path.exists(args.processor_path):
            raise FileNotFoundError(
                f"Offline/local mode requires a local processor path, but not found: {args.processor_path}."
            )

    records = load_universal_input(args.universal_input_file)
    if args.cur_dataset:
        records = [x for x in records if str(x.get("dataset")) == args.cur_dataset]

    if len(records) == 0:
        raise ValueError("No input records found after loading/filtering universal input.")

    base_gpu_ids = [int(gid) for gid in args.gpu_ids.split(",")]
    expanded_gpu_ids = []
    for gid in base_gpu_ids:
        expanded_gpu_ids.extend([gid] * args.workers_per_gpu)

    total_workers = len(expanded_gpu_ids)
    chunk_size = math.ceil(len(records) / total_workers)
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    while len(chunks) < total_workers:
        chunks.append([])

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    print(
        f"Loaded {len(records)} samples. Starting {total_workers} workers "
        f"on GPUs {args.gpu_ids} ({args.workers_per_gpu} workers per GPU)."
    )

    for i in range(total_workers):
        chunk = chunks[i]
        gpu_id = expanded_gpu_ids[i]
        worker_id_local = i % args.workers_per_gpu
        p = mp.Process(target=worker_process, args=(chunk, gpu_id, args, return_dict, worker_id_local))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_results = []
    for worker_key in sorted(return_dict.keys()):
        final_results.extend(return_dict[worker_key])
    final_results.sort(key=lambda x: str(x.get("uid", x.get("question_id", ""))))

    answers_path = os.path.expanduser(os.path.expandvars(args.answers_file))
    answers_dir = os.path.dirname(answers_path)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    with open(answers_path, "wb") as f:
        pickle.dump(final_results, f)

    print(f"Saved {len(final_results)} pseudo labels to {answers_path}")


if __name__ == "__main__":
    main()
