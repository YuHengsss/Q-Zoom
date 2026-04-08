import os
import sys
from pathlib import Path

import torch
import transformers

# --- Add Project Root to Python Path ---
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# --- Qwen-Specific Imports ---
from qwen_src.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwen_src.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


# --- Configuration ---
# You MUST set these variables correctly before running the script.

# This is the number of initial layers from the base model that were KEPT and FROZEN.
# The script will fill in all layers from this index onwards from the original model.
TWIG_K = 24  # !!! EXAMPLE VALUE - SET THIS TO YOUR ACTUAL TWIG_K !!!
twig_T = 3
# Path to your fine-tuned, pruned model directory (the one with the twig layers)
FINETUNED_MODEL_PATH = f"./output/qwen3vl-4b-roi-K{TWIG_K}T{twig_T}-185k-v1bf16-twiginit"

# Hugging Face model ID or local path of the original, full model
ORIGINAL_MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

# Path where the final, merged model will be saved
DESTINATION_MODEL_PATH = FINETUNED_MODEL_PATH + "-filled"

# --- Advanced Configuration ---
DEVICE = "cpu"
MODEL_DTYPE = torch.bfloat16  # Or torch.float16, torch.float32
SAVE_SAFE_SERIALIZATION = True  # Set True to save as safetensors (breaks shared weights)


def run_fill_pruned_model():
    """
    Loads a "twig" fine-tuned model, fills in the pruned (missing) layers from the
    original base model, and saves the complete model to a new directory.
    """
    print(f"Starting to fill pruned parts of the model at {FINETUNED_MODEL_PATH}")
    print(f"Using TWIG_K = {TWIG_K}. Ensure this is correct.")
    print(f"Using device: {DEVICE}")

    # 1. Load Finetuned (Pruned) Model
    print("\n--- Loading Models ---")
    original_config = Qwen3VLConfig.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
    finetuned_config = transformers.AutoConfig.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
    finetuned_config = type(finetuned_config).from_dict(finetuned_config.to_dict())

    print(f"Loading finetuned model from: {FINETUNED_MODEL_PATH}")
    finetuned_model = Qwen3VLForConditionalGeneration.from_pretrained(
        FINETUNED_MODEL_PATH,
        config=finetuned_config,
        torch_dtype=MODEL_DTYPE,
    )
    finetuned_model.to(DEVICE)
    finetuned_model.eval()

    finetuned_llama_model = finetuned_model.model.language_model

    # 2. Load Original Base Model
    print(f"Loading original model from: {ORIGINAL_MODEL_PATH}")
    original_model = Qwen3VLForConditionalGeneration.from_pretrained(
        ORIGINAL_MODEL_PATH,
        config=original_config,
        torch_dtype=MODEL_DTYPE,
    )
    original_model.to(DEVICE)
    original_model.eval()
    original_llama_model = original_model.model.language_model

    original_total_layers = original_config.text_config.num_hidden_layers
    print(f"\nOriginal model has {original_total_layers} layers.")
    print(f"Finetuned model loaded with {len(finetuned_llama_model.layers)} layer slots.")

    # Safety check
    if len(finetuned_llama_model.layers) != original_total_layers:
        print("\nWarning: The number of layer 'slots' in the finetuned model does not match the original config.")
        print("This script will proceed assuming the config correctly defines the architecture and will attempt to fill the layers.")

    # --- 3. Reconstruct and Repopulate Pruned Parts ---
    print("\n--- Copying Weights for Pruned Layers ---")
    with torch.no_grad():
        # Part A: Fill the pruned transformer layers (from TWIG_K to the end)
        print(f"Attempting to fill transformer layers from index {TWIG_K} to {original_total_layers - 1}...")
        for i in range(TWIG_K, original_total_layers):
            if i < len(original_llama_model.layers) and i < len(finetuned_llama_model.layers):
                print(f"  - Copying weights for layer {i}...")
                try:
                    state_to_load = original_llama_model.layers[i].state_dict()
                    finetuned_llama_model.layers[i].load_state_dict(state_to_load)
                except Exception as e:
                    print(f"    ERROR loading state_dict for layer {i}: {e}")
            else:
                print(f"  - Skipping layer {i}: index out of bounds.")

    # --- 4. Save the "Completed" Model ---
    print(f"\n--- Saving Completed Model ---")
    print(f"Saving the merged model to: {DESTINATION_MODEL_PATH}")
    if SAVE_SAFE_SERIALIZATION:
        state_dict = finetuned_model.state_dict()
        seen_ptrs = {}
        for name, tensor in state_dict.items():
            try:
                ptr = tensor.untyped_storage().data_ptr()
            except Exception:
                ptr = tensor.storage().data_ptr()
            if ptr in seen_ptrs:
                state_dict[name] = tensor.clone()
            else:
                seen_ptrs[ptr] = name
        finetuned_model.save_pretrained(
            DESTINATION_MODEL_PATH, safe_serialization=True, state_dict=state_dict
        )
    else:
        finetuned_model.save_pretrained(DESTINATION_MODEL_PATH, safe_serialization=False)

    # Also save the tokenizer and processor for a complete, runnable model directory
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
        tokenizer.save_pretrained(DESTINATION_MODEL_PATH)

        processor = transformers.AutoProcessor.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
        processor.save_pretrained(DESTINATION_MODEL_PATH)
        print("Tokenizer and processor also saved.")
    except Exception as e:
        print(f"Warning: Could not save tokenizer/processor. You may need to copy them manually. Error: {e}")

    print("\nDone. The model should now have its pruned parts filled from the original.")


if __name__ == "__main__":
    os.makedirs(DESTINATION_MODEL_PATH, exist_ok=True)
    command = f"cp {FINETUNED_MODEL_PATH}/*.json {DESTINATION_MODEL_PATH}/"
    os.system(command)
    command = f"cp {FINETUNED_MODEL_PATH}/t* {DESTINATION_MODEL_PATH}/"
    os.system(command)
    run_fill_pruned_model()

    # Remove stale files that don't match the chosen serialization.
    try:
        for name in os.listdir(DESTINATION_MODEL_PATH):
            if SAVE_SAFE_SERIALIZATION:
                if name.endswith(".bin"):
                    os.remove(os.path.join(DESTINATION_MODEL_PATH, name))
            else:
                if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
                    os.remove(os.path.join(DESTINATION_MODEL_PATH, name))
    except Exception as e:
        print(f"Warning: could not clean serialization artifacts: {e}")
