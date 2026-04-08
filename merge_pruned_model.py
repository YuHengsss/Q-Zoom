import torch
import os
import sys
import transformers
from pathlib import Path

# --- Add Project Root to Python Path ---
# This allows us to import from the qwen_src directory
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# --- Qwen-Specific Imports ---
# Import the actual model class you used for training
from qwen_src.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_src.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig


# --- Configuration ---
# You MUST set these variables correctly before running the script.

# This is the number of initial layers from the base model that were KEPT and FROZEN.
# The script will fill in all layers from this index onwards from the original model.
TWIG_K = 21  # !!! EXAMPLE VALUE - SET THIS TO YOUR ACTUAL TWIG_K !!!
twig_T = 3
# Path to your fine-tuned, pruned model directory (the one with the twig layers)
FINETUNED_MODEL_PATH = f"./output/qwen2_5vl-3b-roi-K21T3-152k-v1bf16Mheads-twiginitv5"

# Hugging Face model ID or local path of the original, full model
ORIGINAL_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

# Path where the final, merged model will be saved
DESTINATION_MODEL_PATH = FINETUNED_MODEL_PATH + "-filled"

# --- Advanced Configuration ---
DEVICE = "cpu"
MODEL_DTYPE = torch.bfloat16  # Or torch.float16, torch.float32


def run_fill_pruned_model():
    """
    Loads a "twig" fine-tuned model, fills in the pruned (missing) layers from the
    original base model, and saves the complete model to a new directory.
    """
    print(f"Starting to fill pruned parts of the model at {FINETUNED_MODEL_PATH}")
    print(f"Using TWIG_K = {TWIG_K}. Ensure this is correct.")
    print(f"Using device: {DEVICE}")

    # 1. Load Finetuned (Pruned) Model
    # We load it using the original model's configuration to ensure the full
    # architectural skeleton (including all layer slots) is created.
    print("\n--- Loading Models ---")
    original_config = Qwen2_5_VLConfig.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
    finetuned_config = transformers.AutoConfig.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
    finetuned_config = type(finetuned_config).from_dict(finetuned_config.to_dict())

    print(f"Loading finetuned model from: {FINETUNED_MODEL_PATH}")
    finetuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        FINETUNED_MODEL_PATH,
        config=finetuned_config,  # Use original config to create the full skeleton
        torch_dtype=MODEL_DTYPE,
    )
    finetuned_model.to(DEVICE)
    finetuned_model.eval()
    
    # In Qwen, the main language model part is finetuned_model.model
    finetuned_llama_model = finetuned_model.model

    # 2. Load Original Base Model
    print(f"Loading original model from: {ORIGINAL_MODEL_PATH}")
    original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ORIGINAL_MODEL_PATH,
        config=original_config,
        torch_dtype=MODEL_DTYPE,
    )
    original_model.to(DEVICE)
    original_model.eval()
    original_llama_model = original_model.model

    original_total_layers = original_config.num_hidden_layers
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
    finetuned_model.save_pretrained(DESTINATION_MODEL_PATH)
    
    # Also save the tokenizer and processor for a complete, runnable model directory
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        tokenizer.save_pretrained(DESTINATION_MODEL_PATH)
        
        processor = transformers.AutoProcessor.from_pretrained(FINETUNED_MODEL_PATH)
        processor.save_pretrained(DESTINATION_MODEL_PATH)
        print("Tokenizer and processor also saved.")
    except Exception as e:
        print(f"Warning: Could not save tokenizer/processor. You may need to copy them manually. Error: {e}")


    print("\nDone. The model should now have its pruned parts filled from the original.")


if __name__ == '__main__':
    # Ensure all paths and TWIG_K are set correctly at the top of this script.
    os.makedirs(DESTINATION_MODEL_PATH, exist_ok=True)
    command = f"cp {FINETUNED_MODEL_PATH}/*.json {DESTINATION_MODEL_PATH}/"
    os.system(command)
    command = f"cp {FINETUNED_MODEL_PATH}/t* {DESTINATION_MODEL_PATH}/"
    os.system(command)
    run_fill_pruned_model()
