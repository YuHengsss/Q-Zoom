# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
project_root2 = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root2))
import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from qwen_src.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_src.qwen2_vl import Qwen2VLForConditionalGeneration
_qwen3_import_error = None
try:
    from qwen_src.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )
except Exception as exc:
    print("Failed to import Qwen3VLForConditionalGeneration:", exc)
    _qwen3_import_error = exc
    Qwen3VLForConditionalGeneration = None
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"): #flash_attention_2
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.save_total_limit = 1
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    original_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    config = type(original_config).from_dict(original_config.to_dict())

    config.enable_twig = model_args.enable_twig
    config.twig_K = model_args.twig_K
    config.twig_T = model_args.twig_T
    #config.high_res_K = model_args.high_res_K if hasattr(model_args, 'high_res_K') else model_args.twig_K
    #config.roi_branch = model_args.roi_branch
    config.roi_source = model_args.roi_source
    config.roi_loss = model_args.roi_loss
    config.roi_super_type = model_args.roi_super_type
    config.roi_multi_head = model_args.roi_multi_head
    config.roi_skip_ffn = model_args.roi_skip_ffn  # skip ffn when enabling twig
    config.roi_keep_ffn_mod_ratio = model_args.roi_keep_ffn_mod_ratio
    config.enable_high_res = model_args.enable_high_res
    config.roi_post_training = model_args.roi_post_training
    config.min_pixels = data_args.min_pixels
    config.max_pixels = data_args.max_pixels
    config.reuse_src_pos = model_args.reuse_src_pos
    config.add_noise_to_roi = model_args.add_noise_to_roi

    # if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(
    #         model_args.model_name_or_path.rstrip("/")).name.lower():
    #     model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #         attn_implementation=attn_implementation,
    #         dtype=(torch.bfloat16 if training_args.bf16 else None),
    #         config=config,  # <--- IMPORTANT: Pass your custom config
    #     )
    #     data_args.image_processor = AutoProcessor.from_pretrained(
    #         model_args.model_name_or_path,
    #     ).image_processor
    #     data_args.model_type = "qwen3vl"
    if "qwen3" in model_args.model_name_or_path.lower():
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError(
                f"Qwen3VLForConditionalGeneration is not available; check qwen_src.qwen3_vl imports. "
                f"Import error: {_qwen3_import_error}"
            )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
            config=config,  # <--- IMPORTANT: Pass your custom config
        )
        data_args.model_type = "qwen3vl"
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
    elif "qwen2.5" in model_args.model_name_or_path.lower():

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            config=config,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            config=config,
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        pass
        # model.visual.print_trainable_parameters()
        # model.model.print_trainable_parameters()

    if model_args.enable_twig:
        if not model_args.roi_post_training and not data_args.is_2_5_stage:
            model.load_twig_weights_from_original_model(model_args)

        for param in model.parameters():
            param.requires_grad = False
        if "qwen3" in model_args.model_name_or_path.lower():
            for param in model.model.language_model.embed_tokens.parameters():
                param.requires_grad = False
        llm = model.model if "qwen3" not in model_args.model_name_or_path.lower() else model.model.language_model
        if not model_args.roi_post_training:
            # freeze all parameters except the twig layers
            unfreeze_layers = model_args.twig_freeze if getattr(model_args, 'twig_freeze', None) is not None else 0
            if model_args.enable_high_res:
                for i, high_res_layer in enumerate(llm.high_res_layers[unfreeze_layers:]):
                    for param in high_res_layer.parameters():
                        param.requires_grad = True
                for param in llm.high_res_head.parameters():
                    param.requires_grad = True
            else:
                for i, twig_layer_module in enumerate(llm.twig_layers[unfreeze_layers:]):
                    for param in twig_layer_module.parameters():
                        param.requires_grad = True
        else:
            ##unfreeze parameters after model.model.twig_K
            for i, layer_module in enumerate(llm.layers): ##
                for param in layer_module.parameters():
                    param.requires_grad = True
            ##unfreeze lm_head
            for param in model.lm_head.parameters():
                param.requires_grad = True


    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,
                                                  roi_super_type = model_args.roi_super_type,
                                                  ab_sink=data_args.ab_sink,
                                                  ab_fg_bbox=data_args.ab_fg_bbox,
                                                  fix_res=data_args.fix_res,
                                                  multi_scale_training=data_args.multi_scale_training,
                                                  roi_binary_coeff=data_args.roi_binary_coeff,
                                                  bg_coff=data_args.bg_coff,
                                                  pseudo_blur_kernel_size=data_args.pseudo_blur_kernel_size,
                                                  enable_high_res=model_args.enable_high_res,
                                                  roi_post_training=model_args.roi_post_training
                                                  )

    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'name: {name}, shape: {param.shape}')

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2") #flash_attention_2
