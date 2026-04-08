import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

# --- Qwen Imports ---
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)

# Safe import for Qwen3
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel,
        Qwen3VLModel,
        apply_rotary_pos_emb,
    )

    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False
    Qwen3VisionTransformerPretrainedModel = None
    Qwen3VLModel = None

try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    # Fallback for older transformers versions
    FlashAttentionKwargs = Dict

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

# from transformers.trainer import (
#     ALL_LAYERNORM_LAYERS,
#     get_parameter_names,
# )


# =============================================================================
#  SECTION 1: ATTENTION PATCHES (Source Code Logic)
# =============================================================================

def flash_attention_forward(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=None,
        sliding_window=None,
        position_ids=None,
        softcap=None,
        **kwargs,
):
    """
    Shared helper to call flash_attn_varlen_func for flattened data.
    """
    # Flatten checks
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)

    # cu_seqlens is passed via attention_mask in this packed implementation
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    # Causal logic
    is_causal = True  # generally true for decoder-only

    flash_kwargs = {}
    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=is_causal,
        **flash_kwargs,
    )

    return attn_output.unsqueeze(0), None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Standard Qwen2/2.5 Logic
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    # Note: apply_multimodal_rotary_pos_emb must be available in the context or imported
    # Typically this function is available in the original module scope
    # If using standard transformers, we might need to rely on the module's method if available,
    # or import the helper. For Qwen2VL, it's often a standalone function.

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Qwen3 Logic (includes Norms)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Apply Norms (Qwen3 specific)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def return_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, **kwargs):
    # Bypass mask creation because we use cu_seqlens in attention_mask for packed data
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers

    # Patch Qwen2 VL
    if hasattr(transformers.models, "qwen2_vl"):
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = qwen2vl_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.create_causal_mask = return_mask
        transformers.models.qwen2_vl.modeling_qwen2_vl.create_sliding_window_causal_mask = return_mask

    # Patch Qwen2.5 VL
    if hasattr(transformers.models, "qwen2_5_vl"):
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2vl_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask = return_mask
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_sliding_window_causal_mask = return_mask

    # Patch Qwen3 VL
    if hasattr(transformers.models, "qwen3_vl"):
        transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = qwen3vl_forward
        transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = return_mask

    # Patch Qwen3 MoE
    if hasattr(transformers.models, "qwen3_vl_moe"):
        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = qwen3vl_forward
        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = return_mask

    print("Successfully patched Attention classes for Data Packing.")


# =============================================================================
#  SECTION 2: CUSTOM OPTIMIZER & UTILS (Your logic)
# =============================================================================

def print_trainable_parameters_visual(self) -> None:
    trainable_blocks = []
    non_trainable_blocks = []
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check merger (could be named differently in Q3, but 'merger' is standard)
    if hasattr(self, "merger"):
        is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())
    else:
        is_merger_trainable = "N/A"

    print("Vision Module - Attention Blocks:")
    print(f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}")
    print(f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}")
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    is_embed_trainable = any(param.requires_grad for param in self.embed_tokens.parameters())
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    trainable_layers = []
    non_trainable_layers = []
    for layer_idx, layer in enumerate(self.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    print(f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}")
    print(f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}")


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer

# --- APPLY PATCHES ---
Trainer.create_optimizer = create_optimizer

# Qwen2 VL
Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters

# Qwen2.5 VL
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters

# Qwen3 VL
if QWEN3_AVAILABLE:
    Qwen3VLVisionModel.print_trainable_parameters = print_trainable_parameters_visual
    Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
    # Qwen3VLMoeVisionModel.print_trainable_parameters = print_trainable_parameters_visual
    # Qwen3VLMoeModel.print_trainable_parameters = print_trainable_parameters