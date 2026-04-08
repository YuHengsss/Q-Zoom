# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from .configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers import AutoProcessor

from ..mm_utils import (
    calculate_roi_align_loss_hidden_states,
    extract_visual_reps_from_hidden_states,
    get_batched_sub_images_v2,
    get_singleturn_query_text_hs,
    get_singleturn_query_text_hs_mheads,
    insert_sub_feat_v2,
    update_batched_labels,
)


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def slice_kv_cache(cache: Optional[Cache], keep_len: int, valid_mask: Optional[torch.Tensor] = None, config=None):
    """Keep only the first ``keep_len`` tokens in each KV layer cache."""
    if cache is None:
        return None
    keep_len = max(int(keep_len), 0)
    try:
        new_cache = DynamicCache(config=config) if config is not None else DynamicCache()
    except TypeError:
        # Backward-compatible ctor fallback.
        new_cache = DynamicCache()

    for layer_idx in range(len(cache)):
        key_states, value_states = cache[layer_idx]
        if key_states is None or value_states is None:
            continue

        new_key = key_states[:, :, :keep_len, :]
        new_value = value_states[:, :, :keep_len, :]

        if valid_mask is not None:
            mask = valid_mask[0, :keep_len].bool()
            new_key = new_key[:, :, mask, :]
            new_value = new_value[:, :, mask, :]

        try:
            new_cache.update(new_key, new_value, layer_idx)
        except TypeError:
            # Some cache impls require the extra kwargs arg.
            new_cache.update(new_key, new_value, layer_idx, {})
    return new_cache


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        ####only for debug here
        #self.config._attn_implementation = "flash_attention_2"
        ####
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3VLTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
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


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Qwen3VLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


@auto_docstring
class Qwen3VLPreTrainedModel(PreTrainedModel):
    config: Qwen3VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3VLTextDecoderLayer,
        "attentions": Qwen3VLTextAttention,
    }


class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLVisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


@auto_docstring(
    custom_intro=(
        "Text part of Qwen3VL, "
        "not a pure text-only model, as DeepStack integrates visual features into the early hidden states."
    )
)
class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.mm_config = getattr(config, "mm_config", None)
        self.vision_config = getattr(config, "vision_config", None)
        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", None)

        self.enable_twig = False
        self.generation_attention_mask = None
        self.roi_new_hs_mask = None
        self.roi_surrounding_bbox = None
        self.input_ids_updated = None
        if getattr(config, "enable_twig", False):
            self.enable_twig = True
            self.twig_start_layer = config.twig_K
            self.twig_layer_count = config.twig_T
            self.roi_loss_type = getattr(config, "roi_loss", "bce")
            self.roi_conf_thresh = 0.1
            self.roi_loss = self.roi_loss_type
            self.enable_high_res = getattr(config, "enable_high_res", False)
            self.roi_source = getattr(config, "roi_source", "qk")
            self.roi_supervision_type = getattr(config, "roi_super_type", "v1")
            self.roi_multi_head = getattr(config, "roi_multi_head", False)
            self.roi_enable2stage = getattr(config, "roi_enable2stage", False)
            self.roi_post_training = getattr(config, "roi_post_training", False)
            self.reuse_src_pos = getattr(config, "reuse_src_pos", False)
            self.add_noise_to_roi = getattr(config, "add_noise_to_roi", False)
            self.twig_layers = nn.ModuleList(
                [Qwen3VLTextDecoderLayer(config, layer_idx=self.twig_start_layer + i) for i in range(self.twig_layer_count)]
            )

            self.first_branch_layer = self.twig_start_layer
            self.high_res_head = None
            self.high_res_layers = None
            self.high_res_threshold = getattr(config, "high_res_thresh", 0.1)
            self.high_res_start_layer = self.twig_start_layer
            if self.enable_high_res:
                self.high_res_head = nn.Linear(config.hidden_size, 1)
                self.high_res_layers = nn.ModuleList(
                    [
                        Qwen3VLTextDecoderLayer(
                            config, layer_idx=getattr(config, "high_res_K", self.twig_start_layer) + i
                        )
                        for i in range(getattr(config, "high_res_T", self.twig_layer_count))
                    ]
                )
                self.high_res_start_layer = getattr(config, "high_res_K", self.twig_start_layer)
                self.first_branch_layer = self.high_res_start_layer

            self.twig_K = self.twig_start_layer
            self.twig_T = self.twig_layer_count
            self.img_llm_patch_size = 32
            self.is_debug = False
            self.last_pos_id = None

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        # args for RoI
        image_grid_thw: Optional[torch.LongTensor] = None,
        src_images: Optional[list] = None,
        encode_image_fn: Optional[Callable] = None,
        image_processor: Optional[Any] = None,
        labels: Optional[torch.LongTensor] = None,
        image_token_start: int = 0,
        pos_id_fn: Optional[Callable] = None,
        temp_input_ids: Optional[torch.LongTensor] = None,
        processor: Optional[Any] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            batch_size, seq_length = input_ids.shape[:2]
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]

        # Handle attention mask for fused 2-stage generation
        if self.training is False and seq_length == 1 and self.generation_attention_mask is not None:
            if self.generation_attention_mask.shape[1] == past_key_values.get_seq_length():
                attention_mask = torch.cat(
                    [self.generation_attention_mask,
                     self.generation_attention_mask.new_ones((self.generation_attention_mask.shape[0], 1))],
                    dim=-1
                )
                self.generation_attention_mask = attention_mask

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if cache_position is not None and seq_length == 1 and past_seen_tokens and self.enable_twig:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            if self.enable_twig and self.roi_enable2stage and self.last_pos_id is not None:
                position_ids = self.last_pos_id + 1
                self.last_pos_id = position_ids.clone()

            if not (self.enable_twig and self.roi_enable2stage):
                attention_mask = attention_mask.new_ones([attention_mask.shape[0], past_seen_tokens + inputs_embeds.shape[1]])

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        raw_attn_mask = attention_mask
        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        batch_size, seq_len = hidden_states.shape[:2]
        visual_token_counts = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.long)
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            merge_size = self.vision_config.spatial_merge_size if self.vision_config is not None else 2
            visual_token_counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // (
                merge_size**2
            )

        if seq_len > 1:
            self.bbox_img = None
            self.visual_token_counts = visual_token_counts
            self.visual_token_num = visual_token_counts[0].item() if batch_size > 0 else 0
            self.bbox_image_grid_thw = None

        if self.enable_twig and image_token_start > 0 and seq_len > 1:
            self.generation_attention_mask = None
            self.roi_new_hs_mask = None
            self.last_pos_id = None

            if temp_input_ids is None:
                temp_input_ids = input_ids
            if temp_input_ids is not None and self.image_token_id is not None and self.video_token_id is not None:
                visual_token_mask = (temp_input_ids == self.image_token_id) | (temp_input_ids == self.video_token_id)
            else:
                visual_token_mask = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool)

            hidden_states_initial = hidden_states.clone()
            for layer_idx in range(self.first_branch_layer):
                hidden_states = self.layers[layer_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                # add visual features to the hidden states of first several layers
                if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

            needs_high_res = True
            high_res_hidden_states = None
            self.roi_post_training = self.training and self.roi_post_training ###remove this if want to test the performance without gated high-res
            self.enable_high_res = self.enable_high_res and not self.roi_post_training  ###FIXME
            self.roi_enable2stage = True if self.roi_post_training and self.training else self.roi_enable2stage

            if self.enable_high_res and self.high_res_layers is not None:
                high_res_hidden_states = hidden_states
                for layer_offset, high_res_layer in enumerate(self.high_res_layers):
                    high_res_hidden_states = high_res_layer(
                        high_res_hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=None,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                    high_res_layer_idx = self.high_res_start_layer + layer_offset

                if not self.training and self.high_res_head is not None:
                    high_res_scores = self.high_res_head(high_res_hidden_states[:, -1]).sigmoid().squeeze(-1)
                    self.high_res_pred = high_res_scores.detach().cpu()
                    if (high_res_scores < self.high_res_threshold).all():
                        needs_high_res = False

            if self.first_branch_layer < self.twig_start_layer and not self.enable_high_res:
                for layer_idx in range(self.first_branch_layer, self.twig_start_layer):
                    hidden_states = self.layers[layer_idx](
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=None,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

            hidden_states_at_branch = hidden_states
            rpn_hidden_states = hidden_states
            if needs_high_res and not (self.enable_high_res and self.training):
                rpn_layers = self.twig_layers if self.roi_source == "hidden_states" else self.twig_layers[:-1]
                for twig_layer in rpn_layers:
                    rpn_hidden_states = twig_layer(
                        rpn_hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=None,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

            roi_maps = []
            if self.roi_enable2stage and needs_high_res and image_grid_thw is not None:
                qk_hidden_states = self.twig_layers[-1].input_layernorm(rpn_hidden_states)
                q_proj = self.twig_layers[-1].self_attn.q_proj(qk_hidden_states)
                k_proj = self.twig_layers[-1].self_attn.k_proj(qk_hidden_states)

                head_dim = self.twig_layers[-1].self_attn.head_dim
                num_heads = self.config.num_attention_heads
                num_key_value_heads = self.config.num_key_value_heads
                query_states = self.twig_layers[-1].self_attn.q_norm(
                    q_proj.view(batch_size, -1, num_heads, head_dim)
                ).transpose(1, 2)
                key_states = self.twig_layers[-1].self_attn.k_norm(
                    k_proj.view(batch_size, -1, num_key_value_heads, head_dim)
                ).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                key_states = repeat_kv(key_states, self.twig_layers[-1].self_attn.num_key_value_groups)

                for sample_idx in range(batch_size):
                    if visual_token_mask[sample_idx].sum() == 0:
                        roi_maps.append(None)
                        continue
                    grid_h = image_grid_thw[sample_idx, 1].item() // 2
                    grid_w = image_grid_thw[sample_idx, 2].item() // 2
                    key_sample = key_states[sample_idx, :, visual_token_mask[sample_idx], :]
                    if self.roi_post_training:
                        query_sample,_ = get_singleturn_query_text_hs_mheads(query_states[sample_idx : sample_idx + 1], labels[sample_idx : sample_idx + 1])
                        query_sample = query_sample[0]
                    else:
                        query_sample = query_states[sample_idx, :, -1:, :]
                    first_token_query = query_states[sample_idx,:1, :1, :] #B
                    attn_scores_first_token = torch.matmul(first_token_query, key_sample.transpose(-2, -1)) / math.sqrt(head_dim)
                    #print(attn_scores_first_token.shape, key_sample.shape, query_sample.shape)
                    roi_map_first_token = attn_scores_first_token.mean(dim=0).view(-1, grid_h, grid_w).sigmoid()
                    sink_mask = roi_map_first_token[0] < 0.75
                    if self.is_debug:
                        import numpy as np
                        import torchvision.transforms.functional as torchvision_F
                        from ..mm_utils import plot_image_with_heatmaps

                        maps_to_plot = [
                                {
                                    'map': roi_map_first_token[0],
                                    'title': 'Raw Heatmap',
                                    'blend': False,  # This will be a standard heatmap
                                    'cmap': 'viridis'  # Optional: specify a colormap
                                },
                                {
                                    'map': roi_map_first_token[0] > 0.75,
                                    'title': 'Blended Mask (Thresholded)',
                                    'blend': True  # This will be an overlay on the image
                                }
                            ]
                        plot_image_with_heatmaps(
                            image=np.array(src_images[sample_idx].convert("RGB")),
                            attention_maps=maps_to_plot,
                            dpi=200
                        )
                        pass
                    attn_scores = torch.matmul(query_sample, key_sample.transpose(-2, -1)) / math.sqrt(head_dim)
                    roi_map = attn_scores.mean(dim=0).view(grid_h, grid_w)
                    roi_maps.append(roi_map)

                src_image_features = []
                for sample_idx in range(batch_size):
                    src_image_features.append(
                        hidden_states_initial[sample_idx, image_token_start : image_token_start + visual_token_counts[sample_idx]]
                    )

                (
                    sub_image_features,
                    sub_image_counts,
                    sub_image_grid_thw,
                    roi_mask_list,
                    roi_surrounding_bbox,
                    sub_image_deepstack,
                    bbox_img,
                ) = get_batched_sub_images_v2(
                    roi_maps,
                    src_images,
                    image_processor,
                    encode_image_fn,
                    image_grid_thw,
                    self.roi_conf_thresh,
                    return_deepstack=True,
                    img_llm_patch_size=self.img_llm_patch_size,
                    is_debug=self.is_debug,
                    add_noise_to_roi = self.add_noise_to_roi and self.roi_post_training,
                    sink_mask=sink_mask,
                )
                self.bbox_img = bbox_img
                if any(count > 0 for count in sub_image_counts):
                    (
                        hidden_states,
                        attention_mask,
                        _valid_token_mask,
                        input_ids_updated,
                        inserted_content_mask,
                        position_ids,
                        updated_visual_pos_masks,
                        updated_deepstack_embeds,
                    ) = insert_sub_feat_v2(
                        hidden_states_initial,
                        sub_image_features,
                        sub_image_counts,
                        self.visual_token_counts,
                        image_token_start,
                        labels,
                        raw_attn_mask,
                        roi_mask_list,
                        temp_input_ids,
                        reuse_src_pos=self.reuse_src_pos,
                        position_ids=position_ids,
                        pos_id_fn=pos_id_fn,
                        image_grid_thw=image_grid_thw,
                        bbox_grid_thw_list=sub_image_grid_thw,
                        surrounding_bbox_list=roi_surrounding_bbox,
                        deepstack_visual_embeds=deepstack_visual_embeds,
                        sub_img_deepstack_list=sub_image_deepstack,
                        image_token_id=self.image_token_id,
                        video_token_id=self.video_token_id,
                        return_deepstack=True,
                        insert_after_text=False,
                    )

                    self.roi_new_hs_mask = inserted_content_mask
                    self.generation_attention_mask = attention_mask
                    self.last_position_ids = position_ids[:, :, -1:]
                    self.last_pos_id = self.last_position_ids
                    self.bbox_image_grid_thw = sub_image_grid_thw
                    self.roi_surrounding_bbox = roi_surrounding_bbox
                    self.input_ids_updated = input_ids_updated
                    if updated_visual_pos_masks is not None:
                        visual_pos_masks = updated_visual_pos_masks
                    if updated_deepstack_embeds is not None:
                        deepstack_visual_embeds = updated_deepstack_embeds

                    new_visual_counts = []
                    for ids_i in input_ids_updated:
                        count = (ids_i == self.image_token_id).sum() + (ids_i == self.video_token_id).sum()
                        new_visual_counts.append(count.item())
                    source_img_token_num = self.visual_token_counts[0].item() if self.visual_token_counts.numel() > 0 else 0
                    self.visual_token_counts = torch.tensor(new_visual_counts, device=hidden_states.device, dtype=torch.long)
                    self.visual_token_num = self.visual_token_counts[0].item()

                    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                        text_position_ids = position_ids[0]
                        position_ids = position_ids[1:]
                    else:
                        text_position_ids = position_ids[0]

                    # Reuse KV cache for inference after RoI insertion.
                    if not self.training and self.enable_high_res:
                        past_seq_len = int(image_token_start + source_img_token_num)
                        past_seq_len = min(past_seq_len, hidden_states.shape[1])

                        past_key_values = slice_kv_cache(
                            past_key_values,
                            past_seq_len,
                            config=self.config,
                        )
                        hidden_states_k_past = hidden_states_at_branch[:, :past_seq_len, :]

                        new_position_ids = position_ids[:, :, past_seq_len:]
                        new_text_position_ids = text_position_ids[:, past_seq_len:]
                        new_inputs_embeds = hidden_states[:, past_seq_len:, :]
                        new_seq_len = new_inputs_embeds.shape[1]
                        new_cache_position = torch.arange(
                            past_seq_len,
                            past_seq_len + new_seq_len,
                            device=hidden_states.device,
                        )

                        new_hidden_states = new_inputs_embeds
                        all_position_embeddings = self.rotary_emb(hidden_states, position_ids)
                        new_position_embeddings = (
                            all_position_embeddings[0][:, past_seq_len:, :],
                            all_position_embeddings[1][:, past_seq_len:, :],
                        )
                        new_causal_mask = create_causal_mask(
                            config=self.config,
                            input_embeds=new_hidden_states,
                            attention_mask=attention_mask,
                            cache_position=new_cache_position,
                            past_key_values=past_key_values,
                            position_ids=new_text_position_ids,
                        )
                        new_decoder_position_ids = new_text_position_ids
                        replay_visual_mask = None
                        replay_deepstack_index = None
                        if visual_pos_masks is not None:
                            replay_visual_mask = visual_pos_masks[:, past_seq_len:]
                            if deepstack_visual_embeds is not None:
                                full_visual_mask = visual_pos_masks.bool()
                                full_positions = torch.nonzero(full_visual_mask.reshape(-1), as_tuple=False).squeeze(-1)
                                replay_visual_full_mask = torch.zeros_like(full_visual_mask, dtype=torch.bool)
                                replay_visual_full_mask[:, past_seq_len:] = full_visual_mask[:, past_seq_len:]
                                replay_positions = torch.nonzero(
                                    replay_visual_full_mask.reshape(-1), as_tuple=False
                                ).squeeze(-1)
                                if full_positions.numel() > 0 and replay_positions.numel() > 0:
                                    replay_deepstack_index = torch.searchsorted(full_positions, replay_positions)
                        if (
                            new_causal_mask is None
                            and self.config._attn_implementation == "flash_attention_2"
                        ):
                            new_decoder_position_ids = torch.arange(
                                new_seq_len, device=hidden_states.device, dtype=new_text_position_ids.dtype
                            ).unsqueeze(0).expand(batch_size, -1)

                        for layer_idx in range(self.twig_start_layer):
                            decoder_layer = self.layers[layer_idx]
                            new_hidden_states = decoder_layer(
                                new_hidden_states,
                                attention_mask=new_causal_mask,
                                position_ids=new_decoder_position_ids,
                                past_key_values=past_key_values,
                                cache_position=new_cache_position,
                                position_embeddings=new_position_embeddings,
                                **kwargs,
                            )
                            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                                layer_visual_embeds = deepstack_visual_embeds[layer_idx]
                                if (
                                    layer_visual_embeds is not None
                                    and replay_visual_mask is not None
                                    and replay_deepstack_index is not None
                                    and replay_deepstack_index.numel() > 0
                                ):
                                    replay_layer_embeds = layer_visual_embeds[replay_deepstack_index]
                                    if int(replay_visual_mask.sum().item()) == int(replay_layer_embeds.shape[0]):
                                        new_hidden_states = self._deepstack_process(
                                            new_hidden_states,
                                            replay_visual_mask,
                                            replay_layer_embeds,
                                        )

                        hidden_states_k_new = new_hidden_states
                        hidden_states = torch.cat([hidden_states_k_past, hidden_states_k_new], dim=1)
                        all_cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
                        full_causal_mask = create_causal_mask(
                            config=self.config,
                            input_embeds=hidden_states,
                            attention_mask=attention_mask,
                            cache_position=all_cache_position,
                            past_key_values=past_key_values,
                            position_ids=text_position_ids,
                        )
                        full_decoder_position_ids = text_position_ids
                        if (
                            full_causal_mask is None
                            and self.config._attn_implementation == "flash_attention_2"
                        ):
                            full_decoder_position_ids = torch.arange(
                                hidden_states.shape[1], device=hidden_states.device, dtype=text_position_ids.dtype
                            ).unsqueeze(0).expand(batch_size, -1)

                        for layer_idx in range(self.twig_start_layer, len(self.layers)):
                            decoder_layer = self.layers[layer_idx]
                            hidden_states = decoder_layer(
                                hidden_states,
                                attention_mask=full_causal_mask,
                                position_ids=full_decoder_position_ids,
                                past_key_values=past_key_values,
                                cache_position=all_cache_position,
                                position_embeddings=all_position_embeddings,
                                **kwargs,
                            )
                            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                                hidden_states = self._deepstack_process(
                                    hidden_states,
                                    visual_pos_masks,
                                    deepstack_visual_embeds[layer_idx],
                                )
                    else:
                        past_key_values = None
                        if use_cache and not torch.jit.is_tracing():
                            past_key_values = DynamicCache(config=self.config)
                        cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
                        attention_mask = create_causal_mask(
                            config=self.config,
                            input_embeds=hidden_states,
                            attention_mask=attention_mask,
                            cache_position=cache_position,
                            past_key_values=past_key_values,
                            position_ids=text_position_ids,
                        )
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)

                        for layer_idx, decoder_layer in enumerate(self.layers):
                            hidden_states = decoder_layer(
                                hidden_states,
                                attention_mask=attention_mask,
                                position_ids=text_position_ids,
                                past_key_values=past_key_values,
                                cache_position=cache_position,
                                position_embeddings=position_embeddings,
                                **kwargs,
                            )
                            # add visual features to the hidden states of first several layers
                            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                                hidden_states = self._deepstack_process(
                                    hidden_states,
                                    visual_pos_masks,
                                    deepstack_visual_embeds[layer_idx],
                                )
                else:
                    needs_high_res = False

            if (not needs_high_res) or (self.training and not self.roi_enable2stage):
                for layer_idx in range(self.first_branch_layer, len(self.layers)):
                    hidden_states = self.layers[layer_idx](
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

            self.roi_post_training = self.roi_post_training and self.training
            if not self.roi_post_training:
                if self.training:
                    hidden_states = (
                        rpn_hidden_states if not self.enable_high_res else high_res_hidden_states
                    )
                else:
                    hidden_states = self.norm(hidden_states)
            else:
                hidden_states = self.norm(hidden_states)
        else:
            for layer_idx, decoder_layer in enumerate(self.layers):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                # add visual features to the hidden states of first several layers
                if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )
            hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


@auto_docstring
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        text_config = config.text_config
        text_config.mm_config = config
        text_config.vision_config = config.vision_config
        text_config.image_token_id = config.image_token_id
        text_config.video_token_id = config.video_token_id
        text_config.vision_start_token_id = config.vision_start_token_id
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        text_config.enable_twig = getattr(config, "enable_twig", False)
        text_config.twig_K = getattr(config, "twig_K", 0)
        text_config.twig_T = getattr(config, "twig_T", 0)
        text_config.roi_source = getattr(config, "roi_source", "qk")
        text_config.roi_loss = getattr(config, "roi_loss", "bce")
        text_config.roi_super_type = getattr(config, "roi_super_type", "v1")
        text_config.roi_multi_head = getattr(config, "roi_multi_head", False)
        text_config.roi_enable2stage = getattr(config, "roi_enable2stage", False)
        text_config.enable_high_res = getattr(config, "enable_high_res", False)
        text_config.high_res_K = getattr(config, "high_res_K", text_config.twig_K)
        text_config.high_res_T = getattr(config, "high_res_T", text_config.twig_T)
        text_config.high_res_thresh = getattr(config, "high_res_thresh", 0.1)
        text_config.roi_post_training = getattr(config, "roi_post_training", False)
        text_config.reuse_src_pos = getattr(config, "reuse_src_pos", False)
        text_config.add_noise_to_roi = getattr(config, "add_noise_to_roi", False)
        self.language_model = Qwen3VLTextModel._from_config(text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

        if getattr(self.language_model, "twig_layers", None) is not None:
            #self.twig_layers = self.language_model.twig_layers
            self.twig_K = self.language_model.twig_K
            self.twig_T = self.language_model.twig_T

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        src_images: Optional[list] = None,
        image_processor: Optional[Any] = None,
        labels: Optional[torch.LongTensor] = None,
        processor: Optional[Any] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        image_token_start = 0
        if input_ids is not None and self.image_token_id is not None:
            if (input_ids == self.image_token_id).any():
                image_token_start = torch.where(input_ids[0] == self.image_token_id)[0][0].item()

        def rope_index_wrapper(
            input_ids: Optional[torch.LongTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            return self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )


        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            image_grid_thw=image_grid_thw,
            src_images=src_images,
            image_processor=image_processor,
            encode_image_fn=self.visual,
            labels=labels,
            image_token_start=image_token_start,
            pos_id_fn=rope_index_wrapper,
            temp_input_ids=input_ids,
            processor=processor,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen3VL causal language model (or autoregressive) outputs.
    """
)
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

        self.enable_twig = False
        self.roi_enable2stage = False
        self.roi_post_training = False
        self.enable_high_res = False
        self.image_processor = None
        if getattr(config, "enable_twig", False):
            self.enable_twig = True
            self.roi_source = getattr(config, "roi_source", "qk")
            self.roi_loss = getattr(config, "roi_loss", "bce")
            self.roi_super_type = getattr(config, "roi_super_type", "v1")
            self.roi_multi_head = getattr(config, "roi_multi_head", False)
            self.roi_enable2stage = getattr(config, "roi_enable2stage", False)
            self.roi_post_training = getattr(config, "roi_post_training", False)
            self.enable_high_res = getattr(config, "enable_high_res", False)
            if self.training:
                try:
                    self.image_processor = AutoProcessor.from_pretrained(
                        config.name_or_path, trust_remote_code=True
                    ).image_processor
                    self.image_processor.max_pixels = getattr(config, "max_pixels", None)
                    self.image_processor.min_pixels = getattr(config, "min_pixels", None)
                except Exception:
                    print(
                        "Warning: AutoProcessor.from_pretrained failed, please make sure the model path is correct when post training SD-RPN."
                    )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        roi_target_map: Optional[torch.LongTensor] = None,
        src_images: Optional[list] = None,
        processor: Optional[Any] = None,
        high_res_signal: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            src_images=src_images,
            image_processor=processor.image_processor if hasattr(processor, "image_processor") else self.image_processor,
            labels=labels,
            processor=processor,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None and self.training:
            text_model = self.model.language_model
            image_token_id = self.config.image_token_id
            video_token_id = self.config.video_token_id
            visual_token_count = (input_ids == image_token_id).sum().item()
            visual_mask = (input_ids == image_token_id) | (input_ids == video_token_id)

            position_ids_for_loss = position_ids
            if position_ids_for_loss is None:
                attention_mask_tensor = (
                    attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
                )
                if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                    attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                    if attention_mask_tensor.dtype.is_floating_point:
                        attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                        attention_mask_tensor = (1.0 - attention_mask_tensor).int()

                position_ids_for_loss, _ = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )

            if (not self.enable_twig) or self.roi_enable2stage or visual_token_count == 0 or self.roi_post_training:
                if self.roi_post_training:
                    inserted_content_mask = text_model.roi_new_hs_mask
                    if inserted_content_mask is not None:
                        labels = update_batched_labels(labels, inserted_content_mask, -100)

                logits = logits.float()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                if self.roi_post_training:
                    if text_model.roi_new_hs_mask is None:
                        zero_scalar = torch.tensor(0.0, device=shift_logits.device, dtype=shift_logits.dtype)
                        loss = shift_logits.sum() * zero_scalar
            elif self.enable_high_res:
                query_hidden = get_singleturn_query_text_hs(hidden_states, labels)
                high_res_score = text_model.high_res_head(query_hidden)
                if high_res_signal is not None:
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    high_res_signal = high_res_signal.type_as(high_res_score)
                    valid_labels_mask = (high_res_signal != -100)
                    high_res_signal = high_res_signal[valid_labels_mask]
                    high_res_score = high_res_score[valid_labels_mask]
                    loss = loss_fct(high_res_score.view(-1), high_res_signal.view(-1))
            else:
                batch_size = hidden_states.shape[0]
                hidden_states = text_model.twig_layers[-1].input_layernorm(hidden_states)
                mask_score_samples = []
                roi_target_maps = []
                for sample_idx in range(batch_size):
                    visual_mask_sample = visual_mask[sample_idx]
                    if not visual_mask_sample.any():
                        continue

                    hidden_sample = hidden_states[sample_idx]
                    if self.roi_super_type == "v1":
                        assert roi_target_map is not None, "roi_target_map must be provided for roi branch"
                    elif self.roi_super_type == "lazy":
                        raise NotImplementedError("Lazy super type is not implemented yet")

                    if self.roi_source == "qk":
                        q_proj = text_model.twig_layers[-1].self_attn.q_proj(hidden_sample.unsqueeze(0))
                        k_proj = text_model.twig_layers[-1].self_attn.k_proj(hidden_sample.unsqueeze(0))

                        head_dim = text_model.twig_layers[-1].self_attn.head_dim
                        num_heads = self.config.text_config.num_attention_heads
                        num_key_value_heads = self.config.text_config.num_key_value_heads
                        query_states = text_model.twig_layers[-1].self_attn.q_norm(
                            q_proj.view(1, -1, num_heads, head_dim)
                        ).transpose(1, 2)
                        key_states = text_model.twig_layers[-1].self_attn.k_norm(
                            k_proj.view(1, -1, num_key_value_heads, head_dim)
                        ).transpose(1, 2)

                        position_embeddings = text_model.rotary_emb(
                            hidden_states[sample_idx : sample_idx + 1],
                            position_ids_for_loss[:, sample_idx : sample_idx + 1],
                        )
                        cos, sin = position_embeddings
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                        key_states = repeat_kv(key_states, text_model.twig_layers[-1].self_attn.num_key_value_groups)

                        query_states, _ = get_singleturn_query_text_hs_mheads(query_states, labels[sample_idx : sample_idx + 1])
                        key_states = key_states[:, :, visual_mask_sample]
                        if self.roi_multi_head == False:
                            query_states = query_states.transpose(1, 2).flatten(2,3) #b,q_len, dim
                            key_states = key_states.transpose(1, 2).flatten(2,3) #b,k_len, dim
                            mask_score_sample = query_states @ key_states.transpose(-1, -2)
                            mask_score = mask_score_sample.mean(dim=1) # average over heads and queries -> (1, k_len)
                        else:
                            mask_score_sample = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                            if self.roi_loss == 'mse':
                                mask_score_sample = nn.functional.softmax(mask_score_sample, dim=-1,dtype=torch.float32).to(query_states.dtype)
                            mask_score = mask_score_sample.mean(dim=1).mean(dim=1)  # average over heads and queries -> (1, k_len)

                        # Remove the batch dimension of 1 for loss calculation
                        mask_score = mask_score.squeeze(0)
                        # Get the target for the current sample
                        target_sample = roi_target_map[sample_idx].to(mask_score.device).float() #h_tar, w_tar
                        if target_sample.ndim == 1:
                            htar = wtar = int(math.sqrt(target_sample.shape[0]))
                            target_sample = target_sample.view(htar, wtar)  # (h_tar, w_tar)
                        h_patch, w_patch = image_grid_thw[sample_idx][1], image_grid_thw[sample_idx][2]
                        if h_patch != target_sample.shape[0]*2 or w_patch != target_sample.shape[1]*2:
                            mask_score_reshaped = mask_score.view(-1, h_patch//2, w_patch//2)
                            mask_score = nn.functional.interpolate(
                                mask_score_reshaped.unsqueeze(0),
                                size=(target_sample.shape[0], target_sample.shape[1]),
                                mode='bilinear',
                                align_corners=False,
                            ).squeeze(0).squeeze(0) # (h_tar, w_tar)
                        roi_target_maps.append(target_sample.flatten())
                        mask_score_samples.append(mask_score.flatten())

                roi_target_flat = torch.cat(roi_target_maps, dim=0) if roi_target_maps else torch.tensor([], device=hidden_states.device, dtype=labels.dtype)
                mask_score_flat = torch.cat(mask_score_samples, dim=0) if mask_score_samples else torch.tensor([], device=hidden_states.device, dtype=labels.dtype)

                flat_labels = roi_target_flat.contiguous().view(-1)
                valid_labels_mask = flat_labels != -100
                if valid_labels_mask.shape[0] != mask_score_flat.shape[0]:
                    pass
                scores_for_loss = mask_score_flat[valid_labels_mask]
                labels_for_loss = flat_labels[valid_labels_mask].to(dtype=scores_for_loss.dtype, device=scores_for_loss.device)
                if scores_for_loss.numel() > 0:
                    if self.roi_loss == "bce":
                        loss_fct = torch.nn.BCEWithLogitsLoss()
                        loss = loss_fct(scores_for_loss, labels_for_loss)
                    elif self.roi_loss == "mse":
                        loss = F.mse_loss(scores_for_loss, labels_for_loss)
                    else:
                        raise NotImplementedError(f"ROI loss {self.roi_loss} not implemented")
                else:
                    loss = mask_score_flat.sum() * 0.0
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def load_twig_weights_from_original_model(self, model_args):
        if not self.model.language_model.enable_twig or not self.enable_twig:
            print("Twig is not enabled in the configuration. Skipping loading twig weights.")
            return

        if not model_args.model_name_or_path:
            raise ValueError("model_args.model_name_or_path must be provided to load twig weights.")

        twig_init = getattr(model_args, "twig_init", True)
        if not twig_init:
            print("Skipping init twig weights from pretrained model.")

        enable_high_res_training = getattr(model_args, "enable_high_res", False)
        twig_layers = getattr(self.model.language_model, "twig_layers", None)
        if twig_layers is not None and len(twig_layers) == self.model.language_model.twig_T and not enable_high_res_training:
            for twig_idx in range(self.model.language_model.twig_T):
                if not twig_init:
                    continue
                original_layer_idx = self.model.language_model.twig_K + twig_idx
                base_layers = getattr(self.model.language_model, "layers", [])
                if original_layer_idx < len(base_layers):
                    try:
                        state_dict_to_load = base_layers[original_layer_idx].state_dict()
                        twig_layers[twig_idx].load_state_dict(state_dict_to_load)
                        print(f"  Loaded weights from original_model.model.language_model.layers[{original_layer_idx}]")
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load state_dict for twig_layer {twig_idx} from self.model.language_model.layers[{original_layer_idx}]: {e}"
                        )
                else:
                    raise ValueError(
                        f"Not enough layers in self.model.language_model.layers ({len(base_layers)} layers) "
                        f"to initialize self.model.language_model.twig_layers[{twig_idx}]. Needed layer index {original_layer_idx}."
                    )
        else:
            print(
                f"self.model.language_model.twig_layers is None or has incorrect length. Expected {self.model.language_model.twig_T} layers. Skipping twig_layers initialization."
            )

        if enable_high_res_training and getattr(self.model.language_model, "high_res_layers", None) is not None:
            high_res_layers = self.model.language_model.high_res_layers
            high_res_layer_count = getattr(self.model.language_model, "high_res_T", len(high_res_layers))
            high_res_start_layer = getattr(self.model.language_model, "high_res_K", self.model.twig_K)
            if len(high_res_layers) == high_res_layer_count:
                for high_res_idx in range(high_res_layer_count):
                    if not twig_init:
                        continue
                    original_layer_idx = high_res_start_layer + high_res_idx
                    base_layers = getattr(self.model.language_model, "layers", [])
                    if original_layer_idx < len(base_layers):
                        try:
                            state_dict_to_load = base_layers[original_layer_idx].state_dict()
                            high_res_layers[high_res_idx].load_state_dict(state_dict_to_load)
                            print(f"  Loaded weights from original_model.model.language_model.layers[{original_layer_idx}]")
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to load state_dict for high_res_layers {high_res_idx} "
                                f"from self.model.language_model.layers[{original_layer_idx}]: {e}"
                            )
                    else:
                        raise ValueError(
                            f"Not enough layers in self.model.language_model.layers ({len(base_layers)} layers) "
                            f"to initialize self.model.high_res_layers[{high_res_idx}]. Needed layer index {original_layer_idx}."
                        )
        # if enable_high_res_training:
        #     max_layer_needed = max(self.model.twig_K, getattr(self.model.language_model, "high_res_K", self.model.twig_K))
        # else:
        #     max_layer_needed = self.model.twig_K

        # if max_layer_needed < len(self.model.layers):
        #     print(f"  Pruning self.model.layers from index {max_layer_needed} onwards.")
        #     print(f"    Original number of layers: {len(self.model.layers)}")
        #     self.model.layers = self.model.layers[:max_layer_needed]

__all__ = [
    "Qwen3VLVisionModel",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLTextModel",
]
