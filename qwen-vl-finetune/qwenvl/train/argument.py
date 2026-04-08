import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

     #for ROI prediction
    enable_twig: bool = field(default=False) #enable twig layers for roi prediction
    twig_K: Optional[int] = field(default=0) #how many layers to keep before roi prediction
    twig_T: Optional[int] = field(default=0) # how many layers are used to used as roi predictor
    roi_branch: bool = field(default=False) # just keep the same as enable_twig
    twig_freeze: Optional[int] = field(default=None)  # If None, do not freeze any layers. If int, freeze the last `twig_freeze` layers.
    roi_source: Optional[str] = field(default='qk')  # 'hidden_states' or 'qk'
    roi_loss: Optional[str] = field(default='bce') #bce or focal loss
    twig_init: Optional[bool] = field(default=False) #just set it to true for simplicity, init twig_t weights from original llava's model
    roi_super_type: Optional[str] = field(default='lazy')  #'v1' or 'lazy', v1 for llava's self prediction as supervision while lazy use supervision tokens from llava's annotation
    roi_multi_head: Optional[bool] = field(default=False)  # If True, use multi-head ROI loss, otherwise use single head
    roi_skip_ffn: Optional[bool] = field(default=False)  # If True, skip FFN in roi prediction
    roi_keep_ffn_mod_ratio: Optional[float] = field(default=1000)  # Modulation ratio for skipping FFN in roi prediction
    enable_high_res: Optional[bool] = field(default=False)  # Enable high-resolution input handling
    high_res_K: Optional[int] = field(default=twig_T)  # Number of layers to keep for high-resolution input
    roi_post_training: Optional[bool] = field(default=False)  # If True, enable post-training for ROI prediction
    reuse_src_pos: Optional[bool] = field(default=False)  # If True, reuse source position embeddings for ROI tokens
    add_noise_to_roi: Optional[bool] = field(default=False)  # whether to add noise to roi boxes during training

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)

    roi_samples: int = -1
    roi_data_path: Optional[str] = field(default=None)
    ab_sink: Optional[bool] = field(default=False)  # only used for ablation study,
    ab_fg_bbox: Optional[bool] = field(default=False)  # only used for ablation study, use foreground box as pseudo label
    fix_res: Optional[bool] = field(default=False)  # fix the image resolution to 576 tokens by resizing and expanding square
    multi_scale_training: Optional[bool] = field(default=False)  # use multi-scale training
    roi_binary_coeff: Optional[float] = field(default=0.2)
    bg_coff: Optional[float] = field(default=0.1)
    pseudo_blur_kernel_size: Optional[int] = field(default=3)  # Gaussian blur kernel size for pseudo-label smoothing
    KL_lower_bound: Optional[float] = field(default=0.2)
    KL_upper_bound: Optional[float] = field(default=0.5)
    transition_mode: Optional[bool] = field(default=False)
    is_2_5_stage: Optional[bool] = field(default=False)  # whether it is in the 2.5 stage training
    high_res_signal_strategy: Optional[str] = field(
        default="default"
    )  # default | prediction_vs_gt

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
