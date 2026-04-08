import numpy as np
import torch

from qwen_src.ana_utils import get_attn_weights_map, get_bbox4src, register_hooks
from qwen_src.mm_utils import create_pseudo_labels


class Qwen25Stage1PseudoRoiHelper:
    """ROI helper that mirrors Stage-1 pseudo-label bbox construction for Qwen2.5-VL-7B."""

    _TEXTUAL_REQUIRED_HEADS = {22: [1, 4], 23: [6, 11]}
    _TEXTUAL_SINK_HEADS = {}
    _NATURAL_REQUIRED_HEADS = {16: [1, 7, 17], 19: [17, 20]}
    _NATURAL_SINK_HEADS = {9: [4, 22]}
    _DOC_OCR_TASK_KEYWORDS = ("docvqa", "chartqa", "ocrbench", "infovqa", "textvqa")

    def __init__(self, model, processor, device, device_map="auto"):
        self.model = model
        self.processor = processor
        self.device = device
        self.device_map = device_map
        self.activations, self.hooks = register_hooks(self.model, is_qwen3vl=False)

    def _to_device(self, inputs):
        if self.device_map == "auto":
            return inputs.to("cuda")
        return inputs.to(self.device)

    def _build_inputs(self, image, prompt_text):
        msg = [{"role": "user", "content": [{"type": "text", "text": "<image>\n" + prompt_text}]}]
        text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "").replace(
            "<image>", "<image><|vision_start|><|image_pad|><|vision_end|>"
        )
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        return self._to_device(inputs)

    def _ensure_visual_mask(self, inputs):
        if self.activations.get("visual_mask", None) is not None:
            return
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        input_ids_list = inputs["input_ids"].tolist()[0]
        try:
            pos = input_ids_list.index(vision_start_id) + 1
            pos_end = input_ids_list.index(vision_end_id)
        except ValueError:
            return

        visual_mask = torch.zeros((1, len(input_ids_list)), dtype=torch.bool, device=inputs["input_ids"].device)
        visual_mask[0, pos:pos_end] = True
        self.activations["visual_mask"] = visual_mask

    @staticmethod
    def _aggregate_attention(attn_weights_map, output_shape, head_map):
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
            layer_attn = np.asarray(attn_weights_map[layer_key]["output2images"]).reshape(
                -1, output_shape[0], output_shape[1]
            )
            for head in heads:
                if head < layer_attn.shape[0]:
                    output += layer_attn[head]
        return output / float(total_heads)

    @staticmethod
    def _infer_mode(question_text: str):
        text = (question_text or "").lower()
        if "output the grounding bounding box" in text:
            return "natural"
        return "textual"

    def _resolve_mode(self, task_name: str, question_text: str):
        task = (task_name or "").strip().lower()
        if task:
            if any(k in task for k in self._DOC_OCR_TASK_KEYWORDS):
                return "textual"
            return "natural"
        return self._infer_mode(question_text)

    def _get_heads(self, mode: str):
        if mode == "natural":
            return self._NATURAL_REQUIRED_HEADS, self._NATURAL_SINK_HEADS
        return self._TEXTUAL_REQUIRED_HEADS, self._TEXTUAL_SINK_HEADS

    def compute_bbox(
        self,
        image,
        question_text: str,
        task_name: str = "",
        max_new_tokens: int = 1,
        sink_thresh: float = 1e-2,
        binary_coff: float = 0.2,
        bg_coff: float = 0.1,
        pseudo_blur_kernel_size: int = 3,
    ):
        mode = self._resolve_mode(task_name=task_name, question_text=question_text)
        required_heads, sink_heads = self._get_heads(mode)

        inputs = self._build_inputs(image=image, prompt_text=question_text)
        self.activations.clear()
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                do_sample=False,
                use_cache=True,
            )

        self._ensure_visual_mask(inputs)
        if self.activations.get("visual_mask", None) is None:
            return None

        image_grid_thw = inputs["image_grid_thw"].detach().cpu().numpy().squeeze(0)
        feat_h = int(image_grid_thw[1] // 2)
        feat_w = int(image_grid_thw[2] // 2)
        output_shape = np.array([feat_h, feat_w], dtype=np.int32)
        # Align with Stage-1 path: each feature cell corresponds to a 28x28 region in resized image space.
        image_size = (feat_h * 28, feat_w * 28)
        required_layers = list(required_heads.keys()) + list(sink_heads.keys())
        layers_max = max(required_layers) + 1 if required_layers else 0
        attn_weights_map = get_attn_weights_map(
            self.activations, layers=layers_max, required_layers=required_layers
        )

        grounding_attn_o2i = self._aggregate_attention(attn_weights_map, output_shape, required_heads)
        sink_attn = self._aggregate_attention(attn_weights_map, output_shape, sink_heads)

        pseudo_set = create_pseudo_labels(
            sink_attn=sink_attn,
            grounding_attn_o2i=grounding_attn_o2i,
            sink_thresh=sink_thresh,
            binary_coff=binary_coff,
            K=100,
            pseudo_gaussian_smooth=False,
            ab_sink=False,
            ab_fg_bbox=False,
            mask_known_bg=False,
            original_image_size=None,
            bg_coff=bg_coff,
            pseudo_blur_kernel_size=pseudo_blur_kernel_size,
        )

        fg_bbox = pseudo_set.get("fg_bbox", None)
        if fg_bbox is None:
            return None

        # fg_bbox format: (min_row, min_col, max_row, max_col), with max inclusive.
        noisy_bbox = torch.tensor(
            [[fg_bbox[1], fg_bbox[0], fg_bbox[3] + 1, fg_bbox[2] + 1]],
            dtype=torch.float32,
        )
        bbox_src = get_bbox4src(
            noisy_bbox,
            image,
            feat_size=(feat_h, feat_w),
            img_size=image_size,
        )[0].tolist()
        x1, y1, x2, y2 = [int(v) for v in bbox_src]
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
