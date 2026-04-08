from PIL import Image
from io import BytesIO
import base64
import json
import os
import torch
import math
import ast
import random

from transformers import StoppingCriteria

import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as torchvision_F
import torch.nn as nn
import time


def _save_roi_debug_from_mmutils(src_img, roi_img, bbox_coords, batch_idx):
    """
    Optional ROI debug saver for two-stage ROI path.
    Controlled by env:
      LMMS_SAVE_ROI_DEBUG=1
      LMMS_ROI_DEBUG_ROOT=/path/to/root
      LMMS_ROI_DEBUG_METHOD=ours_stage3
      LMMS_ROI_DEBUG_TASK=vstar_bench
    """
    flag = os.environ.get("LMMS_SAVE_ROI_DEBUG", "0").lower() in ("1", "true", "yes")
    if not flag:
        return
    try:
        root = os.environ.get("LMMS_ROI_DEBUG_ROOT", "./logs/roi_debug")
        method = os.environ.get("LMMS_ROI_DEBUG_METHOD", "ours_stage3")
        task = os.environ.get("LMMS_ROI_DEBUG_TASK", "unknown_task")
        rank = os.environ.get("LOCAL_RANK", "0")
        raw_doc_ids = os.environ.get("LMMS_ROI_DEBUG_DOC_IDS", "")
        doc_ids = [x.strip() for x in raw_doc_ids.split(",") if x.strip()]
        doc_id = None
        if int(batch_idx) < len(doc_ids):
            doc_id = doc_ids[int(batch_idx)]
        did = str(doc_id) if doc_id is not None else "na"
        out_dir = os.path.join(root, task, f"doc_{did}")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        uniq = f"{ts}_r{rank}_d{did}_b{int(batch_idx)}_{int(time.time()*1e6)%1000000:06d}"
        src_path = os.path.join(out_dir, f"{method}_{uniq}_src.jpg")
        roi_path = os.path.join(out_dir, f"{method}_{uniq}_roi.jpg")
        src_img.save(src_path)
        roi_img.save(roi_path)
        meta = {
            "record_type": "roi_crop",
            "method": method,
            "task": task,
            "doc_id": doc_id,
            "batch_idx": int(batch_idx),
            "bbox_xyxy": [int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3])],
            "src_path": src_path,
            "roi_path": roi_path,
        }
        meta_path = os.path.join(out_dir, "roi_meta.jsonl")
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception:
        # keep inference path robust; debug save should never break forward
        pass


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def get_foreground_bbox(array_2d):
    """
    Get bounding box of foreground (non-zero) area in a 2D array.

    Args:
        array_2d: numpy array of shape (24, 24) or any 2D array

    Returns:
        tuple: (min_row, min_col, max_row, max_col) or None if no foreground
               - min_row, min_col: top-left corner of bounding box
               - max_row, max_col: bottom-right corner of bounding box (inclusive)
    """
    # Find all non-zero positions
    fg_positions = np.where(array_2d > 0)
    full_bbox = (0, 0, array_2d.shape[0] - 1, array_2d.shape[1] - 1)
    # Check if there are any foreground pixels
    if len(fg_positions[0]) == 0:
        return full_bbox  # return full image as bbox if no foreground

    # Get bounding box coordinates
    min_row = np.min(fg_positions[0])
    max_row = np.max(fg_positions[0])
    min_col = np.min(fg_positions[1])
    max_col = np.max(fg_positions[1])

    return (min_row, min_col, max_row, max_col)

def get_foreground_mask(
    original_image_size: tuple,
    target_image_size: int,
    min_dimension_size: int = 1,
    return_bbox: bool = False,
) -> np.ndarray:
    """
    Calculates a boolean mask indicating the location of the original image
    content after it has been resized and padded into a square.

    This function mimics the logic of resizing an image to fit within a square
    of `target_image_size` while maintaining aspect ratio, and then padding it
    to become a full square.

    Args:
        original_image_size (tuple): A tuple (width, height) representing the
        target_image_size (int): The side length of the final square image (e.g., 336 for a 336x336 output).
        min_dimension_size (int): The minimum size for any dimension after the initial resize, defaults to 1.

    Returns:
        np.ndarray: A 2D boolean NumPy array of shape
                    (target_image_size, target_image_size) where `True`
                    indicates a foreground pixel and `False` indicates a
                    background (padded) pixel.
    """
    # 1. Get original dimensions
    original_width, original_height = original_image_size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Original image dimensions must be positive.")

    # 2. Calculate the size after aspect-ratio-preserving resize
    max_dim = max(original_width, original_height)

    new_height = max(
        int(original_height / max_dim * target_image_size),
        min_dimension_size
    )
    new_width = max(
        int(original_width / max_dim * target_image_size),
        min_dimension_size
    )

    # 3. Determine the offsets based on the `expand2square` padding logic.
    if new_width > new_height:
        x1 = 0
        y1 = (target_image_size - new_height) // 2
        x2 = new_width
        y2 = new_height + math.ceil((target_image_size - new_height) / 2)
    elif new_height > new_width:
        x1 = (target_image_size - new_width) // 2
        y1 = 0
        x2 = new_width + math.ceil((target_image_size - new_width) / 2)
        y2 = new_height
    else:
        x1, y1 = 0, 0
        x2, y2 = target_image_size, target_image_size

    # Ensure the coordinates do not exceed the canvas dimensions due to rounding.
    x2 = min(x2, target_image_size)
    y2 = min(y2, target_image_size)

    # 4. Create the boolean mask
    mask = np.zeros((target_image_size, target_image_size), dtype=bool)
    # Note: NumPy slicing is [row_start:row_end, col_start:col_end], which corresponds to [y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = True
    if return_bbox:
        return mask, (y1, x1, y2-1, x2-1)
    return mask



def create_pseudo_labels(sink_attn, grounding_attn_o2i, sink_thresh=1e-2, binary_coff=0.2, K=100, max_ratio_limit=0.5,
                         bg_coff=0.1, pseudo_gaussian_smooth=False, ab_sink=False, ab_fg_bbox=False, mask_known_bg = True,
                         original_image_size=None, for_vis = False, use_smoothing = False, pseudo_blur_kernel_size=3):
    """
    Create pseudo labels for foreground, background, and ignore tokens.

    Args:
        sink_attn: array of shape (576,), layer 2 attention
        grounding_attn_o2i: array of shape (576,), original grounding attention
        sink_thresh: float, threshold to identify sink tokens
        binary_coff: float, coefficient to threshold grounding attention
        K: int, number of background tokens to select
        max_ratio_limit: float, maximum ratio of foreground, exceeding this will set the sample as ignore

    Returns:
        dict: {
            'fg_mask': binary mask for foreground tokens,
            'bg_mask': binary mask for background tokens,
            'ignore_mask': binary mask for ignore tokens,
            'labels': combined labels (0=bg, 1=fg, -1=ignore),
            'fg_bbox': foreground bounding box,
            'stats': statistics about the labeling
        }
    """
    # Ensure all inputs are 1D
    grounding_attn_o2i_ori = grounding_attn_o2i.copy()
    h,w = grounding_attn_o2i.shape[0], grounding_attn_o2i.shape[1] if len(grounding_attn_o2i.shape) > 1 else 24

    grounding_attn_o2i = grounding_attn_o2i.flatten()
    sink_attn = sink_attn.flatten()
    #print('binary_coff:', binary_coff, ' bg_coff:', bg_coff)
    if False:
        #print('[Debug] Gaussian smoothing applied to grounding attention')
        grounding_attn_2d = grounding_attn
        grounding_attn_2d = torch.tensor(grounding_attn_2d, dtype=torch.float32)
        # Apply Gaussian smoothing
        grounding_attn_2d = torchvision_F.gaussian_blur(grounding_attn_2d.unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=1.0).flatten()
        grounding_attn = grounding_attn_2d.numpy()

    # 1. Identify sink tokens
    sink_token_mask = (sink_attn >= sink_thresh).astype(bool)
    if mask_known_bg:

        known_fg_mask = get_foreground_mask(original_image_size, h, min_dimension_size=1)
        try:
            sink_token_mask = sink_token_mask | (~known_fg_mask.flatten())
        except:
            print("known_fg_mask shape:", known_fg_mask.shape)
            print("sink_token_mask shape:", sink_token_mask.shape)
            raise ValueError("known_fg_mask and sink_token_mask must have the same number of elements when flattened.")
    grounding_attn = grounding_attn_o2i * (~sink_token_mask).astype(float)  # remove sink tokens

    #smoothing before binary thresholding
    if use_smoothing:
        grounding_attn_2d = grounding_attn.reshape(h, w)
        grounding_attn_smoothed = grounding_attn_2d.copy()
        blurred_grounding_attn_2d = torch.tensor(grounding_attn_smoothed).unsqueeze(0).unsqueeze(0)
        blurred_grounding_attn_2d = torchvision_F.gaussian_blur(blurred_grounding_attn_2d, kernel_size=5, sigma=1.0).squeeze().squeeze()
        blurred_mask = (blurred_grounding_attn_2d > (blurred_grounding_attn_2d.max() * (binary_coff + bg_coff)/2))
        grounding_attn = grounding_attn * blurred_mask.numpy().flatten()


    binary_mask = (grounding_attn > grounding_attn.max() * binary_coff).astype(float)
    grounding_attn *= binary_mask

    # 2. Identify foreground tokens (where grounding_attn > 0)
    fg_mask = (grounding_attn > 0).astype(bool)

    # 3. Get foreground bounding box from grounding_attn
    blur_kernel_size = int(pseudo_blur_kernel_size) if pseudo_blur_kernel_size is not None else 3
    if blur_kernel_size < 1:
        blur_kernel_size = 1
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    smoothed_grounding_attn_2d = grounding_attn.reshape(h, w)
    smoothed_grounding_attn_2d = torch.tensor(smoothed_grounding_attn_2d).unsqueeze(0).unsqueeze(0)
    smoothed_grounding_attn_2d = torchvision_F.gaussian_blur(
        smoothed_grounding_attn_2d, kernel_size=blur_kernel_size, sigma=1.0
    ).squeeze().squeeze()
    fg_bbox = get_foreground_bbox(smoothed_grounding_attn_2d)

    # 4. Create mask for tokens inside fg bounding box
    min_row, min_col, max_row, max_col = fg_bbox

    # Convert 1D indices to 2D coordinates

    row_indices, col_indices = np.divmod(np.arange(h * w), w)

    # Check if each token is inside the bounding box
    in_fg_box_mask = (
        (row_indices >= min_row) & (row_indices <= max_row) &
        (col_indices >= min_col) & (col_indices <= max_col)
    )
    # grounding_attn_o2i_in_box = grounding_attn_o2i * in_fg_box_mask.astype(float)   # only keep values in fg box
    # grounding_attn_o2i_in_box_fg_mask = (grounding_attn_o2i_in_box > grounding_attn_o2i_in_box.max()*bg_coff).astype(bool)
    # fg_mask = fg_mask | grounding_attn_o2i_in_box_fg_mask  # add tokens in fg box with high grounding attn to fg

    # 5. Find candidate background tokens
    # Tokens that are: NOT in fg box and set sink token score to 0
    bg_candidate_mask = (~in_fg_box_mask)  #& (~sink_token_mask)
    grounding_attn_o2i = grounding_attn_o2i * (sink_attn < sink_thresh).astype(float)  # remove sink tokens

    # 6. Select K background tokens from candidates based on grounding_attn_o2i values
    bg_mask = np.zeros(h*w, dtype=bool)

    if bg_candidate_mask.sum() > 0:
        # Get candidate positions and their grounding_attn_o2i values
        candidate_indices = np.where(bg_candidate_mask)[0]
        # candidate_values = grounding_attn_o2i[candidate_indices]

        # Identify which of these candidates have a grounding_attn_o2i value smaller than the threshold
        candidate_grounding_values = grounding_attn_o2i[candidate_indices]
        selected_candidates_by_threshold_mask = (candidate_grounding_values < grounding_attn.max()*bg_coff)

        # Get the original indices (from bg_candidate_mask) of these selected candidates
        selected_bg_indices = candidate_indices[selected_candidates_by_threshold_mask]

        bg_mask[selected_bg_indices] = True

    if mask_known_bg:
        bg_mask = bg_mask | (~known_fg_mask.flatten())

    if ab_sink and ab_fg_bbox:
        labels = grounding_attn_o2i_ori
        return {
            'labels': labels,
        }

    # 7. All other tokens are ignored
    ignore_mask = ~(fg_mask | bg_mask)

    # 8. Create combined labels (-1=ignore, 0=bg, 1=fg)
    if not for_vis:
        labels = np.full(h*w, -100, dtype=int)  # Start with all ignore
    else:
        labels = np.full(h*w, -1, dtype=int)
    if (max_col-min_col + 1) * (max_row-min_row + 1) < max_ratio_limit * h*w and grounding_attn.max()>=5e-3:
        labels[bg_mask] = 0  # Background
        labels[fg_mask] = 1  # Foreground
    else:
        #print(f"Warning: Foreground bounding box too large, setting all as ignore. Ratio: {(max_col-min_col + 1) * (max_row-min_row + 1) / 576:.2f}")
        bg_mask = np.zeros(h*w, dtype=bool)
        fg_mask = np.zeros(h*w, dtype=bool)

    labels = labels.reshape(h, w)
    # 9. Collect statistics
    stats = {
        'num_fg': fg_mask.sum(),
        'num_bg': bg_mask.sum(),
        'num_ignore': ignore_mask.sum(),
        'num_sink': sink_token_mask.sum(),
        'num_candidates': bg_candidate_mask.sum(),
        'fg_bbox': fg_bbox
    }

    return {
        'fg_mask': fg_mask,
        'bg_mask': bg_mask,
        'ignore_mask': ignore_mask,
        'labels': labels,
        'fg_bbox': fg_bbox,
        'stats': stats,
        'sink_token_mask': sink_token_mask,
        'grounding_attn': grounding_attn,
    }


# Helper function for batched bounding box calculation (PyTorch version)
def get_foreground_bbox_torch(attn_map_2d_batch: torch.Tensor, threshold: float = 0.0):
    """
    Calculates foreground bounding boxes for a batch of 2D attention maps.

    Args:
        attn_map_2d_batch (torch.Tensor): Batch of 2D attention maps.
                                          Shape: (B, H, W).
        threshold (float): Threshold to consider a pixel as foreground.

    Returns:
        torch.Tensor: Bounding boxes for each sample.
                      Shape: (B, 4) -> [x_min, y_min, x_max, y_max]
    """
    B, H, W = attn_map_2d_batch.shape
    device = attn_map_2d_batch.device

    bboxes = torch.zeros((B, 4), dtype=torch.long, device=device)

    for i in range(B):
        # Find coordinates of foreground pixels for the current sample
        fg_pixels = (attn_map_2d_batch[i] > threshold).nonzero(as_tuple=False)  # (N_fg_pixels, 2) -> [row, col]

        if fg_pixels.numel() == 0:  # No foreground pixels found
            bboxes[i] = torch.tensor([0, 0, H - 1, W - 1], device=device)  # Default to full image
        else:
            rows = fg_pixels[:, 0]
            cols = fg_pixels[:, 1]
            # bboxes[i] = torch.tensor([
            #     torch.min(rows), torch.min(cols),
            #     torch.max(rows), torch.max(cols)
            # ], device=device)
            bboxes[i] = torch.tensor([
                torch.min(cols), torch.min(rows),
                torch.max(cols)+1, torch.max(rows)+1
            ])
    return bboxes

def create_pseudo_labels_torch(
    sink_attn_batch: torch.Tensor,  # Shape: (B, 576)
    grounding_attn_o2i_batch: torch.Tensor,  # Shape: (B, 576)
    sink_thresh: float = 1e-3,
    binary_coff: float = 0.2,
    # K: int = 100, # K is not used in the provided threshold-based BG selection
    max_ratio_limit: float = 0.5,
    bg_coff: float = 0.1,
    grid_dim: int = 24  # Assuming a 24x24 grid for 576 tokens
):
    """
    Create pseudo labels for foreground, background, and ignore tokens (PyTorch batched version).

    Args:
        sink_attn_batch: Batch of layer 2 attentions, shape (B, 576)
        grounding_attn_o2i_batch: Batch of original grounding attentions, shape (B, 576)
        sink_thresh: Threshold to identify sink tokens
        binary_coff: Coefficient to threshold grounding attention for FG
        max_ratio_limit: Maximum ratio of foreground area, exceeding this sets sample to ignore
        bg_coff: Coefficient to threshold grounding_attn_o2i for BG selection, relative to FG's max attention
        grid_dim: Dimension of the square grid (e.g., 24 for a 24x24 grid)

    Returns:
        dict: {
            'fg_mask': Batched binary mask for foreground (B, 576),
            'bg_mask': Batched binary mask for background (B, 576),
            'ignore_mask': Batched binary mask for ignore (B, 576),
            'labels': Batched combined labels (0=bg, 1=fg, -100=ignore) (B, 576),
            'fg_bbox': Batched foreground bounding boxes (B, 4) [min_r, min_c, max_r, max_c],
            'stats': Dict of batched statistics (e.g., 'num_fg': (B,))
        }
    """
    B, num_tokens = sink_attn_batch.shape
    if num_tokens != grid_dim * grid_dim:
        raise ValueError(f"num_tokens ({num_tokens}) must match grid_dim*grid_dim ({grid_dim * grid_dim}).")
    device = sink_attn_batch.device

    # Ensure inputs are float for calculations involving them
    sink_attn_batch = sink_attn_batch.float()
    grounding_attn_o2i_batch = grounding_attn_o2i_batch.float()

    # 1. Calculate processed grounding_attn for Foreground (FG) identification
    # Remove sink tokens from grounding_attn_o2i influence
    sink_influence_mask = (sink_attn_batch < sink_thresh).float()
    grounding_attn_for_fg = grounding_attn_o2i_batch * sink_influence_mask

    # Binarize grounding_attn_for_fg
    # Get max per sample, keeping batch dim for broadcasting: (B, 1)
    grounding_attn_for_fg_max = torch.max(grounding_attn_for_fg, dim=1, keepdim=True)[0]
    # Avoid division by zero or issues if max is 0 by adding a small epsilon or handling zero max
    # For simplicity, if max is 0, all binary_mask will be 0.
    binary_threshold_fg = grounding_attn_for_fg_max * binary_coff
    binary_mask_fg = (grounding_attn_for_fg > binary_threshold_fg).float()

    grounding_attn_processed_fg = grounding_attn_for_fg * binary_mask_fg

    # 2. Identify sink tokens
    sink_token_mask_batch = (sink_attn_batch >= sink_thresh)  # (B, 576) boolean

    # 3. Identify foreground tokens
    fg_mask_batch = (grounding_attn_processed_fg > 0)  # (B, 576) boolean

    # 4. Get foreground bounding box from grounding_attn_processed_fg
    grounding_attn_2d_batch = grounding_attn_processed_fg.view(B, grid_dim, grid_dim)
    fg_bboxes_batch = get_foreground_bbox_torch(grounding_attn_2d_batch, threshold=1e-9)  # (B, 4)

    # 5. Create mask for tokens inside FG bounding box
    grid_indices_1d = torch.arange(num_tokens, device=device).unsqueeze(0)  # (1, 576)
    row_indices_grid = grid_indices_1d // grid_dim  # (1, 576)
    col_indices_grid = grid_indices_1d % grid_dim  # (1, 576)

    # Expand bbox dims for broadcasting: (B, 1)
    min_row_b = fg_bboxes_batch[:, 0].unsqueeze(1)
    min_col_b = fg_bboxes_batch[:, 1].unsqueeze(1)
    max_row_b = fg_bboxes_batch[:, 2].unsqueeze(1)
    max_col_b = fg_bboxes_batch[:, 3].unsqueeze(1)

    in_fg_box_mask_batch = (
        (row_indices_grid >= min_row_b) & (row_indices_grid <= max_row_b) &
        (col_indices_grid >= min_col_b) & (col_indices_grid <= max_col_b)
    )  # (B, 576) boolean

    # 6. Find candidate background (BG) tokens and select them
    # Tokens that are NOT in fg box. Sink token influence is handled in grounding_attn_o2i_for_bg_selection.
    bg_candidate_mask_batch = (~in_fg_box_mask_batch)  # (B, 576) boolean

    # Prepare grounding_attn_o2i for BG selection (original values, but with sinks zeroed out)
    grounding_attn_o2i_for_bg_selection = grounding_attn_o2i_batch * sink_influence_mask  # (B, 576)

    bg_mask_batch = torch.zeros_like(fg_mask_batch, dtype=torch.bool)  # (B, 576)

    # Threshold for BG selection is relative to the max of *processed* FG attention
    bg_selection_threshold_val = grounding_attn_for_fg_max * bg_coff  # (B, 1)

    for i in range(B):
        sample_bg_candidates_mask = bg_candidate_mask_batch[i]  # (576,)
        if sample_bg_candidates_mask.sum() > 0:
            candidate_indices_sample = torch.where(sample_bg_candidates_mask)[0]  # 1D tensor of indices

            candidate_grounding_values_sample = grounding_attn_o2i_for_bg_selection[
                i, candidate_indices_sample]  # Values for candidates

            # Select BG tokens if their original (sink-masked) grounding value is below threshold
            selected_by_thresh_mask_sample = (candidate_grounding_values_sample < bg_selection_threshold_val[i])

            selected_bg_indices_sample = candidate_indices_sample[selected_by_thresh_mask_sample]

            if selected_bg_indices_sample.numel() > 0:
                bg_mask_batch[i, selected_bg_indices_sample] = True

    # 7. Handle max_ratio_limit: if bbox too large, mark sample's FG and BG as empty
    fg_bbox_areas_batch = (fg_bboxes_batch[:, 2] - fg_bboxes_batch[:, 0] + 1) * \
                          (fg_bboxes_batch[:, 3] - fg_bboxes_batch[:, 1] + 1)  # (B,)

    invalid_ratio_batch_mask = (fg_bbox_areas_batch >= max_ratio_limit * num_tokens)  # (B,) boolean

    if invalid_ratio_batch_mask.any():
        # For samples with too large bbox, set their fg_mask and bg_mask to all False
        fg_mask_batch[invalid_ratio_batch_mask, :] = False
        bg_mask_batch[invalid_ratio_batch_mask, :] = False

    # 8. Create combined labels (-100=ignore, 0=bg, 1=fg)
    # Initialize with ignore value
    labels_batch = torch.full((B, num_tokens), -100, dtype=torch.long, device=device)
    labels_batch[bg_mask_batch] = 0  # Background (uses potentially modified bg_mask_batch)
    labels_batch[fg_mask_batch] = 1  # Foreground (uses potentially modified fg_mask_batch)

    # 9. All other tokens are ignored
    ignore_mask_batch = ~(fg_mask_batch | bg_mask_batch)  # (B, 576) boolean

    # 10. Collect statistics (batched tensors)
    stats_batch = {
        'num_fg': fg_mask_batch.sum(dim=1),
        'num_bg': bg_mask_batch.sum(dim=1),
        'num_ignore': ignore_mask_batch.sum(dim=1),
        'num_sink': sink_token_mask_batch.sum(dim=1),
        'num_bg_candidates_initially': bg_candidate_mask_batch.sum(dim=1),  # Before thresholding BG based on values
        'fg_bbox_area': fg_bbox_areas_batch,
        'failed_max_ratio_limit': invalid_ratio_batch_mask.long()  # 0 or 1
    }

    return {
        'fg_mask': fg_mask_batch,
        'bg_mask': bg_mask_batch,
        'ignore_mask': ignore_mask_batch,
        'labels': labels_batch,
        'fg_bbox': fg_bboxes_batch,  # (B, 4)
        'stats': stats_batch
    }

def get_singleturn_query_text_hs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor
):
    is_response_token = (labels != -100)  # Shape: (batch_size, sequence_length)
    first_response_token_indices_all_samples = torch.argmax(is_response_token.int(), dim=1)
    query_indices = torch.max(
        torch.tensor(0, device=labels.device),
        first_response_token_indices_all_samples - 1
    )
    # Gather the hidden states for the selected query tokens
    batch_size_hs = hidden_states.size(0)
    hidden_dim = hidden_states.size(2)
    idx_expanded = query_indices.view(batch_size_hs, 1, 1).expand(batch_size_hs, 1, hidden_dim)
    query_hidden_states = torch.gather(hidden_states, 1, idx_expanded) #
    return query_hidden_states

def get_singleturn_query_text_hs_mheads(
    qk_states: torch.Tensor,
    labels: torch.Tensor
):
    is_response_token = (labels != -100)  # Shape: (batch_size, sequence_length)
    first_response_token_indices_all_samples = torch.argmax(is_response_token.int(), dim=1)
    first_response_token_indices_all_samples = torch.max(
        torch.tensor(0, device=labels.device),
        first_response_token_indices_all_samples - 1
    )
    query_indices = first_response_token_indices_all_samples

    ##Gather the hidden states for the selected query tokens
    batch, num_heads, sequence_length, hidden_dim = qk_states.shape
    idx_expanded = query_indices.view(batch, 1, 1, 1).expand(batch, num_heads, 1, hidden_dim)
    qk_hidden_states = torch.gather(qk_states, 2, idx_expanded)  # Shape: (batch, num_heads, 1, hidden_dim)

    #qk_hidden_states = qk_states[:,:, first_response_token_indices_all_samples-2:first_response_token_indices_all_samples+1, :]
    return qk_hidden_states, query_indices


from qwen_src.ana_utils import get_bbox4src, get_bbox_from_noisy_map, plot_attention_analysis, plot_image_with_heatmaps
import torchvision.transforms.functional as torchvision_F

def interplot_img_feat(sub_img_feat, sub_img_bboxes, max_ratio=2.0):
    """
    Interpolate image features to match the bounding boxes of sub-images.

    This function takes a batch of feature maps and corresponding bounding boxes.
    It expands each bounding box by `max_ratio`, clips it to the feature map
    boundaries, and then uses RoIAlign to extract and resize the feature
    region within the new box back to the original feature map's dimensions.

    Args:
        sub_img_feat (torch.Tensor): Image features of shape (batch_size, num_features, height, width).
                                     If features are flattened (B, C, H*W), they will be reshaped.
        sub_img_bboxes (list of torch.Tensor or torch.Tensor): A list of bounding boxes for each feature map,
                                                               or a single tensor of shape (batch_size, 4).
                                                               Bboxes are in [x1, y1, x2, y2] format.
        max_ratio (float): The factor by which to expand the bounding boxes.

    Returns:
        torch.Tensor: Interpolated image features of the same shape as the input feature map
                      (batch_size, num_features, height, width).
    """
    # --- Input Validation and Reshaping ---
    if sub_img_feat.dim() == 3:
        # Handle flattened features like (B, C, 24*24)
        batch_size, num_features, dim = sub_img_feat.shape
        # Assume a square feature map if it's flattened
        height = width = int(math.sqrt(num_features))
        if height * width != num_features:
            raise ValueError(f"Cannot reshape flattened features of dim {num_features} to a square map.")
        sub_img_feat = sub_img_feat.view(batch_size, height, width, dim).permute(0, 3, 1, 2)
    elif sub_img_feat.dim() != 4:
        raise ValueError(f"sub_img_feat must have 3 or 4 dimensions, but got {sub_img_feat.dim()}")

    batch_size, dim, height, width = sub_img_feat.shape

    # --- Step 1: Process and Validate Bounding Boxes ---
    bboxes = torch.cat(sub_img_bboxes, dim=0) if isinstance(sub_img_bboxes, list) else sub_img_bboxes

    if bboxes.shape[0] != batch_size:
        raise ValueError(f"Number of bboxes ({bboxes.shape[0]}) must match batch size ({batch_size}).")
    if bboxes.dim() != 2 or bboxes.shape[1] != 4:
        raise ValueError("Bboxes tensor must have shape (batch_size, 4).")

    # Ensure bboxes are on the correct device and are float type for calculations
    bboxes = bboxes.to(sub_img_feat.device)

    # --- Process each feature map individually ---
    output_features = []
    for i in range(batch_size):
        # Isolate the i-th feature map and its bbox
        feature_map_i = sub_img_feat[i:i + 1]  # Keep batch dim for interpolate
        bbox_i = bboxes[i]

        # 1. Calculate original bbox dimensions
        bbox_w = bbox_i[2] - bbox_i[0]
        bbox_h = bbox_i[3] - bbox_i[1]

        # 2. Calculate the initial target size by expanding the bbox
        target_w = bbox_w * max_ratio
        target_h = bbox_h * max_ratio

        # 3. Apply the constraint: if expanded size is larger than the base feature map, scale it down.
        longest_target_side = max(target_w, target_h)
        longest_feat_side = max(height, width)

        if longest_target_side > longest_feat_side:
            scale_down_ratio = longest_feat_side / longest_target_side
            target_w *= scale_down_ratio
            target_h *= scale_down_ratio

        # 4. Perform the resizing using interpolation
        # Target size must be integers
        final_target_size = (int(round(target_h.item())), int(round(target_w.item())))

        # Ensure the target size is at least 1x1
        final_target_size = (max(1, final_target_size[0]), max(1, final_target_size[1]))

        # resized_feat = F.interpolate(
        #     feature_map_i,  # Add a temporary batch dimension
        #     size=final_target_size,
        #     mode='bilinear',
        #     align_corners=False
        # ) # 1, C, final_target_size[0], final_target_size[1]
        #
        # # Squeeze the temporary batch dimension before appending
        # resized_feat = resized_feat.squeeze(0).flatten(1).transpose(0, 1)
        # assert resized_feat.shape[0]<=576, f"Resized feature shape {resized_feat.shape} exceeds 576 tokens limit."

        #print('note that interplot func is not work now!!')
        resized_feat = feature_map_i.squeeze(0).flatten(1).transpose(0, 1)
        output_features.append(resized_feat)  # H*W, C

    # --- NEW: Padding Logic ---
    # 1. Find the maximum sequence length in this batch
    max_len = max(feat.shape[0] for feat in output_features) #hard-code to avoid multi-gpu sync problem #max(feat.shape[0] for feat in output_features)

    # 2. Create the padded output tensor and the attention mask
    padded_features = torch.full(
        (batch_size, max_len, dim),
        fill_value=0,
        dtype=sub_img_feat.dtype,
        device=sub_img_feat.device
    )
    padding_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
        device=sub_img_feat.device
    )

    # 3. Copy the data from the unpadded list into the new padded tensor
    for i, feat in enumerate(output_features):
        seq_len = feat.shape[0]
        padded_features[i, :seq_len, :] = feat
        padding_mask[i, :seq_len] = True


    return padded_features, padding_mask


import torch


def get_roi_interpolated_pos_ids(
        src_position_ids,
        surrounding_bbox,
        image_token_start,
        src_grid_thw,
        bbox_grid_thw,
        visual_token_num_src,
        force_long_output=True
):
    """
    Generates interpolated Position IDs for an RoI based on the source image's coordinate system.

    Args:
        src_position_ids: (3, batch_size, seq_len).
        surrounding_bbox: [x_min, y_min, x_max, y_max] indices on the source feature map.
        image_token_start: Index in the sequence where the source image starts.
        src_grid_thw: [T, H, W] of the source image feature map.
        bbox_grid_thw: [T, H, W] of the sub-image (RoI) feature map.
        visual_token_num_src: Number of tokens in the source image.
        force_long_output: If True, rounds the interpolated IDs and casts to torch.long.
                           If False (default), returns torch.float32 for sub-pixel precision.

    Returns:
        roi_position_ids: (3, batch_size, sub_img_seq_len) interpolated position IDs.
    """

    # 1. Setup dimensions
    t_src, h_src, w_src = int(src_grid_thw[0]), int(src_grid_thw[1]//2), int(src_grid_thw[2]//2)
    t_sub, h_sub, w_sub = int(bbox_grid_thw[0]), int(bbox_grid_thw[1]//2), int(bbox_grid_thw[2]//2)

    # 2. Extract the position IDs corresponding ONLY to the source image
    # Input shape is (3, batch_size, seq_len). We explicitly select batch index 0.
    src_img_pos = src_position_ids[:, 0, image_token_start: image_token_start + visual_token_num_src]

    # Safety check for token length
    if src_img_pos.shape[-1] != (t_src * h_src * w_src):
        src_img_pos = src_img_pos[:, :t_src * h_src * w_src]

    # Reshape strictly the spatial dimensions
    src_h_map = src_img_pos[1].view(t_src, h_src, w_src)
    src_w_map = src_img_pos[2].view(t_src, h_src, w_src)

    # 3. Get the boundary values from the source grid using the BBox indices
    x_min, y_min, x_max, y_max = surrounding_bbox[0].int().tolist()

    x_max = min(x_max, w_src)
    y_max = min(y_max, h_src)

    # We select temporal frame 0 for the values
    h_start_val = src_h_map[0, y_min, x_min].item()
    w_start_val = src_w_map[0, y_min, x_min].item()

    h_end_val = src_h_map[0, y_max - 1, x_min].item()
    w_end_val = src_w_map[0, y_min, x_max - 1].item()

    # 4. Interpolate New Grid (Keep as Float for calculation)
    h_steps = torch.linspace(h_start_val, h_end_val, steps=h_sub, device=src_position_ids.device, dtype=torch.float32)
    w_steps = torch.linspace(w_start_val, w_end_val, steps=w_sub, device=src_position_ids.device, dtype=torch.float32)

    # 5. Create Meshgrid and Flatten
    grid_h, grid_w = torch.meshgrid(h_steps, w_steps, indexing='ij')

    flat_h = grid_h.flatten()
    flat_w = grid_w.flatten()

    # 6. Reconstruct the 3D Position ID tensor
    t_val = src_img_pos[0, 0].item()

    # Handle Temporal ID type based on flag request to ensure consistency
    if force_long_output:
        flat_t = torch.full_like(flat_h, fill_value=t_val, dtype=torch.long)
        flat_h = flat_h.round().long()
        flat_w = flat_w.round().long()
    else:
        flat_t = torch.full_like(flat_h, fill_value=t_val, dtype=torch.float32)
        # flat_h and flat_w are already float32

    # Stack to get (3, sub_img_seq_len)
    roi_position_ids = torch.stack([flat_t, flat_h, flat_w], dim=0)

    # 7. Add batch dimension back: (3, 1, sub_img_seq_len)
    roi_position_ids = roi_position_ids.unsqueeze(1)

    return roi_position_ids


from matplotlib import pyplot as plt
def plot_tensor_2d(tensor, title=None, cmap='viridis', figsize=(6, 6), show_colorbar=True):
    """
    Plot a 2D PyTorch tensor as a heatmap.

    Args:
        tensor: 2D PyTorch tensor
        title: Optional title for the plot
        cmap: Colormap to use (default: 'viridis')
        figsize: Figure size as (width, height) in inches
        show_colorbar: Whether to display a colorbar
    """
    # Ensure the tensor is 2D
    if len(tensor.shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}")

    # Convert to numpy for plotting
    if tensor.requires_grad:
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor.cpu().numpy()

    # Create the plot
    plt.figure(figsize=figsize)
    im = plt.imshow(array, cmap=cmap)

    # Add title if provided
    if title:
        plt.title(title)

    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(im)

    plt.tight_layout()
    plt.show()


def blend_image_with_attention(image_np, attention_map, gamma=0.7, min_bright=0.0):
    """
    Blends an image with an attention map by modulating the image's brightness.

    This function implements the 'blend_mask' logic from your provided code.

    Args:
        image_np (np.ndarray): The original image as a NumPy array (H, W, C)
                               in the range [0, 255].
        attention_map (np.ndarray, torch.Tensor): The raw attention map (h, w).
        gamma (float, optional): Gamma correction factor for the attention map.
                                 Defaults to 0.7.
        min_bright (float, optional): The minimum visibility/brightness
                                      factor to apply to the image. Defaults to 0.0.

    Returns:
        np.ndarray: The blended image as a uint8 NumPy array (H, W, C).
    """

    # --- 1. Process and normalize the attention map ---
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.float().detach().cpu().numpy()

    max_val = attention_map.max()
    norm_map = attention_map / max_val if max_val > 0 else np.zeros_like(attention_map)

    # --- 2. Resize map to match image dimensions ---
    H, W = image_np.shape[:2]
    attn_tensor = torch.from_numpy(norm_map).unsqueeze(0).unsqueeze(0)

    # Ensure tensor is float for interpolation
    resized_attn = F.interpolate(
        attn_tensor.float(),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    )

    # --- 3. Apply gamm
    gamma_corrected_attn = torch.pow(resized_attn, gamma).squeeze().cpu().numpy()

    brightness_modulator = min_bright + (1.0 - min_bright) * gamma_corrected_attn

    # Clip and add new axis for broadcasting (H, W) -> (H, W, 1)
    brightness_modulator = np.clip(brightness_modulator, 0.0, 1.0)[:, :, np.newaxis]

    # --- 4. Blend the image ---
    # Convert image to float [0, 1] for multiplication
    img_float = image_np.astype(np.float32) / 255.0

    blended_img_float = img_float * brightness_modulator

    # --- 5. Convert back to uint8 [0, 255] ---
    blended_image_uint8 = (np.clip(blended_img_float, 0.0, 1.0) * 255).astype(np.uint8)

    return blended_image_uint8

from typing import List


def get_batched_sub_images_v2(
        pred_roi_maps: List[torch.Tensor],
        src_imgs: List,
        image_processor,
        image_encoder,
        image_grid_thw,
        conf_thresh=0.15,
        sim2sink_map=None,
        is_debug=False,
        src_image_feats=None,
        src_kv_cache=None,
        is_training=False,
        return_deepstack=False,
        img_llm_patch_size=28,
        add_noise_to_roi=False,
        sink_mask = None,
):
    """
    Args:
        pred_roi_maps: List of Tensors, one per sample. Shape [H_i, W_i] or None.
        src_imgs: List of PIL Images.
        image_grid_thw: Tensor of shape [Batch, 3].
    Returns:
        sub_img_feats_list: List[Tensor], where each item is [N_tokens, Dim] or None.
        sub_img_nums: List[int], number of sub-images per sample (currently 0 or 1).
        bbox_grid_thw_list: List[Tensor], grid shapes for the sub-images.
        roi_mask_list: List[Tensor], flattening boolean masks for the sub-images.
    """
    sub_img_feats_list = []
    sub_img_nums = []
    bbox_grid_thw_list = []
    roi_mask_list = []
    surrounding_bbox_list = []
    sub_img_deepstack_list = []
    bbox_img = None
    batch_size = len(src_imgs)

    for i in range(batch_size):
        pred_map = pred_roi_maps[i]
        image = src_imgs[i]

        # 1. Handle cases where no map exists (e.g., text-only or empty image)
        if pred_map is None:
            sub_img_nums.append(0)
            sub_img_feats_list.append(None)
            bbox_grid_thw_list.append(None)
            roi_mask_list.append(None)
            surrounding_bbox_list.append(None)
            if return_deepstack:
                sub_img_deepstack_list.append(None)
            continue

        # 2. Get Dynamic Feature Size for THIS sample
        # image_grid_thw[i] is [t, h, w]
        feat_h = image_grid_thw[i, 1].item() // 2
        feat_w = image_grid_thw[i, 2].item() // 2
        image_size = (feat_h * img_llm_patch_size, feat_w * img_llm_patch_size)

        # 3. Sink Mask (Suppress top-left corner artifacts)
        if sink_mask is None:
            sink_mask = torch.ones_like(pred_map)
            if pred_map.numel() > 256:
                # Mask out top-left corner (common attention sink artifact)
                h_lim = max(1, pred_map.shape[0] // 4)
                w_lim = max(1, pred_map.shape[1] // 4)
                sink_mask[:h_lim, :1] = 0
                sink_mask[:1, :w_lim] = 0

        # 4. Process RoI Mask
        # Add batch/channel dims for Gaussian blur: [H, W] -> [1, 1, H, W]
        prob_map = (pred_map.sigmoid() * sink_mask).unsqueeze(0).unsqueeze(0)
        blurred_map = torchvision_F.gaussian_blur(prob_map, kernel_size=3, sigma=1.0).squeeze()
        roi_mask_raw = blurred_map
        roi_mask = (blurred_map > conf_thresh).float()

        if roi_mask.sum() == 0:
            sub_img_nums.append(0)
            sub_img_feats_list.append(None)
            bbox_grid_thw_list.append(None)
            roi_mask_list.append(None)
            surrounding_bbox_list.append(None)
            if return_deepstack:
                sub_img_deepstack_list.append(None)
            continue

        # 5. Crop and Encode
        if add_noise_to_roi and random.random()<0.8:
            surrounding_bbox, roi_mask = get_random_roi_bbox_torch(roi_mask, min_ratio=0.1, max_ratio=0.6)
        else:
            surrounding_bbox = get_foreground_bbox_torch(roi_mask.unsqueeze(0), threshold=0.1)

        # Convert feature-grid bbox to pixel coordinates
        bbox_coords = get_bbox4src(surrounding_bbox, image, feat_size=(feat_h, feat_w), img_size=image_size)[0]
        bbox_img = image.crop((int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3])))
        #_save_roi_debug_from_mmutils(image, bbox_img, bbox_coords, i)

        # Crop the mask to match the new bbox
        roi_mask_cropped = roi_mask[surrounding_bbox[0, 1]:surrounding_bbox[0, 3], surrounding_bbox[0, 0]:surrounding_bbox[0, 2]]
        roi_mask_bbox = roi_mask_raw[surrounding_bbox[0, 1]:surrounding_bbox[0, 3], surrounding_bbox[0, 0]:surrounding_bbox[0, 2]]
        roi_mask_bbox_binary = (roi_mask_bbox > 1e-2).float()
        roi_mask_cropped = roi_mask_cropped * roi_mask_bbox_binary

        try:
            # Process sub-image with Qwen2-VL processor
            ori_max_pixel = image_processor.max_pixels
            ori_min_pixel = image_processor.min_pixels

            bbox_w, bbox_h = int(bbox_coords[2] - bbox_coords[0]), int(bbox_coords[3] - bbox_coords[1])
            dst_min_pixel = ori_min_pixel#int(surrounding_bbox[0, 3] - surrounding_bbox[0, 1]) * int(surrounding_bbox[0, 2] - surrounding_bbox[0, 0]) * 28 * 28
            dst_max_pixel = ori_max_pixel#int(surrounding_bbox[0, 3] - surrounding_bbox[0, 1]) * int(surrounding_bbox[0, 2] - surrounding_bbox[0, 0]) * 28 * 28 * 2
            stime = time.time()
            bbox_img_pixels = image_processor([bbox_img], max_pixels=dst_max_pixel, min_pixels=dst_min_pixel, return_tensors="pt")
            #print("Time for processing sub-image:", time.time() - stime)
            # Restore original settings (important for shared processor)
            image_processor.max_pixels = ori_max_pixel
            image_processor.min_pixels = ori_min_pixel

        except ValueError as e:
            print(f"Error processing sub-image batch {i}: {e}")
            sub_img_nums.append(0)
            sub_img_feats_list.append(None)
            bbox_grid_thw_list.append(None)
            roi_mask_list.append(None)
            surrounding_bbox_list.append(None)
            if return_deepstack:
                sub_img_deepstack_list.append(None)
            continue

        # 6. Extract Features
        pixel_values = bbox_img_pixels.data['pixel_values'].type_as(pred_map)
        bbox_grid_thw = bbox_img_pixels.data['image_grid_thw'].type_as(image_grid_thw)

        # Encode
        stime = time.time()
        encoded_output = image_encoder(pixel_values, grid_thw=bbox_grid_thw,)
        if isinstance(encoded_output, (tuple, list)) and len(encoded_output) >= 2:
            img_feats, deepstack_feats = encoded_output[0], encoded_output[1]
        else:
            img_feats, deepstack_feats = encoded_output, None
        #print("Time for encoding sub-image:", time.time() - stime)


        # 7. Interpolate Mask to match Sub-Image Grid
        # Sub-image grid size might be different from crop size due to resizing
        sub_img_h = bbox_grid_thw[-1, 1] // 2
        sub_img_w = bbox_grid_thw[-1, 2] // 2

        roi_mask_resized = F.interpolate(
            roi_mask_cropped.unsqueeze(0).unsqueeze(0),
            size=(sub_img_h, sub_img_w),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        roi_mask_final = (roi_mask_resized > 0.5).float()

        # Append results
        sub_img_nums.append(1)
        sub_img_feats_list.append(img_feats)
        bbox_grid_thw_list.append(bbox_grid_thw)
        roi_mask_list.append(roi_mask_final.flatten())
        surrounding_bbox_list.append(surrounding_bbox)
        if return_deepstack:
            sub_img_deepstack_list.append(deepstack_feats)

        if is_debug:
            blurred_map = torchvision_F.gaussian_blur((pred_map.sigmoid() * sink_mask).unsqueeze(0).unsqueeze(0),
                                                      kernel_size=3, sigma=1.0).squeeze()
            maps_to_plot = [
                {
                    'map': (pred_map.sigmoid() * sink_mask),
                    'title': 'Raw Heatmap',
                    'blend': False,
                    'cmap': 'viridis'
                },
                {
                    'map': blurred_map,
                    'title': 'Blurred Heatmap',
                    'blend': False,  # This will be a standard heatmap
                    'cmap': 'viridis'  # Optional: specify a colormap
                },
                {
                    'map': blurred_map > conf_thresh,
                    'title': 'Blended Mask (Thresholded)',
                    'blend': True,  # This will be an overlay on the image
                    'min_bright': 0.3,
                }
            ]
            time_stamp = time.strftime("%Y%m%d-%H%M%S")
            plot_image_with_heatmaps(
                image=np.array(src_imgs[i]),
                attention_maps=maps_to_plot,
                dpi=300,
                # save_path=f'./debug_images/{time_stamp}.png'
            )

    if return_deepstack:
        return (
            sub_img_feats_list,
            sub_img_nums,
            bbox_grid_thw_list,
            roi_mask_list,
            surrounding_bbox_list,
            sub_img_deepstack_list,
            bbox_img
        )
    return sub_img_feats_list, sub_img_nums, bbox_grid_thw_list, roi_mask_list, surrounding_bbox_list



from torch.nn.utils.rnn import pad_sequence
def insert_sub_feat_v2(ori_feat, sub_feat_list, sub_img_nums, visual_token_counts, sys_token_num,
                       new_labels, attention_mask, roi_mask_list, input_ids,
                       # --- Pos ID Args ---
                       position_ids=None,
                       pos_id_fn=None,
                       image_grid_thw=None,
                       bbox_grid_thw_list=None,
                       surrounding_bbox_list=None,
                       # -------------------
                       reuse_src_pos=False,
                       mask_roi_bg=False,
                       # --- Layout Flag ---
                       nested_roi_layout=False,
                       insert_after_text=False,
                       # --- DeepStack ---
                       deepstack_visual_embeds=None,
                       sub_img_deepstack_list=None,
                       image_token_id=None,
                       video_token_id=None,
                       return_deepstack=False,
                       ):
    batch_size = ori_feat.shape[0]

    dst_feat_list = []
    valid_token_mask_updated = []
    input_ids_updated_list = []
    new_attention_mask_list = []
    inserted_content_mask_list = []
    position_ids_updated_list = []
    visual_pos_mask_list = []
    deepstack_updated_per_layer = None

    # Detect im_start token ID from the first token (Standard Qwen2.5/3 is 151644)
    im_start_token_id = input_ids[0, 0]

    deepstack_per_layer_samples = []
    if deepstack_visual_embeds is not None:
        for layer_idx, layer_embeds in enumerate(deepstack_visual_embeds):
            if layer_embeds is None:
                deepstack_per_layer_samples.append(None)
                continue
            deepstack_per_layer_samples.append(torch.split(layer_embeds, visual_token_counts.tolist()))
        deepstack_updated_per_layer = [[] for _ in range(len(deepstack_visual_embeds))]

    reuse_flag = 0

    for i in range(batch_size):
        ori_feat_i = ori_feat[i]
        pos_ids_i = position_ids[:, i, :] if position_ids is not None else None
        current_vis_tokens = visual_token_counts[i].item()

        # --- Case 1: No sub-image ---
        if sub_img_nums[i] == 0 or sub_feat_list[i] is None:
            dst_feat_list.append(ori_feat_i)
            input_ids_updated_list.append(input_ids[i])
            valid_mask_i = torch.ones(ori_feat_i.shape[0], dtype=torch.bool, device=ori_feat.device)
            valid_token_mask_updated.append(valid_mask_i)
            new_attention_mask_list.append(attention_mask[i])
            inserted_content_mask_list.append(
                torch.zeros(ori_feat_i.shape[0], dtype=torch.bool, device=ori_feat.device))
            if pos_ids_i is not None:
                position_ids_updated_list.append(pos_ids_i)
            if image_token_id is not None and video_token_id is not None:
                visual_mask = (input_ids[i] == image_token_id) | (input_ids[i] == video_token_id)
                visual_pos_mask_list.append(visual_mask)
            if deepstack_updated_per_layer is not None:
                for layer_idx, layer_samples in enumerate(deepstack_per_layer_samples):
                    if layer_samples is None:
                        deepstack_updated_per_layer[layer_idx].append(None)
                    else:
                        deepstack_updated_per_layer[layer_idx].append(layer_samples[i])
            continue

        # --- Case 2: Sub-image exists ---
        sub_feat_i = sub_feat_list[i]
        roi_mask_i = roi_mask_list[i].bool()

        start_idx = sys_token_num
        end_idx = sys_token_num + current_vis_tokens

        img_start_token_feat = ori_feat_i[start_idx - 1: start_idx - reuse_flag]
        img_end_token_feat = ori_feat_i[end_idx: end_idx + 1 - reuse_flag]

        img_start_token_id = input_ids[i, start_idx - 1: start_idx - reuse_flag]
        img_end_token_id = input_ids[i, end_idx: end_idx + 1 - reuse_flag]

        if pos_ids_i is not None and reuse_src_pos:
            img_start_pos = pos_ids_i[:, start_idx - 1: start_idx - reuse_flag]
            img_end_pos = pos_ids_i[:, end_idx: end_idx + 1 - reuse_flag]

        # ---------------------------------------------------------
        # A. Calculate Split Points for Suffix (Text)
        # ---------------------------------------------------------
        if nested_roi_layout:
            suffix_start_idx = end_idx
        else:
            suffix_start_idx = end_idx + 1

        suffix_ids = input_ids[i, suffix_start_idx:]
        im_start_indices = (suffix_ids == im_start_token_id).nonzero(as_tuple=True)[0]

        if len(im_start_indices) > 0:
            # === CRITICAL LOGIC ===
            # We use [-1] (The Last Occurrence) to split the User's text from the Assistant's start.
            # This puts the insertion point AFTER the User's <|im_end|> and BEFORE <|im_start|>assistant.
            rel_split_idx = im_start_indices[-1].item()
            abs_split_idx = suffix_start_idx + rel_split_idx #-2
        else:
            abs_split_idx = len(input_ids[i])

        prefix_slice = slice(None, suffix_start_idx)  # Up to Source Image Tokens
        suffix_pre_slice = slice(suffix_start_idx, abs_split_idx)  # Source + User Prompt
        suffix_post_slice = slice(abs_split_idx, None)  # <|im_start|>assistant ...

        # ---------------------------------------------------------
        # B. Prepare Sequence Buckets
        # ---------------------------------------------------------
        prefix_parts = {'feat': [], 'ids': [], 'valid': [], 'attn': [], 'inserted': [], 'pos': []}
        suffix_pre_parts = {'feat': [], 'ids': [], 'valid': [], 'attn': [], 'inserted': [], 'pos': []}
        roi_parts = {'feat': [], 'ids': [], 'valid': [], 'attn': [], 'inserted': [], 'pos': []}
        suffix_post_parts = {'feat': [], 'ids': [], 'valid': [], 'attn': [], 'inserted': [], 'pos': []}

        def extract_part(part_dict, slc):
            if slc.start == slc.stop and slc.start is not None: return
            part_dict['feat'].append(ori_feat_i[slc])
            part_dict['ids'].append(input_ids[i, slc])
            length = len(part_dict['feat'][-1])
            part_dict['valid'].append(torch.ones(length, dtype=torch.bool, device=ori_feat.device))
            part_dict['attn'].append(attention_mask[i, slc])
            part_dict['inserted'].append(torch.zeros(length, dtype=torch.bool, device=ori_feat.device))
            if pos_ids_i is not None and reuse_src_pos:
                part_dict['pos'].append(pos_ids_i[:, slc])

        extract_part(prefix_parts, prefix_slice)
        extract_part(suffix_pre_parts, suffix_pre_slice)
        extract_part(suffix_post_parts, suffix_post_slice)

        # --- ROI Construction (Blocks) ---
        grids = bbox_grid_thw_list[i]
        num_blocks = grids.shape[0]
        current_feat_offset = 0
        loop_range = range(1) if reuse_src_pos else range(num_blocks)

        for blk_idx in loop_range:
            t, h_grid, w_grid = grids[blk_idx].tolist()
            blk_len = t * (h_grid // 2) * (w_grid // 2)
            feat_slice = sub_feat_i[current_feat_offset: current_feat_offset + blk_len]
            mask_slice = roi_mask_i[current_feat_offset: current_feat_offset + blk_len]
            current_feat_offset += blk_len

            # Start Token
            roi_parts['feat'].append(img_start_token_feat)
            roi_parts['ids'].append(img_start_token_id)
            roi_parts['valid'].append(torch.ones(len(img_start_token_feat), dtype=torch.bool, device=ori_feat.device))
            roi_parts['attn'].append(
                torch.ones(len(img_start_token_feat), dtype=attention_mask.dtype, device=attention_mask.device))
            roi_parts['inserted'].append(
                torch.ones(len(img_start_token_feat), dtype=torch.bool, device=ori_feat.device))
            if reuse_src_pos and pos_ids_i is not None:
                roi_parts['pos'].append(img_end_pos + 1)

            # Features
            roi_parts['feat'].append(feat_slice)
            slice_ids = input_ids[i, start_idx].unsqueeze(0).repeat(blk_len)
            roi_parts['ids'].append(slice_ids)
            roi_parts['valid'].append(mask_slice)
            roi_parts['attn'].append(torch.ones(blk_len, dtype=attention_mask.dtype, device=attention_mask.device))
            roi_parts['inserted'].append(torch.ones(blk_len, dtype=torch.bool, device=ori_feat.device))
            if reuse_src_pos and pos_ids_i is not None:
                roi_pos_interpolated = get_roi_interpolated_pos_ids_single(
                    src_pos_ids_i=pos_ids_i,
                    surrounding_bbox=surrounding_bbox_list[i],
                    image_token_start=sys_token_num,
                    src_grid_thw=image_grid_thw[i],
                    bbox_grid_thw=bbox_grid_thw_list[i],
                    visual_token_num_src=current_vis_tokens
                )
                roi_parts['pos'].append(roi_pos_interpolated)

            # End Token
            roi_parts['feat'].append(img_end_token_feat)
            roi_parts['ids'].append(img_end_token_id)
            roi_parts['valid'].append(torch.ones(len(img_end_token_feat), dtype=torch.bool, device=ori_feat.device))
            roi_parts['attn'].append(torch.ones(len(img_end_token_feat), dtype=attention_mask.dtype, device=attention_mask.device))
            roi_parts['inserted'].append(torch.ones(len(img_end_token_feat), dtype=torch.bool, device=ori_feat.device))
            if reuse_src_pos and pos_ids_i is not None:
                roi_parts['pos'].append(img_end_pos + (bbox_grid_thw_list[i][0, 1:] // 2).max().item() + 2)

        # ---------------------------------------------------------
        # C. Construct Final Sequence & Handle Pos ID Shifts
        # ---------------------------------------------------------
        parts_feat = []
        parts_ids = []
        parts_valid_mask = []
        parts_attn_mask = []
        parts_inserted_mask = []
        parts_pos_ids = []

        # Calculate Offset for elements appearing AFTER the ROI
        roi_len_offset = (bbox_grid_thw_list[i][0, 1:] // 2).max().item() + 2

        def append_bucket(bucket, pos_shift=0):
            parts_feat.extend(bucket['feat'])
            parts_ids.extend(bucket['ids'])
            parts_valid_mask.extend(bucket['valid'])
            parts_attn_mask.extend(bucket['attn'])
            parts_inserted_mask.extend(bucket['inserted'])
            if len(bucket['pos']) > 0:
                shifted_pos = [p + pos_shift for p in bucket['pos']]
                parts_pos_ids.extend(shifted_pos)

        # --- ASSEMBLY LINE ---
        append_bucket(prefix_parts, pos_shift=0)  # Always first

        if insert_after_text:
            # 1. User Text (No shift, connected to Source)
            append_bucket(suffix_pre_parts, pos_shift=0)
            # 2. ROI (No shift, uses reused positions)
            append_bucket(roi_parts, pos_shift=0)
            # 3. Assistant (Shifted, because it comes AFTER the inserted ROI in the sequence)
            append_bucket(suffix_post_parts, pos_shift=roi_len_offset)
        else:
            # 1. ROI (No shift)
            append_bucket(roi_parts, pos_shift=0)
            # 2. User Text (Shifted)
            append_bucket(suffix_pre_parts, pos_shift=roi_len_offset)
            # 3. Assistant (Shifted)
            append_bucket(suffix_post_parts, pos_shift=roi_len_offset)

        # --- Concatenate ---
        raw_feat = torch.cat(parts_feat, dim=0)
        raw_ids = torch.cat(parts_ids, dim=0)
        raw_valid_mask = torch.cat(parts_valid_mask, dim=0)
        raw_attn_mask = torch.cat(parts_attn_mask, dim=0)
        raw_inserted_mask = torch.cat(parts_inserted_mask, dim=0)

        # ---------------------------------------------------------
        # D. Pos ID Generation (If not reusing)
        # ---------------------------------------------------------
        raw_pos_ids = None
        if reuse_src_pos and len(parts_pos_ids) > 0:
            raw_pos_ids = torch.cat(parts_pos_ids, dim=1)
        elif pos_id_fn is not None:
            # Standard Generation for full sequence
            sample_grids = [image_grid_thw[i].unsqueeze(0) if image_grid_thw[i].dim() == 1 else image_grid_thw[i]]
            if bbox_grid_thw_list[i] is not None:
                sample_grids.append(bbox_grid_thw_list[i])
            sample_grids_thw = torch.cat(sample_grids, dim=0)

            gen_pos, _ = pos_id_fn(
                input_ids=raw_ids.unsqueeze(0),
                image_grid_thw=sample_grids_thw,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=torch.ones_like(raw_ids.unsqueeze(0))
            )
            raw_pos_ids = gen_pos[:, 0, :]

        # ---------------------------------------------------------
        # E. Final Output Construction
        # ---------------------------------------------------------
        if not mask_roi_bg:
            if raw_valid_mask.shape[0] != raw_feat.shape[0]:
                raw_valid_mask = torch.ones_like(raw_feat[:, 0], dtype=torch.bool, device=ori_feat.device)
            else:
                raw_valid_mask = torch.ones_like(raw_valid_mask, dtype=torch.bool, device=ori_feat.device)

        dst_feat_list.append(raw_feat[raw_valid_mask])
        input_ids_updated_list.append(raw_ids[raw_valid_mask])
        new_attention_mask_list.append(raw_attn_mask[raw_valid_mask])
        inserted_content_mask_list.append(raw_inserted_mask[raw_valid_mask])
        valid_token_mask_updated.append(raw_valid_mask[raw_valid_mask])

        if raw_pos_ids is not None:
            position_ids_updated_list.append(raw_pos_ids[:, raw_valid_mask])

        if image_token_id is not None and video_token_id is not None:
            final_ids = raw_ids[raw_valid_mask]
            visual_mask = (final_ids == image_token_id) | (final_ids == video_token_id)
            visual_pos_mask_list.append(visual_mask)

        # Handle DeepStack alignment
        if deepstack_updated_per_layer is not None:
            for layer_idx, layer_samples in enumerate(deepstack_per_layer_samples):
                if layer_samples is None:
                    deepstack_updated_per_layer[layer_idx].append(None)
                    continue

                src_layer_embeds = layer_samples[i]
                roi_layer_embeds = None
                if sub_img_deepstack_list is not None and sub_img_deepstack_list[i] is not None:
                    if layer_idx < len(sub_img_deepstack_list[i]):
                        roi_layer_embeds = sub_img_deepstack_list[i][layer_idx]

                if roi_layer_embeds is None:
                    merged_visual_embeds = src_layer_embeds
                else:
                    roi_layer_embeds = roi_layer_embeds.to(src_layer_embeds.device)
                    if mask_roi_bg:
                        roi_layer_embeds = roi_layer_embeds[roi_mask_i]

                    # DeepStack usually only applies to VISUAL tokens.
                    # We concat the source and ROI visual embeddings:
                    # [Source Visuals] + [ROI Visuals]
                    merged_visual_embeds = torch.cat([src_layer_embeds, roi_layer_embeds], dim=0)

                deepstack_updated_per_layer[layer_idx].append(merged_visual_embeds)

    # ---------------------------------------------------------
    # F. Pack
    # ---------------------------------------------------------
    new_input_embeds = pad_sequence(dst_feat_list, batch_first=True, padding_value=0.0)
    input_ids_final = pad_sequence(input_ids_updated_list, batch_first=True, padding_value=0)
    new_attention_mask = pad_sequence(new_attention_mask_list, batch_first=True, padding_value=0)
    inserted_content_mask = pad_sequence(inserted_content_mask_list, batch_first=True, padding_value=False)
    valid_token_mask_final = pad_sequence(valid_token_mask_updated, batch_first=True, padding_value=False)

    new_position_ids = None
    if len(position_ids_updated_list) > 0:
        pos_T = [p.transpose(0, 1) for p in position_ids_updated_list]
        padded_pos = pad_sequence(pos_T, batch_first=True, padding_value=0)
        new_position_ids = padded_pos.permute(2, 0, 1)

    new_visual_pos_mask = None
    if len(visual_pos_mask_list) > 0:
        new_visual_pos_mask = pad_sequence(visual_pos_mask_list, batch_first=True, padding_value=False)

    updated_deepstack_embeds = None
    if deepstack_updated_per_layer is not None:
        updated_deepstack_embeds = []
        for layer_idx, layer_samples in enumerate(deepstack_updated_per_layer):
            if layer_samples is None or any(sample is None for sample in layer_samples):
                updated_deepstack_embeds.append(None)
                continue
            updated_deepstack_embeds.append(torch.cat(layer_samples, dim=0))

    if return_deepstack:
        return (
            new_input_embeds,
            new_attention_mask,
            valid_token_mask_final,
            input_ids_final,
            inserted_content_mask,
            new_position_ids,
            new_visual_pos_mask,
            updated_deepstack_embeds,
        )
    return new_input_embeds, new_attention_mask, valid_token_mask_final, input_ids_final, inserted_content_mask, new_position_ids

def get_roi_interpolated_pos_ids_single(
        src_pos_ids_i, # Shape: (3, Seq_Len) - Single sample, no batch dim
        surrounding_bbox,
        image_token_start,
        src_grid_thw,
        bbox_grid_thw,
        visual_token_num_src,
        force_long_output=False
):
    # 1. Setup dimensions
    t_src, h_src, w_src = int(src_grid_thw[0]), int(src_grid_thw[1]//2), int(src_grid_thw[2]//2)
    t_sub, h_sub, w_sub = int(bbox_grid_thw[0, 0]), int(bbox_grid_thw[0, 1]//2), int(bbox_grid_thw[0, 2]//2)

    # 2. Extract the position IDs corresponding ONLY to the source image
    src_img_pos = src_pos_ids_i[:, image_token_start: image_token_start + visual_token_num_src]

    # Safety check for token length
    if src_img_pos.shape[-1] != (t_src * h_src * w_src):
        src_img_pos = src_img_pos[:, :t_src * h_src * w_src]

    # Reshape strictly the spatial dimensions
    src_h_map = src_img_pos[1].view(t_src, h_src, w_src)
    src_w_map = src_img_pos[2].view(t_src, h_src, w_src)

    # 3. Get the boundary values from the source grid
    x_min, y_min, x_max, y_max = surrounding_bbox[0].int().tolist()
    x_max = min(x_max, w_src)
    y_max = min(y_max, h_src)

    t_val = src_img_pos[0, 0].item()
    t_val_roi = t_val + max(h_src, w_src) + 2 ##

    # We select temporal frame 0 for the values
    h_start_val = src_h_map[0, y_min, x_min].item()
    w_start_val = src_w_map[0, y_min, x_min].item()
    h_end_val = src_h_map[0, y_max - 1, x_min].item() #h_start_val + h_sub - 1##src_h_map[0, y_max - 1, x_min].item()
    w_end_val = src_w_map[0, y_min, x_max - 1].item() #w_start_val + w_sub - 1##src_w_map[0, y_min, x_max - 1].item()

    # 4. Interpolate New Grid
    h_steps = torch.linspace(h_start_val, h_end_val, steps=h_sub, device=src_pos_ids_i.device, dtype=torch.float32)
    w_steps = torch.linspace(w_start_val, w_end_val, steps=w_sub, device=src_pos_ids_i.device, dtype=torch.float32)

    # 5. Create Meshgrid and Flatten
    grid_h, grid_w = torch.meshgrid(h_steps, w_steps, indexing='ij')
    flat_h = grid_h.flatten()
    flat_w = grid_w.flatten()

    # 6. Reconstruct the 3D Position ID tensor
    if force_long_output:
        flat_t = torch.full_like(flat_h, fill_value=t_val_roi, dtype=torch.long)
        flat_h = flat_h.round().long()
        flat_w = flat_w.round().long()
    else:
        flat_t = torch.full_like(flat_h, fill_value=t_val_roi, dtype=torch.float32)

    # Stack to get (3, sub_img_seq_len)
    roi_position_ids = torch.stack([flat_t, flat_h, flat_w], dim=0)

    return roi_position_ids

def update_batched_labels(original_labels, inserted_content_mask, ignore_index=-100):
    """
    Updates labels by inserting ignore_index where inserted_content_mask is True.

    Args:
        original_labels: [Batch, Orig_Seq_Len]
        inserted_content_mask: [Batch, New_Seq_Len] Boolean mask from insert_sub_feat_v2
                               (True = new content, False = original content/padding)
        ignore_index: int, default -100

    Returns:
        new_labels: [Batch, New_Seq_Len]
    """
    batch_size = inserted_content_mask.shape[0]
    new_seq_len = inserted_content_mask.shape[1]

    # Initialize new labels with ignore_index
    new_labels = torch.full((batch_size, new_seq_len), ignore_index,
                            dtype=original_labels.dtype, device=original_labels.device)

    for i in range(batch_size):
        # 1. Identify slots for original labels
        # These are positions where we did NOT insert new content
        # Note: This includes the original padding if it existed.
        non_inserted_indices = torch.nonzero(~inserted_content_mask[i], as_tuple=True)[0]

        # 2. Extract valid original labels
        # We need to ensure lengths match.
        # The number of 'False' in mask should theoretically match len(original_labels[i])
        # IF no deletion happened.

        # However, 'inserted_content_mask' was padded to New_Seq_Len.
        # 'original_labels' might also be padded.
        # We simply fill the "False" slots in the new sequence with the tokens from the old sequence.

        valid_len = min(len(non_inserted_indices), len(original_labels[i]))

        if valid_len > 0:
            # Map original labels to the non-inserted positions
            new_labels[i, non_inserted_indices[:valid_len]] = original_labels[i, :valid_len]

    return new_labels


from PIL import Image, ImageDraw

def draw_dashed_line(draw, start, end, fill='red', width=2, dash_length=10, space_length=5):
    """
    Helper to draw a dashed line on a PIL ImageDraw object.
    """
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx ** 2 + dy ** 2)

    if dist == 0:
        return

    # Normalize direction
    ux = dx / dist
    uy = dy / dist

    curr_dist = 0
    while curr_dist < dist:
        # Start of dash
        x_s = x1 + ux * curr_dist
        y_s = y1 + uy * curr_dist

        # End of dash
        end_dist = min(curr_dist + dash_length, dist)
        x_e = x1 + ux * end_dist
        y_e = y1 + uy * end_dist

        draw.line([(x_s, y_s), (x_e, y_e)], fill=fill, width=width)
        curr_dist += dash_length + space_length


def draw_bbox_on_image(image, bbox_coords, color='red', width=3):
    """
    Helper function responsible ONLY for drawing the bounding box on a given image.

    Args:
        image: PIL Image object.
        bbox_coords: Tuple/List (x_min, y_min, x_max, y_max).
        color: Box color.
        width: Line width.

    Returns:
        annotated_img: A copy of the image with the bbox drawn.
    """
    # Create a copy to avoid modifying the original image in place
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    x_min, y_min, x_max, y_max = map(int, bbox_coords)

    # PIL rectangle includes x_max/y_max, so we draw strictly
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

    return annotated_img


def create_joint_image_fixed_size(
        src_img,
        bbox_coords,
        src_target_hw,
        roi_target_hw,
        gap=20,
        color='red',
        width=3
):
    """
    Creates a 'zoom-in' visualization joint image with EXPLICIT target sizes.

    Args:
        src_img: PIL Image (Raw Source).
        bbox_coords: Tuple (x_min, y_min, x_max, y_max) in RAW coordinates.
        src_target_hw: Tuple (height, width) for the resized source image.
        roi_target_hw: Tuple (height, width) for the resized RoI image.
        gap: Int, pixels between source and RoI.
        color: Str, color of annotations.
        width: Int, stroke width.

    Returns:
        joint_img: PIL Image, the combined visualization.
    """
    # 1. Unpack Dimensions
    src_h_tgt, src_w_tgt = src_target_hw
    roi_h_tgt, roi_w_tgt = roi_target_hw

    raw_w_src, raw_h_src = src_img.size

    # 2. Resize Source Image to Target Size
    # We use BICUBIC for high quality downsampling/upsampling
    src_img_resized = src_img.resize((src_w_tgt, src_h_tgt), resample=Image.BICUBIC)

    # 3. Handle RoI Crop & Resize
    x_min, y_min, x_max, y_max = map(int, bbox_coords)

    # Ensure bounds on raw image
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(raw_w_src, x_max), min(raw_h_src, y_max)

    if x_max <= x_min or y_max <= y_min:
        # Fallback: Just return resized source if bbox is invalid
        return src_img_resized

    # Crop RoI from RAW image first (to preserve maximum detail)
    roi_img_raw = src_img.crop((x_min, y_min, x_max, y_max))

    # Resize RoI to Target Size
    roi_img_resized = roi_img_raw.resize((roi_w_tgt, roi_h_tgt), resample=Image.BICUBIC)

    # 4. Create Canvas
    total_w = src_w_tgt + gap + roi_w_tgt
    total_h = max(src_h_tgt, roi_h_tgt)

    joint_img = Image.new('RGB', (total_w, total_h), (255, 255, 255))

    # 5. Paste Images (Center vertically)
    src_y_offset = (total_h - src_h_tgt) // 2
    roi_y_offset = (total_h - roi_h_tgt) // 2
    roi_x_offset = src_w_tgt + gap

    joint_img.paste(src_img_resized, (0, src_y_offset))
    joint_img.paste(roi_img_resized, (roi_x_offset, roi_y_offset))

    # 6. Draw Annotations
    draw = ImageDraw.Draw(joint_img)

    # A. Scale BBox Coordinates for the Resized Source Image
    # Formula: coord_resized = coord_raw * (size_resized / size_raw)
    scale_x = src_w_tgt / raw_w_src
    scale_y = src_h_tgt / raw_h_src

    box_x0 = int(x_min * scale_x)
    box_y0 = int(y_min * scale_y) + src_y_offset
    box_x1 = int(x_max * scale_x)
    box_y1 = int(y_max * scale_y) + src_y_offset

    # Draw BBox on Source
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], outline=color, width=width)

    # B. Draw Border around RoI
    draw.rectangle(
        [roi_x_offset, roi_y_offset, roi_x_offset + roi_w_tgt, roi_y_offset + roi_h_tgt],
        outline=color, width=width
    )

    # C. Draw Connecting Lines
    # Top Line: Top-Right of SrcBox -> Top-Left of RoI
    draw_dashed_line(
        draw,
        (box_x1, box_y0),
        (roi_x_offset, roi_y_offset),
        fill=color, width=width
    )

    # Bottom Line: Bottom-Right of SrcBox -> Bottom-Left of RoI
    draw_dashed_line(
        draw,
        (box_x1, box_y1),
        (roi_x_offset, roi_y_offset + roi_h_tgt),
        fill=color, width=width
    )

    return joint_img

def verify_joint_image(joint_img, batch_idx=0, bbox_coords=None):
    """
    Plots the joint image for visual verification.
    """
    plt.figure(figsize=(15, 10))
    plt.imshow(joint_img)

    title_text = f"Batch {batch_idx}: Joint Image Verification"
    if bbox_coords is not None:
        title_text += f"\nBBox: {list(map(int, bbox_coords))}"

    plt.title(title_text)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


import torch


def create_roi_causal_mask(
        batch_size, seq_len,
        attention_mask,  # Original 1D [Batch, Seq]
        input_ids,
        image_grid_thw,  # [Batch, 3] tensor
        bbox_grid_thw_list,  # List of tensors
        bbox_list,  # List of tensors [x1, y1, x2, y2]
        image_token_id,
        mask_redundant_src=True  # <--- NEW FLAG
):
    device = attention_mask.device

    # 1. Base Causal Mask [Batch, 1, Seq, Seq]
    # Start with standard causal (lower triangular)
    causal = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).clone()

    # Apply padding
    padding = attention_mask.bool().view(batch_size, 1, 1, seq_len)
    mask = mask & padding

    for i in range(batch_size):
        if bbox_list[i] is None or bbox_grid_thw_list[i] is None:
            continue

        # --- A. Locate Source Image ---
        img_indices = torch.where(input_ids[i] == image_token_id)[0]
        if len(img_indices) == 0: continue

        src_start = img_indices[0].item()
        t_src, h_src, w_src = image_grid_thw[i][0].item(), image_grid_thw[i][1].item() // 2, image_grid_thw[i][
            2].item() // 2
        src_len = t_src * h_src * w_src
        src_end = src_start + src_len

        # --- B. Locate RoI Image ---
        # Find start of second image block
        roi_candidates = img_indices[img_indices >= src_end]
        if len(roi_candidates) > 0:
            roi_start = roi_candidates[0].item()
        else:
            roi_start = src_end

        t_roi = bbox_grid_thw_list[i][0, 0].item()
        h_roi = bbox_grid_thw_list[i][0, 1].item() // 2
        w_roi = bbox_grid_thw_list[i][0, 2].item() // 2
        roi_end = roi_start + (t_roi * h_roi * w_roi)

        if roi_end > seq_len: roi_end = seq_len

        # --- C. Calculate "Max Visible Source Index" for each RoI Token ---
        x1, y1, x2, y2 = bbox_list[i][0].float().tolist()

        h_steps = torch.linspace(y1, y2, steps=h_roi + 1, device=device)[:-1]
        w_steps = torch.linspace(x1, x2, steps=w_roi + 1, device=device)[:-1]

        grid_h, grid_w = torch.meshgrid(h_steps, w_steps, indexing='ij')

        flat_h = grid_h.flatten().long().clamp(0, h_src - 1)
        flat_w = grid_w.flatten().long().clamp(0, w_src - 1)

        mapped_src_indices_relative = flat_h * w_src + flat_w
        mapped_src_indices_absolute = src_start + mapped_src_indices_relative

        # Ranges and Limits
        roi_range_len = roi_end - roi_start
        valid_len = min(roi_range_len, mapped_src_indices_absolute.shape[0])
        limits = mapped_src_indices_absolute[:valid_len]  # [N_roi] - Shows which src token corresponds to each RoI token

        src_indices = torch.arange(src_start, src_end, device=device)  # [N_src]

        # --- D. Apply Bi-Directional Visibility (Existing Logic) ---
        if valid_len > 0:
            limits_vec = limits.unsqueeze(1)  # [N_roi, 1]
            src_vec = src_indices.unsqueeze(0)  # [1, N_src]

            # 1. RoI sees Source (Lower Left)
            block_mask_roi_sees_src = src_vec <= limits_vec
            mask[i, 0, roi_start:roi_start + valid_len, src_start:src_end] = block_mask_roi_sees_src

            # 2. Source sees RoI (Upper Right)
            src_vec_2 = src_indices.unsqueeze(1)  # [N_src, 1]
            limits_vec_2 = limits.unsqueeze(0)  # [1, N_roi]
            block_mask_src_sees_roi = src_vec_2 > limits_vec_2
            mask[i, 0, src_start:src_end, roi_start:roi_start + valid_len] = block_mask_src_sees_roi

        # --- E. Redundancy Removal (NEW LOGIC) ---
        if mask_redundant_src and valid_len > 0:
            # We want to mask the source tokens that are covered by the RoI.
            # However, RoI tokens map to Source tokens "Many-to-One" (since RoI is larger/zoomed).
            # We need to find the SET of source indices that are mapped to by ANY RoI token.

            # Find unique Source indices covered by the RoI
            covered_src_indices = torch.unique(limits)  # [N_covered]

            if len(covered_src_indices) > 0:
                # 2. Determine where the "Covered Region" ends
                # We assume standard raster order, so max index is the end of the block
                last_covered_idx = torch.max(covered_src_indices).item()

                # 3. Define the "Blind Zone"
                # ANY token coming after the covered region (spatially or temporally)
                # should NOT see the covered source tokens.
                # This includes:
                #   a. Late Source tokens (Src_Late)
                #   b. RoI tokens (if they come later)
                #   c. Text/Prompt tokens
                rows_after_coverage = torch.arange(last_covered_idx + 1, seq_len, device=device)
                ##rows_after_coverage = torch.arange(roi_start, roi_start + valid_len, device=device)

                # 4. Apply the Mask
                # We mask the 'covered_src_indices' columns for all 'rows_after_coverage'
                # Expand for broadcasting or meshgrid

                # Check bounds to be safe
                if len(rows_after_coverage) > 0:
                    # Create grid: Rows = Post-Coverage, Cols = Covered Source
                    row_grid, col_grid = torch.meshgrid(rows_after_coverage, covered_src_indices, indexing='ij')

                    # Set visibility to False (Masked)
                    mask[i, 0, row_grid, col_grid] = False

    #plot_attention_mask(mask, batch_idx=0, title="Updated Causal Attention Mask with RoI")
    return mask

def extract_visual_reps_from_hidden_states(
        hidden_states,  # [Batch, Seq_Len, Dim]
        input_ids,  # [Batch, Seq_Len]
        inserted_content_mask,  # [Batch, Seq_Len] (True for RoI block)
        image_token_id
):
    """
    Separates Source and RoI visual tokens from the full LLM hidden states.
    """
    batch_size = hidden_states.shape[0]

    src_reps_list = []
    roi_reps_list = []

    # 1. Create Masks
    # Source tokens: Is Image Token AND Is NOT in the inserted block
    is_img = (input_ids == image_token_id)
    mask_src = is_img & (~inserted_content_mask)

    # RoI tokens: Is Image Token AND Is in the inserted block
    mask_roi = is_img & (inserted_content_mask)

    for i in range(batch_size):
        # Extract Source Tokens for sample i
        # Shape: [N_src_tokens, Dim]
        src_tokens = hidden_states[i][mask_src[i]]
        src_reps_list.append(src_tokens if src_tokens.numel() > 0 else None)

        # Extract RoI Tokens for sample i
        # Shape: [N_roi_tokens, Dim]
        roi_tokens = hidden_states[i][mask_roi[i]]
        roi_reps_list.append(roi_tokens if roi_tokens.numel() > 0 else None)

    return src_reps_list, roi_reps_list


def calculate_roi_align_loss_hidden_states(
        src_reps_list,  # From hidden_states
        roi_reps_list,  # From hidden_states
        surrounding_bbox_list,
        bbox_grid_thw_list,  # RoI Grids
        image_grid_thw,  # Source Grids
        hidden_size
):
    total_loss = 0.0
    valid_samples = 0

    for i in range(len(src_reps_list)):
        src_feat_flat = src_reps_list[i]
        roi_feat_flat = roi_reps_list[i]

        # Skip if either is missing
        if src_feat_flat is None or roi_feat_flat is None:
            continue

        # --- 1. Reconstruct Source Map ---
        t_src = image_grid_thw[i, 0].item()
        h_src = image_grid_thw[i, 1].item() // 2
        w_src = image_grid_thw[i, 2].item() // 2

        # Safety Check: Does flat length match grid?
        if src_feat_flat.shape[0] != (t_src * h_src * w_src):
            continue

        src_map = src_feat_flat.transpose(0, 1).view(1, hidden_size, h_src, w_src)

        # --- 2. Reconstruct RoI Map ---
        t_roi = bbox_grid_thw_list[i][0, 0].item()
        h_roi = bbox_grid_thw_list[i][0, 1].item() // 2
        w_roi = bbox_grid_thw_list[i][0, 2].item() // 2

        if roi_feat_flat.shape[0] != (t_roi * h_roi * w_roi):
            continue

        roi_map = roi_feat_flat.transpose(0, 1).view(1, hidden_size, h_roi, w_roi)

        # --- 3. Crop Source Feature Map to the BBox ---
        # We need the source crop to define the spatial target for the RoI
        x1, y1, x2, y2 = surrounding_bbox_list[i][0].int().tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_src, x2), min(h_src, y2)

        if x2 <= x1 or y2 <= y1: continue

        src_crop = src_map[:, :, y1:y2, x1:x2]

        # Determine target size from the Source Crop (Low Res)
        target_h, target_w = src_crop.shape[-2:]

        # --- 4. Alignment: Downsample RoI (High Res) to match Source Crop (Low Res) ---
        # We assume RoI contains MORE info. We compress it to verify it matches the context.
        # Use 'area' (adaptive average pooling equivalent) for better downsampling quality
        # or 'bilinear' if standard resize is preferred. 'area' is usually better for downsampling features.
        roi_map_downsampled = F.interpolate(
            roi_map,
            size=(target_h, target_w),
            mode='area'
        )

        # --- 5. Loss ---
        # Target is the Source (Anchor), effectively the "Teacher"
        # Input is the Downsampled RoI ("Student")
        target = src_crop.detach()

        loss = F.mse_loss(roi_map_downsampled, target)
        total_loss += loss
        valid_samples += 1

    if valid_samples > 0:
        return total_loss / valid_samples
    else:
        device = src_reps_list[0].device if src_reps_list[0] is not None else torch.device('cuda')
        return torch.tensor(0.0, device=device, requires_grad=True)


def interpolate_rope_for_roi(
        src_rope,  # [Seq_Src, Dim]
        src_grid_thw,  # [1, 3] -> (T, H, W)
        roi_grid_thw,  # [1, 3] -> (T, H, W)
        roi_bbox  # [1, 4] -> (x1, y1, x2, y2) normalized or relative to src grid
):
    """
    Interpolates Source RoPE embeddings to match RoI geometry.
    """
    # 1. Unpack Dimensions
    t_src, h_src, w_src = src_grid_thw[0].tolist()
    t_roi, h_roi, w_roi = roi_grid_thw[0].tolist()
    dim = src_rope.shape[-1]

    # We assume T=1 for single images. If video, we need to handle T dim.
    # Reshape Source RoPE: [Seq, Dim] -> [H, W, Dim] -> [Dim, H, W] (for grid_sample)
    # Note: src_rope might include temporal frames, logic assumes T=1 for simplicity here
    src_rope_map = src_rope.view(h_src, w_src, dim).permute(2, 0, 1).unsqueeze(0)  # [1, Dim, H, W]

    # 2. Extract BBox Coordinates (relative to Source Grid)
    # roi_bbox is typically [x1, y1, x2, y2]
    x1, y1, x2, y2 = roi_bbox[0].tolist()

    # Clamp bounds
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_src, int(x2)), min(h_src, int(y2))

    if x2 <= x1 or y2 <= y1:
        # Fallback: Just resize the whole image if bbox is broken
        return torch.nn.functional.interpolate(src_rope_map, size=(h_roi, w_roi), mode='bilinear').squeeze(0).permute(1,2,0).reshape(-1, dim)
    # 3. Crop
    src_crop = src_rope_map[..., y1:y2, x1:x2]

    # 4. Interpolate to RoI Grid Size
    # We use bilinear interpolation to preserve the phase structure of RoPE
    roi_rope_map = torch.nn.functional.interpolate(
        src_crop,
        size=(h_roi, w_roi),
        mode='bilinear',
        align_corners=False
    )  # [1, Dim, H_roi, W_roi]

    # 5. Flatten back: [1, Dim, H, W] -> [H, W, Dim] -> [Seq, Dim]
    roi_rope = roi_rope_map.squeeze(0).permute(1, 2, 0).reshape(-1, dim)

    return roi_rope

def plot_attention_mask(mask, batch_idx=0, title="Causal Attention Mask"):
    """
    Plots the attention mask for a specific sample in the batch.

    Args:
        mask: Tensor of shape [Batch, 1, Seq_Len, Seq_Len] (Boolean or Float)
        batch_idx: Index of the sample to plot.
    """
    # 1. Select the specific sample and squeeze dimensions
    # Mask shape is usually [Batch, 1, Seq, Seq] -> We want [Seq, Seq]
    if mask.dim() == 4:
        mask_2d = mask[batch_idx, 0, :, :]
    elif mask.dim() == 3:
        mask_2d = mask[batch_idx, :, :]
    else:
        mask_2d = mask

    # 2. Convert to Numpy and Boolean for visualization
    # If mask is float (-inf/0.0), convert to boolean (0.0 is True/Visible, -inf is False/Masked)
    if mask_2d.dtype == torch.float or mask_2d.dtype == torch.float16 or mask_2d.dtype == torch.bfloat16:
        # Assuming 0.0 means visible (Yellow) and -inf means masked (Purple)
        viz_data = (mask_2d == 0).float().cpu().numpy()
    else:
        # Boolean: True means visible
        viz_data = mask_2d.float().cpu().numpy()

    # 3. Plot
    plt.figure(figsize=(10, 10))
    # 'viridis': Yellow=1 (Visible), Purple=0 (Masked)
    plt.imshow(viz_data, cmap='viridis', origin='upper')

    plt.colorbar(label="Visibility (1=Visible, 0=Masked)")
    plt.title(f"{title} (Batch {batch_idx})")
    plt.xlabel("Key Position (Source Token)")
    plt.ylabel("Query Position (Target Token)")

    # Optional: Draw grid lines for better readability if sequence is short
    if viz_data.shape[0] < 100:
        plt.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()
    pass


def apply_link_embedding_to_sequence(
        hidden_states: torch.Tensor,
        link_embedding: torch.nn.Parameter,
        input_ids: torch.Tensor,
        image_token_id: int,
        inserted_content_mask: torch.Tensor,
        surrounding_bbox_list: list,
        image_grid_thw: torch.Tensor
):
    """
    Adds a learnable embedding to:
    1. All tokens belonging to the inserted RoI (defined by inserted_content_mask).
    2. The specific tokens in the Source Image that correspond to the RoI BBox.

    Args:
        hidden_states: [Batch, Seq_Len, Dim] - The full sequence embeddings.
        link_embedding: [1, 1, Dim] - The learnable zero-inited tensor.
        input_ids: [Batch, Seq_Len] - To identify image tokens.
        image_token_id: int - Token ID for images (e.g., 151655).
        inserted_content_mask: [Batch, Seq_Len] - Boolean mask, True for RoI tokens.
        surrounding_bbox_list: List of [1, 4] tensors - BBoxes [x1, y1, x2, y2] in FEATURE GRID coordinates.
        image_grid_thw: [Batch, 3] - Source image grid dimensions (T, H, W).

    Returns:
        hidden_states: The modified embeddings.
    """
    # Clone to avoid unintended in-place modification side effects if using elsewhere
    # (Optional: remove .clone() if you want strict in-place memory savings)
    #hidden_states = hidden_states.clone()
    link_embed_casted = link_embedding.to(dtype=hidden_states.dtype)
    batch_size = hidden_states.shape[0]

    for i in range(batch_size):
        # --- 1. Highlight RoI Visual Tokens ---
        # Get mask for ALL inserted content (Start + Visual + End)
        roi_block_mask = inserted_content_mask[i].bool()
        # Identify Visual Tokens (exclude Start/End special tokens)
        is_visual_token = (input_ids[i] == image_token_id)
        # Intersection: Only inserted tokens that are ALSO visual tokens
        roi_visual_mask = roi_block_mask & is_visual_token
        if roi_visual_mask.sum() == 0 or surrounding_bbox_list[i] is None:
            continue

        # Add embedding to RoI visual tokens
        hidden_states[i, roi_visual_mask] = hidden_states[i, roi_visual_mask] + link_embed_casted.squeeze(0)
        # --- 2. Highlight Corresponding Source Tokens ---
        bbox = surrounding_bbox_list[i]

        # A. Identify Source Tokens
        # Source tokens are: (Is Image Token) AND (Is NOT Inserted Content)
        is_img_token = (input_ids[i] == image_token_id)
        src_mask = is_img_token & (~roi_block_mask)

        # Get the indices of source tokens in the full sequence
        src_indices = torch.nonzero(src_mask, as_tuple=True)[0]

        # B. Reconstruct Source Grid
        # Check dimensions
        t_src = image_grid_thw[i, 0].item()
        h_src = image_grid_thw[i, 1].item() // 2  # Feature grid is half the input grid in Qwen2-VL
        w_src = image_grid_thw[i, 2].item() // 2

        # Safety check: Does the number of source tokens match the grid?
        # (If not, truncation happened)
        if src_indices.shape[0] != (t_src * h_src * w_src):
            continue  # Skip if grid mismatch

        # C. Calculate Spatial Mask for the BBox
        x1, y1, x2, y2 = bbox[0].int().tolist()

        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_src, x2), min(h_src, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Create a 2D boolean mask for the feature map [H, W]
        # We use a flat index approach to map 2D -> 1D
        grid_indices = torch.arange(h_src * w_src, device=hidden_states.device).view(h_src, w_src)

        # Select the region
        selected_grid_indices = grid_indices[y1:y2, x1:x2].flatten()

        # D. Map back to Sequence Indices
        # src_indices contains the sequence positions of the flattened image
        # selected_grid_indices contains the relative offsets within that image
        # Note: If T > 1, this logic highlights the bbox in ALL frames.
        # To handle T>1 properly, we repeat the pattern.

        if t_src > 1:
            frame_size = h_src * w_src
            all_selected = []
            for t in range(t_src):
                all_selected.append(selected_grid_indices + (t * frame_size))
            selected_grid_indices = torch.cat(all_selected)

        # Get the final indices in the main hidden_states tensor
        final_highlight_indices = src_indices[selected_grid_indices]

        # E. Add Embedding to Source Tokens
        hidden_states[i, final_highlight_indices] += link_embed_casted.squeeze(0).squeeze(0)

    return hidden_states


def get_random_roi_bbox_torch(roi_mask, min_ratio=0.1, max_ratio=1.0):
    """
    Generates a totally random bounding box within the dimensions of the roi_mask,
    and returns a new mask corresponding to that box.

    Args:
        roi_mask (torch.Tensor): Tensor of shape (H, W) or (1, H, W).
        min_ratio (float): Minimum side length ratio relative to full image.
        max_ratio (float): Maximum side length ratio relative to full image.

    Returns:
        tuple:
            - bbox (torch.Tensor): [[y1, x1, y2, x2]] on the same device.
            - new_mask (torch.Tensor): Binary mask (same shape as input) with 1s inside the bbox.
    """
    # 1. Get Image Dimensions
    if roi_mask.dim() == 3:
        _, H, W = roi_mask.shape
    else:
        H, W = roi_mask.shape

    # 2. Randomly sample dimensions
    r_h = random.uniform(min_ratio, max_ratio)
    r_w = random.uniform(min_ratio, max_ratio)

    new_h = int(H * r_h)
    new_w = int(W * r_w)

    # Safety: Ensure dimensions are at least 1 pixel
    new_h = max(1, min(new_h, H))
    new_w = max(1, min(new_w, W))

    # 3. Randomly position the box
    max_y_start = H - new_h
    max_x_start = W - new_w

    start_y = random.randint(0, max_y_start)
    start_x = random.randint(0, max_x_start)

    end_y = start_y + new_h
    end_x = start_x + new_w

    # 4. Create the new mask
    new_mask = torch.zeros_like(roi_mask)

    if new_mask.dim() == 3:
        new_mask[0, start_y:end_y, start_x:end_x] = 1
    else:
        new_mask[start_y:end_y, start_x:end_x] = 1

    bbox = torch.tensor([[start_x, start_y, end_x, end_y]], device=roi_mask.device)

    return bbox, new_mask
# if reuse_visual_hs:
#     roi_mask = roi_mask[surrounding_bbox[0, 1]:surrounding_bbox[0, 3], surrounding_bbox[0, 0]:surrounding_bbox[0, 2]]
#
#     visual_hs = image_encoder.visual_hs_before_merge  # (seq_len, dim)
#     unit = image_encoder.spatial_merge_unit  # 4
#     hs_grouped = visual_hs.view(-1, unit, visual_hs.shape[-1])
#     reverse_indices = torch.argsort(image_encoder.window_index)
#     hs_block_order = hs_grouped[reverse_indices].flatten(0, 1)
#     surrounding_bbox_2x = surrounding_bbox * 2
#
#     t, h, w = image_grid_thw[0]
#     hs_blocked = hs_block_order.view(h // 2, w // 2, 2, 2, -1)
#     hs_raster = hs_blocked.permute(0, 2, 1, 3, 4).contiguous().view(h, w, -1)
#     hs_raster = hs_raster.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
#     hs_upsampled = F.interpolate(
#         hs_raster,
#         scale_factor=2,
#         mode='bilinear',
#         align_corners=False
#     )
#     b, c, h2, w2 = hs_upsampled.shape
#     # Group 2x2 blocks again for the MLP
#     hs_grouped_new = hs_upsampled.view(b, c, h2 // 2, 2, w2 // 2, 2)
#     hs_grouped_new = hs_grouped_new.permute(0, 2, 4, 3, 5, 1).contiguous()
#     merged_input = hs_grouped_new.view(-1, c)
#
#     merged_input_norm = image_encoder.merger.ln_q(merged_input).view(-1, image_encoder.merger.hidden_size)
#     full_visual_hs = image_encoder.merger.mlp(merged_input_norm).view(h2 // 2, w2 // 2, -1)
#     cropped_visual_hs = full_visual_hs[surrounding_bbox_2x[0, 1]:surrounding_bbox_2x[0, 3],
#                         surrounding_bbox_2x[0, 0]:surrounding_bbox_2x[0, 2], :]
#     bbox_image_grid_thw = torch.tensor([[1, cropped_visual_hs.shape[0] * 2, cropped_visual_hs.shape[1] * 2]],
#                                        device=cropped_visual_hs.device)
#     cropped_visual_hs = cropped_visual_hs.reshape(-1, cropped_visual_hs.shape[-1]).contiguous()
#
#     # visual_hs_2x = image_encoder.visual_hs_before_merge.view(seq_len // spatial_merge_unit, spatial_merge_unit,-1)  # seq_len, dim
#     # visual_hs_2x = visual_hs_2x[torch.argsort(image_encoder.window_index), :].flatten(0, 1)  # seq_len, dim
#     # # visual_hs_2x = restore_spatial_structure(
#     # #     visual_hs_2x, image_grid_thw, image_encoder.window_index, image_encoder.spatial_merge_unit)[0]
#     # visual_hs_2x = visual_hs_2x.view(image_grid_thw[0, 1], image_grid_thw[0, 2], -1)  # H, W, D
#     # surrounding_bbox_2x = surrounding_bbox * 2
#     # cropped_visual_hs = visual_hs_2x[surrounding_bbox_2x[0, 1]:surrounding_bbox_2x[0, 3], surrounding_bbox_2x[0, 0]:surrounding_bbox_2x[0, 2], :]
#     # bbox_image_grid_thw = torch.tensor([[1, cropped_visual_hs.shape[0], cropped_visual_hs.shape[1]]], device=cropped_visual_hs.device)
#     # cropped_visual_hs = image_encoder.merger.ln_q(cropped_visual_hs.flatten(0, 1))  # H_crop*W_crop, D
#     # cropped_visual_hs = cropped_visual_hs.unsqueeze(-1).repeat(1, 1, image_encoder.spatial_merge_unit).flatten(-2)  # H_crop*W_crop, D_new
#     # cropped_visual_hs = image_encoder.merger.mlp(cropped_visual_hs.view(-1, image_encoder.merger.hidden_size))
#     #
#     # interplot the roi mask to the shape of cropped_visual_hs
#     roi_mask = F.interpolate(roi_mask.unsqueeze(0).unsqueeze(0),
#                              size=(bbox_image_grid_thw[0, 1] // 2, bbox_image_grid_thw[0, 2] // 2), mode='bilinear',
#                              align_corners=False).squeeze()
#     roi_mask = (roi_mask > 0.5).float()
#     batch_image_tensor.append(cropped_visual_hs)
#     sub_img_nums.append(1)
# else:
