import base64
import copy
import json
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.chat.qwen2_5_roi_baseline_helper import (
    Qwen25Stage1PseudoRoiHelper,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages
from qwen_src.mm_utils import expand2square
import re
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_chat")
class Qwen2_5_VL(Qwen2_5_VLSimple):
    is_simple = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._roi_baseline_helper = None
        self.save_raw_response = os.environ.get("LMMS_SAVE_RAW_RESPONSE", "0").lower() in ("1", "true", "yes")
        self.raw_response_root = os.environ.get("LMMS_RAW_RESPONSE_ROOT", "./logs/raw_responses")
        self.raw_response_method = os.environ.get("LMMS_RAW_RESPONSE_METHOD", "qwen2_5_vl")

    def _append_raw_response(self, task_name: str, doc_id, question: str, raw_response: str, clean_response: str):
        if not self.save_raw_response:
            return
        try:
            task = task_name or os.environ.get("LMMS_ROI_DEBUG_TASK", "unknown_task")
            out_dir = os.path.join(self.raw_response_root, self.raw_response_method, task)
            os.makedirs(out_dir, exist_ok=True)
            rank = int(getattr(self, "rank", 0))
            out_path = os.path.join(out_dir, f"raw_responses_rank{rank}.jsonl")
            row = {
                "task": task,
                "doc_id": int(doc_id) if doc_id is not None else None,
                "question": question,
                "raw_response": raw_response,
                "clean_response": clean_response,
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            eval_logger.warning(f"raw response save failed: {e}")

    def _append_roi_debug_meta(self, task_name: str, doc_id, payload: dict):
        save_roi_debug = os.environ.get("LMMS_SAVE_ROI_DEBUG", "0").lower() in ("1", "true", "yes")
        if not save_roi_debug:
            return
        try:
            task = task_name or os.environ.get("LMMS_ROI_DEBUG_TASK", "unknown_task")
            method = os.environ.get("LMMS_ROI_DEBUG_METHOD", self.raw_response_method or "qwen2_5_vl")
            root = os.environ.get("LMMS_ROI_DEBUG_ROOT", "./logs/roi_debug")
            rank = int(getattr(self, "rank", 0))
            did = "na"
            did_val = None
            if doc_id is not None:
                try:
                    did_val = int(doc_id)
                    did = str(did_val)
                except Exception:
                    did = str(doc_id)
                    did_val = doc_id

            out_dir = os.path.join(root, task, f"doc_{did}")
            os.makedirs(out_dir, exist_ok=True)
            row = {
                "record_type": "inference_result",
                "method": method,
                "task": task,
                "doc_id": did_val,
                "rank": rank,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            }
            row.update(payload or {})
            out_path = os.path.join(out_dir, "roi_meta.jsonl")
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            eval_logger.warning(f"roi meta save failed: {e}")

    def _decode_image(self, image_obj) -> Optional[Image.Image]:
        if isinstance(image_obj, Image.Image):
            return image_obj.convert("RGB")
        if isinstance(image_obj, str):
            try:
                if image_obj.startswith("data:image"):
                    encoded = image_obj.split(",", 1)[1]
                    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
                return Image.open(image_obj).convert("RGB")
            except Exception:
                return None
        return None

    def _extract_primary_image_and_text(self, hf_messages, fallback_context: str = ""):
        for msg in hf_messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            first_image = None
            text_chunks = []
            for item in content:
                item_type = item.get("type")
                if item_type == "image" and first_image is None:
                    first_image = self._decode_image(item.get("image"))
                elif item_type == "text":
                    text_chunks.append(item.get("text", ""))
            question_text = " ".join([x for x in text_chunks if x]).strip()
            if not question_text:
                question_text = str(fallback_context or "").strip()
            return first_image, question_text
        return None, str(fallback_context or "").strip()

    def _insert_crop_image(self, hf_messages, crop_img: Image.Image):
        for msg in hf_messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            insert_idx = len(content)
            for idx, item in enumerate(content):
                if item.get("type") == "text":
                    insert_idx = idx
                    break
            content.insert(insert_idx, {"type": "image", "image": crop_img})
            msg["content"] = content
            return

    def _compute_roi_baseline_bbox(self, image: Image.Image, question_text: str, task_name: str = ""):
        if image is None or not question_text:
            return None

        if self._roi_baseline_helper is None:
            self._roi_baseline_helper = Qwen25Stage1PseudoRoiHelper(
                model=self.model,
                processor=self.processor,
                device=self.device,
                device_map=self.device_map,
            )
        return self._roi_baseline_helper.compute_bbox(
            image=image,
            question_text=question_text,
            task_name=task_name,
            max_new_tokens=32,
            sink_thresh=1e-2,
            binary_coff=0.2,
            bg_coff=0.05,
            pseudo_blur_kernel_size=3,
        )

    def augment_batched_messages(self, batched_messages, contexts=None, task_name: str = "", **kwargs):
        """Hook for model variants to edit HF-format messages before tokenization."""
        if process_vision_info is None:
            return batched_messages
        if not getattr(self, "roi_baseline", False) or getattr(self, "two_stage_roi", False):
            return batched_messages

        augmented = []
        for idx, hf_messages in enumerate(batched_messages):
            context = ""
            if contexts is not None and idx < len(contexts):
                context = contexts[idx]
            candidate = copy.deepcopy(hf_messages)
            image, question_text = self._extract_primary_image_and_text(candidate, fallback_context=context)
            if image is None:
                augmented.append(candidate)
                continue

            try:
                bbox = self._compute_roi_baseline_bbox(
                    image,
                    question_text,
                    task_name=task_name,
                )
            except Exception as e:
                eval_logger.warning(f"ROI baseline bbox failed, fallback to original image: {e}")
                bbox = None

            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if x2 > x1 and y2 > y1:
                    crop_img = image.crop((x1, y1, x2, y2))
                    self._insert_crop_image(candidate, crop_img)
            augmented.append(candidate)
        return augmented

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        roi_identification_latency = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            if self.two_stage_roi and hasattr(self.model, "model"):
                self.model.model.roi_conf_thresh = self.roi_conf_thresh
                self.model.model.high_res_thresh = self.high_res_thresh
            grounding_task_names = ["refcocog", "screenspot"]
            is_grounding_task = any([name in task_name.lower() for name in grounding_task_names])

            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            # visuals = [[expand2square(img, (127, 127, 127)) for img in visual] for visual in visuals]
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            all_images = []
            # ####for debugging: tensor([1880, 1100, 2048, 1266], device='cuda:0')-sample1 xyxy format, set image in this area to 0
            # ###tensor([1309,  833, 1981, 1000], device='cuda:0')-sample116
            # tmp_image = np.array(image_list[0])
            # tmp_image[833:1000,1309:1981,:] = 127
            # image_list[0] = Image.fromarray(tmp_image)
            #### End debugging
            image_list = visuals
            pil_images = [img.convert("RGB") for img in image_list if img]  ## should assert bs=1 here
            all_images.extend(pil_images)

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames
            ##add Answer the question using a single word or phrase. to the prompt if it does not exist
            for messages in chat_messages:
                for message in messages.messages:
                    for content in message.content:
                        # if not is_grounding_task:
                        #     pass
                        # if content.type == "text" and "Answer the question using a single word or phrase." in content.text:
                        #     content.text = content.text.replace("Answer the question using a single word or phrase.", "").strip()

                        if content.type == "text" and "Answer the question using a single word or phrase." not in content.text:
                            content.text = content.text + "\nAnswer the question using a single word or phrase."
                        # else:
                        #     if content.type == "text" and self.two_stage_roi:
                        #         content.text = content.text + "\n If there is two images, please only refer to the first image for answer."
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            contexts = list(ctx)
            augment_start_time = time.time()
            try:
                batched_messages = self.augment_batched_messages(
                    batched_messages,
                    contexts=contexts,
                    task_name=task_name,
                    doc_ids=doc_id,
                )
                augment_elapsed = time.time() - augment_start_time
            except Exception as e:
                augment_elapsed = time.time() - augment_start_time
                eval_logger.warning(f"augment_batched_messages failed, fallback to original messages: {e}")
            #batched_messages[0][0]['content'][0]['image'] = image_list[0] # for debugging
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            # # # #all_images = image_inputs
            # debug_ids = {
            #     'docvqa_val': [256, 404, 498, 754, 854, 870, 872, 1300, 1312, 1952, 2408, 2484, 2794, 3674, 3880, 4038, 4620, 4728, 5340, 71, 95, 111, 129, 491, 841, 1127, 2079, 2371, 3265, 3685, 4013, 4203, 4343, 4367, 4775, 4799, 4935, 5115, 5313, 5341],
            #     'textvqa_val': [16, 152, 192, 252, 402, 474, 704, 856, 868, 1114, 1128, 1354, 1496, 1620, 1750, 1874, 1930, 1966, 2022, 2124, 2152, 2174, 2260, 2286, 2484, 2686, 2736, 2776, 2958, 3498, 3608, 3622, 3680, 3692, 3920, 3946, 4056, 4140, 4246, 4272, 4274, 4294, 4396, 4418, 4602, 4792, 4828, 19, 123, 275, 311, 505, 595, 627, 711, 829, 1035, 1063, 1113, 1163, 1193, 1265, 1299, 1363, 1655, 1667, 2035, 2187, 2393, 2465, 2477, 2585, 2631, 2809, 2881, 2901, 2941, 2981, 3115, 3249, 3289, 3555, 3739, 3757, 3883, 4043, 4183, 4353, 4457, 4467, 4615, 4707, 4821, 4897, 4947, 4969],
            #     'infovqa_val': [60, 126, 152, 642, 730, 732, 996, 1006, 1364, 1476, 1482, 1652, 1704, 1752, 1864, 1908, 1910, 1988, 1996, 2058, 2124, 2130, 2154, 2196, 2224, 2372, 2630, 2790, 167, 181, 309, 341, 543, 583, 591, 731, 975, 1001, 1149, 1203, 1233, 1313, 1317, 1383, 1395, 1401, 1539, 1913, 2055, 2155, 2185, 2321, 2381, 2649, 2785],
            #     'chartqa': [18, 74, 108, 138, 164, 222, 264, 324, 464, 582, 728, 798, 818, 958, 1078, 1198, 1242, 1292, 1686, 1968, 31, 33, 53, 87, 113, 135, 195, 211, 261, 263, 323, 327, 439, 463, 489, 571, 585, 667, 687, 715, 805, 909, 915, 977, 999, 1053, 2433, 2487],
            #     'ocrbench': [206, 230, 248, 256, 300, 338, 384, 536, 860, 215, 225, 367, 401, 447, 665]
            # }
            #
            # if int(doc_id[0]) not in debug_ids.get(task_name.lower(), []):
            #     res.append('A')
            #     self.cache_hook.add_partial("generate_until", (texts[0], gen_kwargs), res)
            #     pbar.update(1)
            #     continue
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None
            #current_gen_kwargs["max_new_tokens"] = 256
            inputs.data['src_images'] = all_images
            inputs.data['processor'] = self.processor
            # Optional debug context for downstream ROI-crop save hooks.
            os.environ["LMMS_ROI_DEBUG_TASK"] = str(task_name)
            try:
                os.environ["LMMS_ROI_DEBUG_DOC_IDS"] = ",".join(str(int(x)) for x in doc_id)
            except Exception:
                os.environ["LMMS_ROI_DEBUG_DOC_IDS"] = ",".join(str(x) for x in doc_id)

            start_time = time.time()

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            # skip_special_tokens = not is_grounding_task
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Calculate timing metrics for batch
            generation_elapsed = end_time - start_time
            batch_elapsed = generation_elapsed + augment_elapsed
            batch_token_lengths = [len(ids) for ids in generated_ids_trimmed]
            batch_tokens = sum(batch_token_lengths)
            e2e_latency += batch_elapsed
            roi_identification_latency += augment_elapsed
            total_tokens += batch_tokens

            visual_token_num = self.model.model.visual_token_num if hasattr(self.model.model, "visual_token_num") else None
            high_res_pred_score = None
            if self.rec_high_res_signal and getattr(self.model.model, "high_res_pred", None) is not None:
                high_res_pred_score = self.model.model.high_res_pred.float().cpu().detach().numpy()

            sample_elapsed = batch_elapsed / max(len(answers), 1)
            pred_vector = None
            if high_res_pred_score is not None:
                pred_vector = np.asarray(high_res_pred_score).reshape(-1)

            for sample_idx, (ans, context) in enumerate(zip(answers, texts)):
                clean_ans = parse_reasoning_model_answer(ans)
                if is_grounding_task:
                    ###normalize the bbox coordinates to original image size
                    image_h, image_w = int(inputs['image_grid_thw'][0][1]*14), int(inputs['image_grid_thw'][0][2]*14)
                    pattern = r"\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*"
                    match = re.search(pattern, clean_ans)
                    if match:
                        ##normalize bbox to 0-1
                        bbox = [float(match.group(i)) for i in range(1, 5)]
                        bbox[::2] = [bbox[0]/image_w, bbox[2]/image_w]
                        bbox[1::2] = [bbox[1]/image_h, bbox[3]/image_h]
                        clean_ans = f"({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f})"
                    else:
                        clean_ans = "(0, 0, 0, 0)"

                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")

                sample_doc_id = doc_id[sample_idx] if sample_idx < len(doc_id) else doc_id[0]
                sample_tokens = batch_token_lengths[sample_idx] if sample_idx < len(batch_token_lengths) else 0
                sample_roi_elapsed = augment_elapsed / max(len(answers), 1)
                tmp_sample = {
                    "question": context,
                    "visual_token_num": visual_token_num,
                    "sample_tokens": sample_tokens,
                    "sample_latency": sample_elapsed,
                    "sample_roi_latency": sample_roi_elapsed,
                    "sample_tps": (sample_tokens / sample_elapsed) if sample_elapsed > 0 else 0.0,
                }
                if pred_vector is not None and sample_idx < len(pred_vector):
                    tmp_sample["high_res_pred_score"] = float(pred_vector[sample_idx])
                elif high_res_pred_score is not None:
                    tmp_sample["high_res_pred_score"] = high_res_pred_score
                if self.save_raw_response:
                    tmp_sample["raw_response"] = ans
                    tmp_sample["clean_response"] = clean_ans

                self.high_res_pred_dict[sample_doc_id] = tmp_sample
                self._append_raw_response(
                    task_name=task_name,
                    doc_id=sample_doc_id,
                    question=context,
                    raw_response=ans,
                    clean_response=clean_ans,
                )
                self._append_roi_debug_meta(
                    task_name=task_name,
                    doc_id=sample_doc_id,
                    payload={
                        "question": context,
                        "raw_response": ans,
                        "clean_response": clean_ans,
                        "sample_latency_sec": sample_elapsed,
                        "sample_roi_latency_sec": sample_roi_elapsed,
                        "sample_tokens": int(sample_tokens),
                        "sample_tps": (sample_tokens / sample_elapsed) if sample_elapsed > 0 else 0.0,
                    },
                )

            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
                "processed_samples": len(res),
                "roi_identification_latency": roi_identification_latency,
                "roi_identification_ratio": (roi_identification_latency / e2e_latency) if e2e_latency > 0 else 0.0,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        # Optionally dump per-sample high-res prediction scores for analysis.
        # Disabled by default; set LMMS_DUMP_HIGH_RES_PRED=1 to enable.
        if os.environ.get("LMMS_DUMP_HIGH_RES_PRED", "0") == "1":
            task_name = task[0]
            rank = str(self.rank)
            self.high_res_pred_dict['metric_dict'] = metric_dict
            save_dir = os.environ.get("LMMS_DUMP_DIR", "./logs/high_res_pred")
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{task_name}_analysis_rank{rank}_{self.model_name}_min{self.min_pixels}_max{self.max_pixels}.pkl"
            import pickle
            with open(save_path, "wb") as f:
                pickle.dump(self.high_res_pred_dict, f)
            print('Save prediction results to ', save_path)
        return res
