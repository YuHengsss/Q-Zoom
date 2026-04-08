import os
import time
from typing import List

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
from lmms_eval.models.simple.qwen3vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages
from qwen_src.mm_utils import expand2square
import re

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

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
        total_tokens = 0

        language_model = self.model.language_model if hasattr(self.model, "language_model") else None
        if language_model is not None:
            if hasattr(language_model, "roi_conf_thresh"):
                language_model.roi_conf_thresh = self.roi_conf_thresh
            if hasattr(language_model, "high_res_threshold"):
                language_model.high_res_threshold = self.high_res_thresh

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            if language_model is not None and self.two_stage_roi:
                if hasattr(language_model, "roi_conf_thresh"):
                    language_model.roi_conf_thresh = self.roi_conf_thresh
                if hasattr(language_model, "high_res_threshold"):
                    language_model.high_res_threshold = self.high_res_thresh
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
            image_list = visuals
            pil_images = [img.convert("RGB") for img in image_list if img and isinstance(img, Image.Image)]
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

            for messages in chat_messages:
                for message in messages.messages:
                    for content in message.content:
                        if content.type == "text" and "Answer the question using a single word or phrase." not in content.text:
                            content.text = content.text + "\nAnswer the question using a single word or phrase."



            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            if self.two_stage_roi:
                system_message = {"role": "system", "content": "You are a helpful assistant."}
                for msg_list in batched_messages:
                    if msg_list[0]['role'] != 'system':
                        msg_list.insert(0, system_message)
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)

            # debug_ids = {
            #     'docvqa_val': [256, 404, 498, 754, 854, 870, 872, 1300, 1312, 1952, 2408, 2484, 2794, 3674, 3880, 4038, 4620, 4728, 5340, 71, 95, 111, 129, 491, 841, 1127, 2079, 2371, 3265, 3685, 4013, 4203, 4343, 4367, 4775, 4799, 4935, 5115, 5313, 5341],
            #     'textvqa_val': [8, 46, 74, 108, 156, 178, 246, 276, 406, 536, 568, 574, 644, 656, 672, 764, 808, 868, 876, 892, 904, 972, 982, 1046, 1180, 1214, 1272, 1296, 1326, 1356, 1450, 1528, 1556, 1684, 1714, 1724, 1782, 1840, 1896, 2094, 2164, 2208, 2260, 2262, 2356, 2392, 2486, 2494, 2692, 2706, 2740, 2798, 2868, 2896, 2918, 2958, 3006, 3014, 3050, 3106, 3134, 3142, 3198, 3272, 3314, 3322, 3334, 3424, 3444, 3526, 3540, 3562, 3674, 3712, 3716, 3768, 3810, 3844, 3858, 3860, 3936, 3938, 3984, 4030, 4032, 4034, 4054, 4062, 4092, 4136, 4196, 4224, 4310, 4340, 4346, 4380, 4392, 4396, 4456, 4556, 4576, 4666, 4690, 4696, 4728, 4792, 4798, 4830, 4936, 4994, 33, 41, 155, 289, 313, 333, 391, 401, 425, 489, 495, 505, 529, 535, 539, 579, 685, 825, 867, 1073, 1113, 1117, 1163, 1253, 1265, 1275, 1323, 1325, 1363, 1365, 1525, 1565, 1601, 1685, 1687, 1697, 1801, 1825, 2009, 2147, 2161, 2185, 2215, 2223, 2239, 2291, 2371, 2385, 2413, 2431, 2491, 2505, 2511, 2537, 2563, 2601, 2647, 2763, 2775, 2877, 2911, 2977, 2985, 3123, 3139, 3171, 3221, 3269, 3289, 3299, 3335, 3385, 3399, 3419, 3449, 3465, 3567, 3591, 3623, 3639, 3683, 3697, 3701, 3753, 3899, 3907, 4013, 4045, 4081, 4129, 4183, 4219, 4265, 4323, 4345, 4391, 4489, 4491, 4535, 4571, 4591, 4607, 4625, 4711, 4725, 4855, 4857, 4861, 4941, 4977, 4987],
            #     'infovqa_val': [60, 126, 152, 642, 730, 732, 996, 1006, 1364, 1476, 1482, 1652, 1704, 1752, 1864, 1908, 1910, 1988, 1996, 2058, 2124, 2130, 2154, 2196, 2224, 2372, 2630, 2790, 167, 181, 309, 341, 543, 583, 591, 731, 975, 1001, 1149, 1203, 1233, 1313, 1317, 1383, 1395, 1401, 1539, 1913, 2055, 2155, 2185, 2321, 2381, 2649, 2785],
            #     'chartqa': [18, 74, 108, 138, 164, 222, 264, 324, 464, 582, 728, 798, 818, 958, 1078, 1198, 1242, 1292, 1686, 1968, 31, 33, 53, 87, 113, 135, 195, 211, 261, 263, 323, 327, 439, 463, 489, 571, 585, 667, 687, 715, 805, 909, 915, 977, 999, 1053, 2433, 2487],
            #     'ocrbench': [198],
            #     #'textvqa_val': [74, 90, 108, 150, 170, 174, 238, 332, 358, 474, 510, 536, 550, 568, 574, 612, 622, 634, 650, 672, 722, 776, 778, 794, 876, 892, 976, 1046, 1048, 1184, 1266, 1354, 1360, 1450, 1462, 1468, 1520, 1574, 1616, 1626, 1684, 1762, 1806, 1824, 2034, 2042, 2176, 2200, 2238, 2260, 2262, 2284, 2356, 2366, 2380, 2390, 2392, 2486, 2718, 2742, 2768, 2838, 3050, 3198, 3272, 3332, 3340, 3444, 3562, 3716, 3776, 3786, 3844, 3920, 3938, 3980, 4030, 4034, 4054, 4062, 4136, 4140, 4142, 4196, 4236, 4242, 4274, 4392, 4396, 4456, 4556, 4572, 4576, 4582, 4588, 4666, 4758, 4798, 4816, 4824, 4936, 3, 109, 169, 183, 251, 391, 425, 481, 489, 495, 567, 663, 711, 769, 789, 817, 841, 915, 975, 1005, 1045, 1053, 1101, 1217, 1311, 1347, 1365, 1373, 1397, 1483, 1565, 1601, 1687, 1783, 1791, 1867, 2103, 2207, 2213, 2223, 2229, 2299, 2379, 2425, 2431, 2477, 2589, 2591, 2629, 2643, 2647, 2785, 2795, 2931, 2975, 2977, 3007, 3159, 3173, 3193, 3201, 3207, 3269, 3335, 3375, 3385, 3527, 3577, 3631, 3669, 3697, 3701, 3809, 3899, 3969, 4013, 4035, 4131, 4195, 4309, 4323, 4345, 4347, 4391, 4413, 4457, 4485, 4571, 4591, 4601, 4607, 4625, 4711, 4725, 4751, 4757, 4797, 4941, 4977, 4985]
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
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            inputs.data["src_images"] = all_images
            inputs.data["processor"] = self.processor

            # language_model.enable_twig = True
            # language_model.roi_enable2stage = True

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
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Calculate timing metrics for batch
            batch_elapsed = end_time - start_time
            batch_token_lengths = [len(ids) for ids in generated_ids_trimmed]
            batch_tokens = sum(batch_token_lengths)
            e2e_latency += batch_elapsed
            total_tokens += batch_tokens

            # ################# Start debugging ##################
            # if language_model is not None:
            #     if hasattr(language_model, "enable_twig"):
            #         language_model.enable_twig = False
            #     if hasattr(language_model, "roi_enable2stage"):
            #         language_model.roi_enable2stage = False
            #     bbox_img = getattr(language_model, "bbox_img", None)
            #     if bbox_img is not None:
            #         debug_messages = []
            #         for msg in batched_messages:
            #             msg_copy = []
            #             for item in msg:
            #                 if item.get("role") == "user" and isinstance(item.get("content"), list):
            #                     content = list(item["content"])
            #                     content.append(
            #                         {
            #                             "type": "image",
            #                             "image": bbox_img,
            #                             "max_pixels": self.max_pixels,
            #                             "min_pixels": self.min_pixels,
            #                         }
            #                     )
            #                     msg_copy.append({"role": item["role"], "content": content})
            #                 else:
            #                     msg_copy.append(item)
            #             debug_messages.append(msg_copy)
            #
            #         debug_texts = [
            #             self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            #             for m in debug_messages
            #         ]
            #         #debug_texts[0] = debug_texts[0].replace('\nAnswer the question using a single word or phrase.', '')
            #         debug_image_inputs, debug_video_inputs = process_vision_info(debug_messages)
            #         debug_inputs = self.processor(
            #             text=debug_texts,
            #             images=debug_image_inputs,
            #             videos=debug_video_inputs,
            #             padding=True,
            #             return_tensors="pt",
            #         )
            #         if self.device_map == "auto":
            #             debug_inputs = debug_inputs.to("cuda")
            #         else:
            #             debug_inputs = debug_inputs.to(self.device)
            #         debug_inputs.data["src_images"] = all_images
            #         debug_inputs.data["processor"] = self.processor
            #
            #         debug_cont = self.model.generate(
            #             **debug_inputs,
            #             eos_token_id=self.tokenizer.eos_token_id,
            #             pad_token_id=pad_token_id,
            #             do_sample=current_gen_kwargs["do_sample"],
            #             temperature=current_gen_kwargs["temperature"],
            #             top_p=current_gen_kwargs["top_p"],
            #             num_beams=current_gen_kwargs["num_beams"],
            #             max_new_tokens=current_gen_kwargs["max_new_tokens"],
            #             top_k=current_gen_kwargs.get("top_k", None),
            #             use_cache=self.use_cache,
            #         )
            #         debug_trimmed = [
            #             out_ids[len(in_ids) :] for in_ids, out_ids in zip(debug_inputs.input_ids, debug_cont)
            #         ]
            #         debug_answers = self.processor.batch_decode(
            #             debug_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            #         )
            #         print("Debug bbox answers:", debug_answers)
            visual_token_num = None
            if hasattr(self.model, "model") and hasattr(self.model.model, "visual_token_num"):
                visual_token_num = self.model.model.visual_token_num
            elif language_model is not None and hasattr(language_model, "visual_token_num"):
                visual_token_num = language_model.visual_token_num

            high_res_pred = None
            if language_model is not None and getattr(language_model, "high_res_pred", None) is not None:
                high_res_pred = language_model.high_res_pred
            elif hasattr(self.model, "model") and getattr(self.model.model, "high_res_pred", None) is not None:
                high_res_pred = self.model.model.high_res_pred

            high_res_pred_score = None
            pred_vector = None
            if self.rec_high_res_signal and high_res_pred is not None:
                high_res_pred_score = high_res_pred.float().cpu().detach().numpy()
                pred_vector = np.asarray(high_res_pred_score).reshape(-1)

            sample_elapsed = batch_elapsed / max(len(answers), 1)
            for sample_idx, (ans, context) in enumerate(zip(answers, texts)):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")

                sample_doc_id = doc_id[sample_idx] if sample_idx < len(doc_id) else doc_id[0]
                sample_tokens = batch_token_lengths[sample_idx] if sample_idx < len(batch_token_lengths) else 0
                tmp_sample = {
                    "question": context,
                    "visual_token_num": visual_token_num,
                    "sample_tokens": sample_tokens,
                    "sample_latency": sample_elapsed,
                    "sample_tps": (sample_tokens / sample_elapsed) if sample_elapsed > 0 else 0.0,
                }
                if pred_vector is not None and sample_idx < len(pred_vector):
                    tmp_sample["high_res_pred_score"] = float(pred_vector[sample_idx])
                elif high_res_pred_score is not None:
                    tmp_sample["high_res_pred_score"] = high_res_pred_score
                self.high_res_pred_dict[sample_doc_id] = tmp_sample

        res = re_ords.get_original(res)

        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
                "processed_samples": len(res),
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        if True:
            task_name = task[0]
            rank = str(self.rank)
            self.high_res_pred_dict["metric_dict"] = metric_dict
            save_dir = os.environ.get("LMMS_DUMP_DIR", "./logs/high_res_pred")
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{task_name}_analysis_rank{rank}_{self.model_name}_min{self.min_pixels}_max{self.max_pixels}.pkl"
            import pickle
            with open(save_path, "wb") as f:
                pickle.dump(self.high_res_pred_dict, f)
            print("Save prediction results to ", save_path)
        return res
