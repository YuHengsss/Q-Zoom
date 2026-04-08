import os
import re

# All Q-Zoom dataset paths are resolved at runtime via the QZOOM_DATA_ROOT
# environment variable so that this file does not have to encode any user-
# specific filesystem layout. Set:
#
#   export QZOOM_DATA_ROOT=/path/to/your/datasets
#
# Inside QZOOM_DATA_ROOT, the loaders expect the JSONL filenames listed
# below; rename / symlink your own data to match, or override the entries
# directly with `os.environ["QZOOM_<NAME>_ANNOT"]`.

QZOOM_DATA_ROOT = os.environ.get("QZOOM_DATA_ROOT", "PATH_TO_QZOOM_DATA_ROOT")


def _entry(name: str, default_filename: str) -> dict:
    annot = os.environ.get(
        f"QZOOM_{name.upper()}_ANNOT",
        f"{QZOOM_DATA_ROOT}/{default_filename}",
    )
    data = os.environ.get(f"QZOOM_{name.upper()}_DATA", QZOOM_DATA_ROOT)
    return {"annotation_path": annot, "data_path": data}


# --- Stock Cambrian / placeholder entries kept for upstream compatibility ---
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}
CAMBRIAN_737K_PACK = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": "",
}
MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}
CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}
VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

# --- Q-Zoom training datasets ---
# Each entry resolves its annotation/data paths from QZOOM_DATA_ROOT or
# from a per-dataset env var override.
ROI_QWEN25_3B_POST_SFT = _entry("qwen25_3b", "roi-post-sft-qwen2_5vl-3b.jsonl")
ROI_QWEN25_7B_POST_SFT = _entry("qwen25_7b", "roi-post-sft-qwen2_5vl-7b.jsonl")
ROI_QWEN3_4B_POST_SFT = _entry("qwen3_4b", "roi-post-sft-qwen3vl-4b.jsonl")
ROI_QWEN3_8B_POST_SFT = _entry("qwen3_8b", "roi-post-sft-qwen3vl-8b.jsonl")

# Generic alias used by the example Stage-2 training scripts. The
# default filename embeds ${BACKBONE_TAG} so different backbones do not
# overwrite each other (matches examples/stage2_train_eval/01_build_stage2_data.sh).
# Override QZOOM_QZOOM_STAGE2_POST_SFT_ANNOT to point at the exact JSONL
# you produced.
_QZOOM_BACKBONE_TAG = os.environ.get("BACKBONE_TAG", "qwen2_5vl_7b")
QZOOM_STAGE2_POST_SFT = _entry(
    "qzoom_stage2_post_sft",
    f"qzoom_post_sft_stage2_{_QZOOM_BACKBONE_TAG}.jsonl",
)


data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "roi_qwen25_3b_post_sft": ROI_QWEN25_3B_POST_SFT,
    "roi_qwen25_7b_post_sft": ROI_QWEN25_7B_POST_SFT,
    "roi_qwen3_4b_post_sft": ROI_QWEN3_4B_POST_SFT,
    "roi_qwen3_8b_post_sft": ROI_QWEN3_8B_POST_SFT,
    "qzoom_stage2_post_sft": QZOOM_STAGE2_POST_SFT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
