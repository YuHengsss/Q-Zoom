import argparse
import os
import re
import subprocess
import sys

# Allow running this script from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_cmd(cmd, cwd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_cmd_with_env(cmd, cwd, env):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_model_tag(model_path):
    if not model_path:
        return "model"
    raw = str(model_path).strip().rstrip("/\\")
    if not raw:
        return "model"
    tail = raw.split("/")[-1].split("\\")[-1]
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tail).strip("_")
    return tag if tag else "model"


def _expand(path):
    return os.path.expanduser(os.path.expandvars(path))


def resolve_local_model_path(model_path, search_roots):
    model_path = _expand(model_path)
    if not model_path:
        return model_path
    if os.path.exists(model_path):
        return model_path

    tail = model_path.rstrip("/\\").split("/")[-1]
    for root in search_roots:
        if not root:
            continue
        candidate = os.path.join(_expand(root), tail)
        if os.path.exists(candidate):
            print(f"[PathResolve] Resolved '{model_path}' -> '{candidate}'")
            return candidate
    return model_path


def require_arg(args, name):
    value = getattr(args, name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"--{name.replace('_', '-')} is required for this run.")


def main():
    parser = argparse.ArgumentParser(description="Unified Stage2 pipeline from one merged universal input jsonl.")
    parser.add_argument("--code-root", type=str, default=os.getcwd())
    parser.add_argument("--universal-input-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")

    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument("--roi-model-path", type=str, default="")
    parser.add_argument("--model-type", type=str, default="auto")
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument("--gpu-ids", type=str, default="0,1")
    parser.add_argument("--use-short-prompt", action="store_true")

    parser.add_argument("--base-output", type=str, default="")
    parser.add_argument("--roi-output", type=str, default="")

    parser.add_argument("--judge-model-path", type=str, default="")
    parser.add_argument("--judge-model-type", type=str, default="auto")
    parser.add_argument("--judge-processor-path", type=str, default="")
    parser.add_argument("--judge-gpu-ids", type=str, default="")
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--judge-num-workers", type=int, default=4)
    parser.add_argument("--judge-dataset-name", type=str, default="universal")
    parser.add_argument("--reg-output", type=str, default="")
    parser.add_argument("--imp-output", type=str, default="")

    parser.add_argument("--post-sft-output", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--mixing-ratio", type=float, default=0.0)
    parser.add_argument("--use-gt", action="store_true")
    parser.add_argument("--convert-use-short-prompt", action="store_true")

    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-roi", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--model-search-roots",
        type=str,
        default="",
        help="Comma-separated local roots to resolve repo-style model names (e.g., Qwen/...).",
    )
    args = parser.parse_args()

    code_root = os.path.expanduser(os.path.expandvars(args.code_root))
    universal_input_file = os.path.expanduser(os.path.expandvars(args.universal_input_file))
    image_folder = os.path.expanduser(os.path.expandvars(args.image_folder))
    data_root = os.path.expanduser(os.path.expandvars(args.data_root))
    search_roots = [x.strip() for x in str(args.model_search_roots).split(",") if x.strip()]

    args.base_model_path = resolve_local_model_path(args.base_model_path, search_roots) if args.base_model_path else ""
    args.roi_model_path = resolve_local_model_path(args.roi_model_path, search_roots) if args.roi_model_path else ""
    args.judge_model_path = resolve_local_model_path(args.judge_model_path, search_roots) if args.judge_model_path else ""
    args.processor_path = resolve_local_model_path(args.processor_path, search_roots) if args.processor_path else ""
    args.judge_processor_path = (
        resolve_local_model_path(args.judge_processor_path, search_roots) if args.judge_processor_path else ""
    )

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    model_ref_for_tag = args.roi_model_path if args.roi_model_path else args.base_model_path
    model_tag = build_model_tag(model_ref_for_tag)
    posttrain_root = ""

    base_output = args.base_output if args.base_output else f"{posttrain_root}/stage2_base_{model_tag}.pkl"
    roi_output = args.roi_output if args.roi_output else f"{posttrain_root}/stage2_roi_{model_tag}.pkl"
    reg_output = args.reg_output if args.reg_output else f"{posttrain_root}/regressions-stage2-{model_tag}.json"
    imp_output = args.imp_output if args.imp_output else f"{posttrain_root}/improvements-stage2-{model_tag}.json"
    post_sft_output = (
        args.post_sft_output if args.post_sft_output else f"./roi-post-sft-stage2-{model_tag}.jsonl"
    )

    base_output = os.path.expanduser(os.path.expandvars(base_output))
    roi_output = os.path.expanduser(os.path.expandvars(roi_output))
    reg_output = os.path.expanduser(os.path.expandvars(reg_output))
    imp_output = os.path.expanduser(os.path.expandvars(imp_output))
    post_sft_output = os.path.expanduser(os.path.expandvars(post_sft_output))

    if not os.path.isfile(universal_input_file):
        raise FileNotFoundError(f"Universal input file not found: {universal_input_file}")

    ensure_parent(base_output)
    ensure_parent(roi_output)
    ensure_parent(reg_output)
    ensure_parent(imp_output)
    ensure_parent(post_sft_output)

    print(
        "Resolved outputs:\n"
        f"  base_output: {base_output}\n"
        f"  roi_output: {roi_output}\n"
        f"  reg_output: {reg_output}\n"
        f"  imp_output: {imp_output}\n"
        f"  post_sft_output: {post_sft_output}"
    )

    offline_env = os.environ.copy()
    if args.offline:
        offline_env["HF_HUB_OFFLINE"] = "1"
        offline_env["TRANSFORMERS_OFFLINE"] = "1"
        offline_env["HF_DATASETS_OFFLINE"] = "1"

    if args.offline:
        for path_name, path_value in [
            ("base_model_path", args.base_model_path),
            ("roi_model_path", args.roi_model_path),
            ("judge_model_path", args.judge_model_path),
        ]:
            if path_value and not os.path.exists(path_value):
                raise FileNotFoundError(
                    f"Offline mode requires local path for --{path_name.replace('_', '-')}: {path_value}"
                )

    if not args.skip_base:
        require_arg(args, "base_model_path")
        cmd = [
            sys.executable,
            "analysis/making_base_model_response.py",
            "--model-path",
            args.base_model_path,
            "--question-file",
            universal_input_file,
            "--image-folder",
            image_folder,
            "--dst-save-path",
            base_output,
            "--gpu-ids",
            args.gpu_ids,
            "--model-type",
            args.model_type,
        ]
        if args.processor_path:
            cmd.extend(["--processor-path", args.processor_path])
        if args.use_short_prompt:
            cmd.append("--use-short-prompt")
        run_cmd_with_env(cmd, cwd=code_root, env=offline_env)

    if not args.skip_roi:
        require_arg(args, "roi_model_path")
        cmd = [
            sys.executable,
            "analysis/making_roi_model_response.py",
            "--model-path",
            args.roi_model_path,
            "--question-file",
            universal_input_file,
            "--image-folder",
            image_folder,
            "--dst-save-path",
            roi_output,
            "--gpu-ids",
            args.gpu_ids,
            "--model-type",
            args.model_type,
        ]
        if args.processor_path:
            cmd.extend(["--processor-path", args.processor_path])
        if args.use_short_prompt:
            cmd.append("--use-short-prompt")
        run_cmd_with_env(cmd, cwd=code_root, env=offline_env)

    if not args.skip_filter:
        require_arg(args, "judge_model_path")
        judge_gpu_ids = args.judge_gpu_ids if args.judge_gpu_ids else args.gpu_ids
        cmd = [
            sys.executable,
            "analysis/filter_roi_fails.py",
            "--model-path",
            args.judge_model_path,
            "--model-type",
            args.judge_model_type,
            "--base-pkl",
            base_output,
            "--roi-pkl",
            roi_output,
            "--dataset-name",
            args.judge_dataset_name,
            "--save-path",
            reg_output,
            "--save-path-improvements",
            imp_output,
            "--batch-size",
            str(args.judge_batch_size),
            "--num_workers",
            str(args.judge_num_workers),
            "--gpu-ids",
            judge_gpu_ids,
        ]
        if args.judge_processor_path:
            cmd.extend(["--processor-path", args.judge_processor_path])
        run_cmd_with_env(cmd, cwd=code_root, env=offline_env)

    if not args.skip_convert:
        cmd = [
            sys.executable,
            "analysis/convert_json2qwen_training.py",
            "--reg-file",
            reg_output,
            "--imp-file",
            imp_output,
            "--output",
            post_sft_output,
            "--data-root",
            data_root,
            "--mixing-ratio",
            str(args.mixing_ratio),
        ]
        if args.use_gt:
            cmd.append("--use-gt")
        if args.convert_use_short_prompt:
            cmd.append("--use-short-prompt")
        run_cmd_with_env(cmd, cwd=code_root, env=offline_env)

    print("Stage2 unified pipeline finished.")


if __name__ == "__main__":
    main()
