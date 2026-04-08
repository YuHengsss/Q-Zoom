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
    parser = argparse.ArgumentParser(description="Unified Stage3 pipeline from one merged universal input jsonl.")
    parser.add_argument("--code-root", type=str, default=os.getcwd())
    parser.add_argument("--universal-input-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")

    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-type", type=str, default="auto")
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument("--gpu-ids", type=str, default="0,1")

    parser.add_argument("--stage3-output", type=str, default="")

    parser.add_argument("--skip-generate", action="store_true")
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
    search_roots = [x.strip() for x in str(args.model_search_roots).split(",") if x.strip()]

    args.model_path = resolve_local_model_path(args.model_path, search_roots) if args.model_path else ""
    args.processor_path = resolve_local_model_path(args.processor_path, search_roots) if args.processor_path else ""

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    model_tag = build_model_tag(args.model_path)
    stage3_output = (
        args.stage3_output
        if args.stage3_output
        else f"/stage3results/stage3_{model_tag}.pkl"
    )
    stage3_output = os.path.expanduser(os.path.expandvars(stage3_output))

    if not os.path.isfile(universal_input_file):
        raise FileNotFoundError(f"Universal input file not found: {universal_input_file}")

    ensure_parent(stage3_output)
    offline_env = os.environ.copy()
    if args.offline:
        offline_env["HF_HUB_OFFLINE"] = "1"
        offline_env["TRANSFORMERS_OFFLINE"] = "1"
        offline_env["HF_DATASETS_OFFLINE"] = "1"

    if not args.skip_generate:
        require_arg(args, "model_path")
        if args.offline and not os.path.exists(args.model_path):
            raise FileNotFoundError(
                f"Offline mode requires local path for --model-path: {args.model_path}"
            )
        cmd = [
            sys.executable,
            "analysis/making_gated_training_samples.py",
            "--model-path",
            args.model_path,
            "--question-file",
            universal_input_file,
            "--image-folder",
            image_folder,
            "--dst-save-path",
            stage3_output,
            "--gpu-ids",
            args.gpu_ids,
            "--model-type",
            args.model_type,
        ]
        if args.processor_path:
            cmd.extend(["--processor-path", args.processor_path])
        run_cmd_with_env(cmd, cwd=code_root, env=offline_env)

    print(f"Stage3 unified pipeline finished. Output: {stage3_output}")


if __name__ == "__main__":
    main()
