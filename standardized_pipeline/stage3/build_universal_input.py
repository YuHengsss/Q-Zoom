import argparse
import json
import os
import sys
from collections import Counter

# Allow running this script from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def expand_vars(text, env_map):
    if not isinstance(text, str):
        return text
    out = text
    for key, value in env_map.items():
        out = out.replace(f"${{{key}}}", str(value))
    return os.path.expanduser(os.path.expandvars(out))


def load_json_or_jsonl(path):
    path = os.path.expanduser(os.path.expandvars(path))
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def choose_question(record, question_key=None):
    if question_key and question_key in record:
        return record[question_key]
    for key in ("text", "prompt", "question"):
        if key in record:
            return record[key]
    return None


def choose_question_id(record, id_key=None, fallback="0"):
    if id_key and id_key in record:
        return record[id_key]
    for key in ("question_id", "id"):
        if key in record:
            return record[key]
    return fallback


def normalize_image_path(raw_image, image_prefix):
    image = str(raw_image)
    if os.path.isabs(image):
        return image
    if image_prefix:
        normalized_prefix = image_prefix.replace("\\", "/").rstrip("/")
        normalized_image = image.replace("\\", "/")
        if not normalized_image.startswith(normalized_prefix + "/"):
            image = os.path.join(normalized_prefix, image)
    return image.replace("\\", "/")


def build_records(manifest):
    data_root = manifest.get("data_root", "")
    env_map = {"data_root": data_root}

    sources = manifest.get("sources", [])
    if not isinstance(sources, list) or len(sources) == 0:
        raise ValueError("Manifest must contain non-empty 'sources' list.")

    all_records = []
    skipped = 0
    for src_idx, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        enabled = bool(source.get("enabled", True))
        if not enabled:
            continue

        name = str(source.get("name", f"source_{src_idx}"))
        dataset = str(source.get("dataset", name))
        question_file = expand_vars(source.get("question_file"), env_map)
        image_prefix = expand_vars(source.get("image_prefix", ""), env_map)
        filter_substr = source.get("filter_substr")
        id_key = source.get("id_key")
        image_key = source.get("image_key", "image")
        question_key = source.get("question_key")
        answer_key = source.get("answer_key", "answer")
        answers_key = source.get("answers_key", "answers")
        limit = source.get("limit")
        extra_fields = source.get("extra_fields", {})
        if limit is not None:
            limit = int(limit)

        if not question_file:
            raise ValueError(f"Source '{name}' missing question_file.")

        rows = load_json_or_jsonl(question_file)
        if not isinstance(rows, list):
            raise ValueError(f"Source '{name}' question_file must be list/jsonl.")

        kept = 0
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                skipped += 1
                continue

            if image_key not in row:
                skipped += 1
                continue
            raw_question = choose_question(row, question_key=question_key)
            if raw_question is None:
                skipped += 1
                continue

            image = normalize_image_path(row[image_key], image_prefix)
            if filter_substr and filter_substr not in image:
                continue

            question_id = choose_question_id(row, id_key=id_key, fallback=f"{name}_{row_idx}")
            item = {
                "question_id": str(question_id),
                "image": image,
                "text": str(raw_question),
                "dataset": dataset,
                "source": name,
            }
            if answer_key and answer_key in row:
                item["answer"] = row[answer_key]
            if answers_key and answers_key in row:
                item["answers"] = row[answers_key]
            if isinstance(extra_fields, dict):
                item.update(extra_fields)

            all_records.append(item)
            kept += 1
            if limit is not None and kept >= limit:
                break

    summary = {
        "total": len(all_records),
        "skipped": skipped,
        "dataset_counts": dict(Counter([x.get("dataset", "unknown") for x in all_records])),
        "source_counts": dict(Counter([x.get("source", "unknown") for x in all_records])),
    }
    return all_records, summary


def save_jsonl(records, output_file):
    output_file = os.path.expanduser(os.path.expandvars(output_file))
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            rec = dict(rec)
            rec["original_line_number"] = idx
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge multi-source Stage3 json/jsonl into one universal jsonl.")
    parser.add_argument("--manifest-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records, summary = build_records(manifest)
    save_jsonl(records, args.output_file)

    print(f"Saved Stage3 universal input jsonl: {args.output_file}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
