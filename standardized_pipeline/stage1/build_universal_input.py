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


ANSWER_SUFFIX = "Answer the question using a single word or phrase."


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def normalize_mode(mode, default="textual"):
    if mode is None:
        return default
    text = str(mode).strip().lower()
    if text in {"natural", "nat"}:
        return "natural"
    if text in {"textual", "text"}:
        return "textual"
    return default


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
        return json.load(f)


def load_manifest(path):
    manifest = load_json_or_jsonl(path)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest file must be a JSON object.")
    if "sources" not in manifest or not isinstance(manifest["sources"], list):
        raise ValueError("Manifest must contain a 'sources' list.")
    return manifest


def resolve_prompt(record, question_key=None):
    if question_key and question_key in record:
        return record[question_key]
    for key in ("text", "prompt", "question"):
        if key in record:
            return record[key]
    return None


def resolve_question_id(record, id_key=None, fallback_idx=0):
    if id_key and id_key in record:
        return str(record[id_key])
    for key in ("question_id", "id"):
        if key in record:
            return str(record[key])
    return str(fallback_idx)


def load_overrides(manifest):
    overrides = []

    inline = manifest.get("example_overrides", [])
    if isinstance(inline, list):
        overrides.extend(inline)
    elif inline:
        raise ValueError("'example_overrides' must be a list if provided.")

    overrides_file = manifest.get("example_overrides_file")
    if overrides_file:
        loaded = load_json_or_jsonl(overrides_file)
        if isinstance(loaded, dict):
            loaded = [loaded]
        if not isinstance(loaded, list):
            raise ValueError("example_overrides_file must contain a list/dict/jsonl.")
        overrides.extend(loaded)

    return overrides


def build_override_maps(overrides):
    by_uid = {}
    by_pair = {}
    by_qid = {}

    for item in overrides:
        if not isinstance(item, dict):
            continue
        uid = item.get("uid")
        question_id = item.get("question_id")
        dataset = item.get("dataset")
        if uid:
            by_uid[str(uid)] = item
        if question_id is not None and dataset:
            by_pair[(str(question_id), str(dataset))] = item
        if question_id is not None and not dataset:
            by_qid[str(question_id)] = item

    return by_uid, by_pair, by_qid


def apply_override(entry, by_uid, by_pair, by_qid):
    override = None
    uid = entry.get("uid")
    qid = str(entry.get("question_id"))
    dataset = str(entry.get("dataset"))

    if uid in by_uid:
        override = by_uid[uid]
    elif (qid, dataset) in by_pair:
        override = by_pair[(qid, dataset)]
    elif qid in by_qid:
        override = by_qid[qid]

    if override is None:
        return entry

    merged = dict(entry)
    for key, value in override.items():
        if key == "uid":
            continue
        merged[key] = value
    return merged


def build_universal_records(manifest):
    defaults = manifest.get("defaults", {})
    default_mode = normalize_mode(defaults.get("mode", "textual"), default="textual")
    default_short_prompt = parse_bool(defaults.get("short_prompt", True), default=True)
    default_use_system_prompt = parse_bool(defaults.get("use_system_prompt", False), default=False)
    default_max_new_tokens = int(defaults.get("max_new_tokens", 128))

    overrides = load_overrides(manifest)
    by_uid, by_pair, by_qid = build_override_maps(overrides)

    universal = []
    skipped_missing_fields = 0
    skipped_filter = 0

    for src_idx, source in enumerate(manifest["sources"]):
        if not isinstance(source, dict):
            continue

        source_name = str(source.get("name", f"source_{src_idx}"))
        dataset = str(source.get("dataset", source_name))
        question_file = source.get("question_file")
        if not question_file:
            raise ValueError(f"Source '{source_name}' missing required field: question_file")

        records = load_json_or_jsonl(question_file)
        if not isinstance(records, list):
            raise ValueError(f"Source '{source_name}' question_file must load to a list/jsonl.")

        filter_substr = source.get("filter_substr")
        image_key = source.get("image_key", "image")
        question_key = source.get("question_key")
        id_key = source.get("id_key")
        limit = source.get("limit")
        if limit is not None:
            limit = int(limit)

        source_mode = normalize_mode(source.get("mode", default_mode), default=default_mode)
        source_short_prompt = parse_bool(source.get("short_prompt", default_short_prompt), default=default_short_prompt)
        source_use_system_prompt = parse_bool(
            source.get("use_system_prompt", default_use_system_prompt),
            default=default_use_system_prompt,
        )
        source_max_new_tokens = int(source.get("max_new_tokens", default_max_new_tokens))
        allow_row_level_overrides = parse_bool(source.get("allow_row_level_overrides", True), default=True)

        extra_fields = source.get("extra_fields", {})
        if extra_fields and not isinstance(extra_fields, dict):
            raise ValueError(f"Source '{source_name}' extra_fields must be a dict.")

        local_count = 0
        for row_idx, row in enumerate(records):
            if not isinstance(row, dict):
                skipped_missing_fields += 1
                continue

            image = row.get(image_key)
            prompt = resolve_prompt(row, question_key=question_key)
            if image is None or prompt is None:
                skipped_missing_fields += 1
                continue

            image = str(image)
            if filter_substr and filter_substr not in image:
                skipped_filter += 1
                continue

            question_id = resolve_question_id(row, id_key=id_key, fallback_idx=row_idx)
            uid = f"{dataset}:{source_name}:{question_id}:{row_idx}"

            entry = {
                "uid": uid,
                "question_id": question_id,
                "dataset": dataset,
                "image": image,
                "text": str(prompt),
                "mode": source_mode,
                "short_prompt": source_short_prompt,
                "use_system_prompt": source_use_system_prompt,
                "max_new_tokens": source_max_new_tokens,
                "source": source_name,
            }
            entry.update(extra_fields)

            if allow_row_level_overrides:
                for opt_key in ("mode", "short_prompt", "use_system_prompt", "max_new_tokens"):
                    if opt_key in row:
                        entry[opt_key] = row[opt_key]

            entry = apply_override(entry, by_uid, by_pair, by_qid)

            entry["mode"] = normalize_mode(entry.get("mode", default_mode), default=default_mode)
            entry["short_prompt"] = parse_bool(entry.get("short_prompt", default_short_prompt), default_short_prompt)
            entry["use_system_prompt"] = parse_bool(
                entry.get("use_system_prompt", default_use_system_prompt), default_use_system_prompt
            )
            entry["max_new_tokens"] = int(entry.get("max_new_tokens", default_max_new_tokens))

            if not entry["text"]:
                skipped_missing_fields += 1
                continue

            universal.append(entry)
            local_count += 1
            if limit is not None and local_count >= limit:
                break

    summary = {
        "total_records": len(universal),
        "skipped_missing_fields": skipped_missing_fields,
        "skipped_by_filter": skipped_filter,
        "dataset_counts": dict(Counter([x["dataset"] for x in universal])),
        "mode_counts": dict(Counter([x["mode"] for x in universal])),
    }
    return universal, summary


def save_jsonl(records, output_file):
    output_file = os.path.expanduser(os.path.expandvars(output_file))
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build standardized universal Stage1 input JSONL.")
    parser.add_argument("--manifest-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_file)
    records, summary = build_universal_records(manifest)
    save_jsonl(records, args.output_file)

    print(f"Saved universal Stage1 input: {args.output_file}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
