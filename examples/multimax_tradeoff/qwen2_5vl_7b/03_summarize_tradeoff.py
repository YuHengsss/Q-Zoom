import argparse
import csv
import glob
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path


DOC_TASKS = ["docvqa_val", "chartqa", "ocrbench", "infovqa_val", "textvqa_val"]
HR_TASKS = ["vstar_bench", "mmerealworld_lite", "hrbench4k", "hrbench8k"]
TASK_TOTAL_SAMPLES = {
    "docvqa_val": 5301,
    "infovqa_val": 2801,
    "ocrbench": 1000,
    "textvqa_val": 5000,
    "chartqa": 2500,
    "vstar_bench": 191,
    "mmerealworld_lite": 1920,
    "hrbench": 1600,
    "hrbench4k": 800,
    "hrbench8k": 800,
}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_pct(v):
    if v is None:
        return None
    x = float(v)
    return x * 100.0 if x <= 1.0 else x


def aggregate_pkl(files):
    sample_count = 0
    total_latency = 0.0
    total_tokens = 0
    vt_sum = 0.0
    vt_count = 0
    for fp in files:
        with open(fp, "rb") as f:
            data = pickle.load(f)
        metric = data.get("metric_dict", {})
        total_latency += float(metric.get("e2e_latency", 0.0))
        total_tokens += int(metric.get("total_tokens", 0))
        for k, v in data.items():
            if k == "metric_dict":
                continue
            sample_count += 1
            if isinstance(v, dict) and "visual_token_num" in v:
                vv = v["visual_token_num"]
                if hasattr(vv, "item"):
                    vv = vv.item()
                vt_sum += float(vv)
                vt_count += 1
    return {
        "samples": sample_count,
        "latency": total_latency,
        "tokens": total_tokens,
        "sps": (sample_count / total_latency) if total_latency > 0 else 0.0,
        "tps": (total_tokens / total_latency) if total_latency > 0 else 0.0,
        "avg_visual_tokens": (vt_sum / vt_count) if vt_count > 0 else 0.0,
    }


def parse_perf_metrics(task, result_obj):
    res = result_obj.get("results", {})
    if task == "docvqa_val":
        return [{"metric_task": task, "acc": to_pct(res.get(task, {}).get("anls,none"))}]
    if task == "chartqa":
        return [{"metric_task": task, "acc": to_pct(res.get(task, {}).get("relaxed_overall,none"))}]
    if task == "ocrbench":
        return [{"metric_task": task, "acc": to_pct(res.get(task, {}).get("ocrbench_accuracy,none"))}]
    if task == "infovqa_val":
        return [{"metric_task": task, "acc": to_pct(res.get(task, {}).get("anls,none"))}]
    if task == "textvqa_val":
        return [{"metric_task": task, "acc": to_pct(res.get(task, {}).get("exact_match,none"))}]
    if task == "vstar_bench":
        return [{"metric_task": task, "acc": to_pct(res.get(task, {}).get("vstar_overall_acc,none"))}]
    if task == "mmerealworld_lite":
        return [{"metric_task": task, "acc": to_pct(res.get(task, {}).get("mme_realworld_score,none"))}]
    if task == "hrbench":
        return [
            {"metric_task": "hrbench4k", "acc": to_pct(res.get("hrbench4k", {}).get("average,none"))},
            {"metric_task": "hrbench8k", "acc": to_pct(res.get("hrbench8k", {}).get("average,none"))},
        ]
    return []


def collect_tp_stats(tp_row):
    pkl_dir = Path(tp_row["pkl_dir"])
    all_pkls = sorted(glob.glob(str(pkl_dir / "*.pkl")))
    if not all_pkls:
        return {"default": {"sps": 0.0, "avg_visual_tokens": 0.0}}

    out = {"default": aggregate_pkl(all_pkls)}
    if tp_row["task"] == "hrbench":
        p4 = [p for p in all_pkls if "hrbench4k_" in os.path.basename(p)]
        p8 = [p for p in all_pkls if "hrbench8k_" in os.path.basename(p)]
        if p4:
            out["hrbench4k"] = aggregate_pkl(p4)
        if p8:
            out["hrbench8k"] = aggregate_pkl(p8)
    return out


def expected_tp_min_tokens(variant: str, task: str, perf_min_tokens: int, max_tokens: int) -> int:
    """Mirror min_t policy in 02_run_throughput_sdpa.sh."""
    min_t = int(perf_min_tokens)
    # if int(max_tokens) == 256:
    #     min_t = 128
    # if variant == "stage3" and task == "ocrbench":
    #     min_t = 128
    return min_t


def main():
    parser = argparse.ArgumentParser(description="Summarize performance/visual-cost/throughput tradeoff results.")
    parser.add_argument("--perf-manifest", type=Path, required=True)
    parser.add_argument("--tp-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    perf_rows = load_jsonl(args.perf_manifest)
    tp_rows = load_jsonl(args.tp_manifest)

    tp_map = {}
    tp_candidates = defaultdict(list)
    for r in tp_rows:
        key = (r["variant"], int(r["min_tokens"]), int(r["max_tokens"]), r["task"])
        stats = collect_tp_stats(r)
        tp_map[key] = stats
        tp_candidates[(r["variant"], int(r["max_tokens"]), r["task"])].append(
            {"min_tokens": int(r["min_tokens"]), "stats": stats}
        )

    detailed = []
    miss_count = 0
    fallback_count = 0
    for r in perf_rows:
        perf_min = int(r["min_tokens"])
        max_t = int(r["max_tokens"])
        exp_min = expected_tp_min_tokens(r["variant"], r["task"], perf_min, max_t)

        key_expected = (r["variant"], exp_min, max_t, r["task"])
        key_perf = (r["variant"], perf_min, max_t, r["task"])

        tp_stat = None
        tp_min_used = None
        match_status = "none"

        if key_expected in tp_map:
            tp_stat = tp_map[key_expected]
            tp_min_used = exp_min
            match_status = "expected_min_match"
        elif key_perf in tp_map:
            tp_stat = tp_map[key_perf]
            tp_min_used = perf_min
            match_status = "perf_min_match"
        else:
            cands = tp_candidates.get((r["variant"], max_t, r["task"]), [])
            if len(cands) == 1:
                tp_stat = cands[0]["stats"]
                tp_min_used = cands[0]["min_tokens"]
                match_status = "fallback_unique_variant_max_task"
                fallback_count += 1
            elif len(cands) > 1:
                # Prefer expected min if present among candidates.
                chosen = [c for c in cands if c["min_tokens"] == exp_min]
                if chosen:
                    tp_stat = chosen[0]["stats"]
                    tp_min_used = chosen[0]["min_tokens"]
                    match_status = "fallback_multi_choose_expected_min"
                    fallback_count += 1

        if tp_stat is None:
            tp_stat = {"default": {"sps": 0.0, "avg_visual_tokens": 0.0}}
            tp_min_used = None
            match_status = "missing_tp_default_zero"
            miss_count += 1

        result_obj = json.loads(Path(r["results_file"]).read_text(encoding="utf-8"))
        metrics = parse_perf_metrics(r["task"], result_obj)

        for m in metrics:
            metric_task = m["metric_task"]
            stat = tp_stat.get(metric_task, tp_stat["default"])
            group = "doc_ocr" if metric_task in DOC_TASKS else "hr"
            detailed.append(
                {
                    "variant": r["variant"],
                    "min_tokens": perf_min,
                    "max_tokens": max_t,
                    "tp_min_tokens_used": tp_min_used,
                    "tp_match_status": match_status,
                    "task": metric_task,
                    "group": group,
                    "accuracy": m["acc"],
                    "throughput_sps": stat.get("sps", 0.0),
                    "avg_visual_tokens": stat.get("avg_visual_tokens", 0.0),
                    "checkpoint": r["checkpoint"],
                }
            )

    global_rows = []
    seen = sorted(set((d["variant"], d["max_tokens"]) for d in detailed))
    for variant, max_t in seen:
        for group, tasks in [("doc_ocr", DOC_TASKS), ("hr", HR_TASKS)]:
            subset = [d for d in detailed if d["variant"] == variant and d["max_tokens"] == max_t and d["task"] in tasks]
            if not subset:
                continue
            acc = [x["accuracy"] for x in subset if x["accuracy"] is not None]
            sps = [x["throughput_sps"] for x in subset if x["throughput_sps"] is not None]
            vtok = [x["avg_visual_tokens"] for x in subset if x["avg_visual_tokens"] is not None]
            rect_num = 0.0
            rect_den = 0.0
            rect_vtok_num = 0.0
            rect_vtok_den = 0.0
            for x in subset:
                task_name = x["task"]
                sps_i = x["throughput_sps"]
                if sps_i is None or float(sps_i) <= 0:
                    continue
                n_i = float(TASK_TOTAL_SAMPLES.get(task_name, 0))
                if n_i <= 0:
                    continue
                rect_num += n_i
                rect_den += n_i / float(sps_i)
                vtok_i = x.get("avg_visual_tokens")
                if vtok_i is not None:
                    rect_vtok_num += n_i * float(vtok_i)
                    rect_vtok_den += n_i
            rectified_sps = (rect_num / rect_den) if rect_den > 0 else None
            rectified_vtok = (rect_vtok_num / rect_vtok_den) if rect_vtok_den > 0 else None
            global_rows.append(
                {
                    "variant": variant,
                    "max_tokens": max_t,
                    "group": group,
                    "avg_accuracy": sum(acc) / len(acc) if acc else None,
                    "avg_throughput_sps": sum(sps) / len(sps) if sps else None,
                    "rectified_throughput_sps": rectified_sps,
                    "avg_visual_tokens": sum(vtok) / len(vtok) if vtok else None,
                    "rectified_avg_visual_tokens": rectified_vtok,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_json = args.output_dir / "detailed_tradeoff.json"
    detailed_csv = args.output_dir / "detailed_tradeoff.csv"
    global_json = args.output_dir / "global_tradeoff.json"
    global_csv = args.output_dir / "global_tradeoff.csv"

    detailed_json.write_text(json.dumps({"rows": detailed}, ensure_ascii=False, indent=2), encoding="utf-8")
    global_json.write_text(json.dumps({"rows": global_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "min_tokens",
                "max_tokens",
                "tp_min_tokens_used",
                "tp_match_status",
                "task",
                "group",
                "accuracy",
                "throughput_sps",
                "avg_visual_tokens",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed)

    with global_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "max_tokens",
                "group",
                "avg_accuracy",
                "avg_throughput_sps",
                "rectified_throughput_sps",
                "avg_visual_tokens",
                "rectified_avg_visual_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(global_rows)

    print(f"Saved: {detailed_json}")
    print(f"Saved: {detailed_csv}")
    print(f"Saved: {global_json}")
    print(f"Saved: {global_csv}")
    print(f"[TradeoffSummary] fallback matches used: {fallback_count}")
    print(f"[TradeoffSummary] missing TP rows defaulted to zero: {miss_count}")


if __name__ == "__main__":
    main()
