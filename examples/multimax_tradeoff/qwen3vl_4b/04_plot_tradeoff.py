import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in [
                "max_tokens",
                "avg_accuracy",
                "avg_throughput_sps",
                "rectified_throughput_sps",
                "avg_visual_tokens",
                "rectified_avg_visual_tokens",
            ]:
                if r.get(k) in ("", None):
                    r[k] = None
                else:
                    r[k] = float(r[k]) if k != "max_tokens" else int(float(r[k]))
            rows.append(r)
    return rows


def _x_throughput(r):
    return r["rectified_throughput_sps"] if r.get("rectified_throughput_sps") is not None else r.get(
        "avg_throughput_sps")


def _x_visual_tokens(r):
    return r["rectified_avg_visual_tokens"] if r.get("rectified_avg_visual_tokens") is not None else r.get(
        "avg_visual_tokens")


def _find_setting_row(data, variant, max_tokens):
    for r in data:
        if r.get("variant") == variant and int(r.get("max_tokens")) == int(max_tokens):
            return r
    return None


def _annotate_reference_arrows(ax_tp, ax_vt, data, group_name):
    target_max = 1024 if group_name == "doc_ocr" else 576
    base_row = _find_setting_row(data, "baseline", 4096)
    ours_row = _find_setting_row(data, "stage3", target_max)
    if base_row is None or ours_row is None:
        return

    bx_tp, by = _x_throughput(base_row), base_row["avg_accuracy"]
    ox_tp, oy = _x_throughput(ours_row), ours_row["avg_accuracy"]
    bx_vt = _x_visual_tokens(base_row)
    ox_vt = _x_visual_tokens(ours_row)

    arrow_props = dict(arrowstyle="->", linestyle="--", linewidth=1.5, color="#34495e")

    if bx_tp and ox_tp and by is not None and oy is not None:
        ax_tp.annotate("", xy=(ox_tp, oy), xytext=(bx_tp, by), arrowprops=arrow_props)
        speedup = (ox_tp / bx_tp) if bx_tp > 0 else None
        if speedup is not None:
            mx, my = (bx_tp + ox_tp) / 2.0, (by + oy) / 2.0
            ax_tp.text(mx, my + 0.15, f"{speedup:.2f}x speedup", fontsize=16, color="#34495e", ha="center", va="bottom")

    if bx_vt and ox_vt and by is not None and oy is not None:
        ax_vt.annotate("", xy=(ox_vt, oy), xytext=(bx_vt, by), arrowprops=arrow_props)
        reduction = (1.0 - (ox_vt / bx_vt)) if bx_vt > 0 else None
        if reduction is not None:
            mx, my = (bx_vt + ox_vt) / 2.0, (by + oy) / 2.0
            ax_vt.text(mx, my + 0.15, f"{reduction * 100:.1f}% token reduction", fontsize=16, color="#34495e",
                       ha="center", va="bottom")


def plot_group(rows, group_name, out_path):
    data = [r for r in rows if r["group"] == group_name]
    if not data:
        return

    # Updated styles to match target image: Dashed lines for baseline, Stars for Ours
    styles = {
        "baseline": {"color": "#7f8c8d", "marker": "o", "linestyle": "--", "markersize": 10},
        "stage1": {"color": "#1f77b4", "marker": "s", "linestyle": "-", "markersize": 10},
        "stage3": {"color": "#e74c3c", "marker": "*", "linestyle": "-", "markersize": 15},
    }
    label_map = {
        "baseline": "Baseline",
        "stage1": "SD-RPN",
        "stage3": "Ours",
    }

    # Updated font configurations for a cleaner, bolder look
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.labelweight": "bold",
            "axes.titlesize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for variant in ["baseline", "stage1", "stage3"]:
        sub = [r for r in data if r["variant"] == variant]
        sub = sorted(sub, key=lambda x: x["max_tokens"])
        if not sub:
            continue
        s = styles.get(variant, {"color": "black", "marker": "o", "linestyle": "-", "markersize": 8})

        x1_rectified = [_x_throughput(r) for r in sub]
        x2 = [_x_visual_tokens(r) for r in sub]
        y = [r["avg_accuracy"] for r in sub]
        labels = [str(r["max_tokens"]) for r in sub]

        axes[0].plot(
            x1_rectified, y,
            marker=s["marker"], color=s["color"], linestyle=s["linestyle"],
            label=label_map.get(variant, variant), linewidth=2.5, markersize=s["markersize"]
        )
        axes[1].plot(
            x2, y,
            marker=s["marker"], color=s["color"], linestyle=s["linestyle"],
            label=label_map.get(variant, variant), linewidth=2.5, markersize=s["markersize"]
        )

        # Updated data labels to include "Max-" and use bold font
        for xi, yi, lb in zip(x1_rectified, y, labels):
            axes[0].annotate(f"Max-{lb}", (xi, yi), textcoords="offset points", xytext=(0, 12),
                             ha="center", fontsize=14, fontweight="bold")
        for xi, yi, lb in zip(x2, y, labels):
            axes[1].annotate(f"Max-{lb}", (xi, yi), textcoords="offset points", xytext=(0, 12),
                             ha="center", fontsize=14, fontweight="bold")

    _annotate_reference_arrows(axes[0], axes[1], data, group_name)

    # Titles updated to remove "Rectified"
    # axes[0].set_title("Accuracy vs. Throughput")
    axes[0].set_xlabel("Throughput (samples/s)")
    axes[0].set_ylabel("Average Accuracy (%)")

    # axes[1].set_title("Accuracy vs. Visual Cost")
    axes[1].set_xlabel("Visual Tokens")
    axes[1].set_ylabel("Average Accuracy (%)")

    # Clean dotted grid and light borders to match the target image
    for ax in axes:
        ax.grid(True, linestyle=":", color="gray", alpha=0.4)
        ax.legend(loc="best", frameon=True, edgecolor="#cccccc")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot doc_ocr and hr tradeoff figures.")
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.global_csv)

    plot_group(rows, "doc_ocr", args.output_dir / "doc_ocr_tradeoff.png")
    plot_group(rows, "hr", args.output_dir / "hr_tradeoff.png")


if __name__ == "__main__":
    main()
