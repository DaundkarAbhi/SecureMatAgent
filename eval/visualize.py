"""
eval/visualize.py — Chart generation for SecureMatAgent evaluation results.

Generates three charts:
  1. RAGAS metrics by domain (grouped bar chart)
  2. Tool selection confusion matrix (heatmap)
  3. Faithfulness vs answer relevancy per question (scatter plot)

All charts are saved to eval/results/charts/.

Usage:
    # From CLI (after running eval/run_eval.py):
    python eval/visualize.py \\
        --ragas-json eval/results/ragas_scores.json \\
        --tool-json  eval/results/tool_accuracy.json \\
        --output     eval/results/charts/

    # From Python:
    from eval.visualize import generate_all_charts
    generate_all_charts(
        ragas_json="eval/results/ragas_scores.json",
        tool_json="eval/results/tool_accuracy.json",
        output_dir="eval/results/charts/",
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart 1: RAGAS metrics by domain (grouped bar chart)
# ---------------------------------------------------------------------------


def _chart_ragas_by_domain(
    ragas_data: Dict[str, Any], output_dir: Path
) -> Optional[str]:
    """
    Grouped bar chart: faithfulness, answer_relevancy, keyword_overlap per domain.
    Falls back to custom metrics if RAGAS scores unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend (safe for scripts)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping domain chart")
        return None

    per_domain = ragas_data.get("per_domain", {})
    if not per_domain:
        logger.warning("No per_domain data — skipping domain chart")
        return None

    domains = list(per_domain.keys())
    ragas_used = ragas_data.get("meta", {}).get("ragas_used", False)

    if ragas_used:
        metrics = {
            "Faithfulness": [per_domain[d].get("mean_faithfulness") for d in domains],
            "Answer Relevancy": [
                per_domain[d].get("mean_answer_relevancy") for d in domains
            ],
            "Keyword Overlap": [
                per_domain[d].get("mean_keyword_overlap") for d in domains
            ],
        }
    else:
        metrics = {
            "Answer Similarity": [
                per_domain[d].get("mean_answer_similarity") for d in domains
            ],
            "Keyword Overlap": [
                per_domain[d].get("mean_keyword_overlap") for d in domains
            ],
        }

    # Replace None with 0
    for key in metrics:
        metrics[key] = [v if v is not None else 0.0 for v in metrics[key]]

    n_metrics = len(metrics)
    n_domains = len(domains)
    x = np.arange(n_domains)
    bar_width = 0.25
    offsets = (
        np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * bar_width
    )

    fig, ax = plt.subplots(figsize=(max(8, n_domains * 2), 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (label, values) in enumerate(metrics.items()):
        bars = ax.bar(
            x + offsets[i],
            values,
            bar_width * 0.9,
            label=label,
            color=colors[i % len(colors)],
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xlabel("Domain", fontsize=11)
    ax.set_ylabel("Score (0–1)", fontsize=11)
    title = "RAGAS Metrics by Domain" if ragas_used else "Custom Eval Metrics by Domain"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "ragas_by_domain.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Chart 2: Tool confusion matrix (heatmap)
# ---------------------------------------------------------------------------


def _chart_tool_confusion_matrix(
    tool_data: Dict[str, Any], output_dir: Path
) -> Optional[str]:
    """
    Heatmap of tool selection confusion matrix.
    Rows = expected tool, columns = actual (predicted) tool.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping confusion matrix chart")
        return None

    try:
        import seaborn as sns

        _has_seaborn = True
    except ImportError:
        _has_seaborn = False
        logger.info("seaborn not available — using plain matplotlib for heatmap")

    cm = tool_data.get("confusion_matrix", {})
    if not cm:
        logger.warning("No confusion matrix data — skipping heatmap")
        return None

    # Collect all tools that appear in expected or actual
    all_labels = sorted(
        set(list(cm.keys()) + [t for row in cm.values() for t in row.keys()])
    )

    import numpy as np

    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=int)
    label_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    for expected, actuals in cm.items():
        if expected not in label_idx:
            continue
        for actual, count in actuals.items():
            if actual in label_idx:
                matrix[label_idx[expected], label_idx[actual]] += count

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))

    if _has_seaborn:
        import seaborn as sns

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=all_labels,
            yticklabels=all_labels,
            linewidths=0.5,
            ax=ax,
        )
    else:
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_yticklabels(all_labels)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10)

    ax.set_xlabel("Actual Tool Used", fontsize=11)
    ax.set_ylabel("Expected Tool", fontsize=11)
    ax.set_title("Tool Selection Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()

    out_path = output_dir / "tool_confusion_matrix.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Chart 3: Faithfulness vs Answer Relevancy scatter
# ---------------------------------------------------------------------------


def _chart_faithfulness_vs_relevancy(
    ragas_data: Dict[str, Any], output_dir: Path
) -> Optional[str]:
    """
    Scatter plot of faithfulness vs answer_relevancy per question.
    Falls back to answer_similarity vs keyword_overlap if RAGAS not available.
    Points are coloured by domain.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping scatter chart")
        return None

    per_q = ragas_data.get("per_question", [])
    if not per_q:
        logger.warning("No per-question data — skipping scatter chart")
        return None

    ragas_used = ragas_data.get("meta", {}).get("ragas_used", False)

    if ragas_used:
        x_vals = [q.get("faithfulness") for q in per_q]
        y_vals = [q.get("answer_relevancy") for q in per_q]
        x_label, y_label = "Faithfulness", "Answer Relevancy"
        title = "Faithfulness vs Answer Relevancy"
    else:
        x_vals = [q.get("custom_answer_similarity") for q in per_q]
        y_vals = [q.get("custom_keyword_overlap") for q in per_q]
        x_label, y_label = "Answer Similarity (cosine)", "Keyword Overlap (F1)"
        title = "Answer Similarity vs Keyword Overlap"

    # Filter out None values
    filtered = [
        (x, y, q)
        for x, y, q in zip(x_vals, y_vals, per_q)
        if x is not None and y is not None
    ]
    if not filtered:
        logger.warning("No valid scatter data — skipping scatter chart")
        return None

    x_plot = [item[0] for item in filtered]
    y_plot = [item[1] for item in filtered]
    domains = [item[2].get("domain", "unknown") for item in filtered]
    ids = [item[2].get("id", "") for item in filtered]
    neg = [item[2].get("negative_case", False) for item in filtered]

    # Color by domain
    unique_domains = sorted(set(domains))
    palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
    domain_color = {d: palette[i % len(palette)] for i, d in enumerate(unique_domains)}
    colors = [domain_color[d] for d in domains]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot non-negatives
    for x, y, color, label, is_neg, qid in zip(
        x_plot, y_plot, colors, domains, neg, ids
    ):
        marker = "x" if is_neg else "o"
        ax.scatter(x, y, c=color, marker=marker, s=80, alpha=0.8, zorder=3)

    # Legend for domains
    for domain in unique_domains:
        ax.scatter([], [], c=domain_color[domain], marker="o", label=domain, s=60)
    ax.scatter([], [], c="gray", marker="x", label="negative case", s=60)
    ax.legend(fontsize=8, loc="lower right")

    # Annotate worst points (bottom-left quadrant)
    for x, y, qid, is_neg in zip(x_plot, y_plot, ids, neg):
        if not is_neg and x < 0.3 and y < 0.3:
            ax.annotate(
                qid, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7
            )

    # Reference lines
    ax.axhline(
        0.7, color="gray", linestyle="--", alpha=0.5, linewidth=0.8, label="_h0.7"
    )
    ax.axvline(
        0.7, color="gray", linestyle="--", alpha=0.5, linewidth=0.8, label="_v0.7"
    )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = output_dir / "faithfulness_vs_relevancy.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_all_charts(
    ragas_json: Optional[str] = None,
    tool_json: Optional[str] = None,
    output_dir: str = "eval/results/charts/",
) -> List[str]:
    """
    Generate all evaluation charts.

    Args:
        ragas_json:  Path to ragas_scores.json (or None to skip RAGAS charts).
        tool_json:   Path to tool_accuracy.json (or None to skip tool charts).
        output_dir:  Directory to save chart PNG files.

    Returns:
        List of paths to generated chart files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ragas_data: Optional[Dict] = None
    tool_data: Optional[Dict] = None

    if ragas_json and Path(ragas_json).exists():
        with open(ragas_json, encoding="utf-8") as f:
            ragas_data = json.load(f)

    if tool_json and Path(tool_json).exists():
        with open(tool_json, encoding="utf-8") as f:
            tool_data = json.load(f)

    generated: List[str] = []

    if ragas_data:
        path = _chart_ragas_by_domain(ragas_data, out_dir)
        if path:
            generated.append(path)
        path = _chart_faithfulness_vs_relevancy(ragas_data, out_dir)
        if path:
            generated.append(path)

    if tool_data:
        path = _chart_tool_confusion_matrix(tool_data, out_dir)
        if path:
            generated.append(path)

    logger.info("Generated %d chart(s) in %s", len(generated), out_dir)
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate evaluation charts for SecureMatAgent"
    )
    p.add_argument("--ragas-json", default="eval/results/ragas_scores.json")
    p.add_argument("--tool-json", default="eval/results/tool_accuracy.json")
    p.add_argument("--output", default="eval/results/charts/")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    args = _parse_args()
    generated = generate_all_charts(
        ragas_json=args.ragas_json,
        tool_json=args.tool_json,
        output_dir=args.output,
    )
    print(f"Generated {len(generated)} chart(s):")
    for p in generated:
        print(f"  {p}")
