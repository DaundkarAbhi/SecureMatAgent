"""
eval/run_eval.py — CLI entry point for SecureMatAgent evaluation pipeline.

Usage:
    # Full evaluation (30 questions, RAGAS + tool accuracy)
    python eval/run_eval.py --test-set eval/test_set.json --output eval/results/

    # Quick eval (10-question subset, no RAGAS)
    python eval/run_eval.py --test-set eval/test_set.json --output eval/results/ --quick

    # Custom subset
    python eval/run_eval.py --test-set eval/test_set.json --output eval/results/ \\
        --ids MS-001 MS-003 CY-001

    # Skip RAGAS (custom eval only, much faster)
    python eval/run_eval.py --test-set eval/test_set.json --output eval/results/ --no-ragas

    # Skip tool accuracy eval (RAGAS only)
    python eval/run_eval.py --test-set eval/test_set.json --output eval/results/ --no-tool-eval

    # Generate charts after running
    python eval/run_eval.py --test-set eval/test_set.json --output eval/results/ --visualize
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path and set LOCAL_DEV
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("LOCAL_DEV", "true")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.run_eval")


# ---------------------------------------------------------------------------
# Quick-mode question IDs (10 representative questions)
# ---------------------------------------------------------------------------
QUICK_IDS = [
    "MS-001",  # easy factual retrieval
    "MS-003",  # calculator
    "MS-007",  # anomaly detection
    "MS-010",  # hard calculator
    "MS-015",  # negative case
    "CY-001",  # NIST access control
    "CY-003",  # incident response
    "CY-010",  # negative case
    "XD-001",  # cross-domain
    "XD-005",  # cross-domain negative
]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _ragas_results_to_dict(results) -> dict:
    """Convert EvalResults dataclass to a JSON-serialisable dict."""

    def _opt(v):
        return round(v, 4) if v is not None else None

    def _sr_to_dict(sr) -> dict:
        return {
            "id": sr.question_id,
            "question": sr.question[:120],
            "domain": sr.domain,
            "difficulty": sr.difficulty,
            "negative_case": sr.negative_case,
            "expected_tool": sr.expected_tool,
            "tools_used": sr.tools_used,
            "latency_ms": round(sr.latency_ms, 1),
            "faithfulness": _opt(sr.faithfulness),
            "answer_relevancy": _opt(sr.answer_relevancy),
            "context_precision": _opt(sr.context_precision),
            "context_recall": _opt(sr.context_recall),
            "custom_answer_similarity": _opt(sr.custom_answer_similarity),
            "custom_keyword_overlap": _opt(sr.custom_keyword_overlap),
            "custom_context_coverage": _opt(sr.custom_context_coverage),
            "error": sr.error,
        }

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "ragas_used": results.ragas_used,
            "ragas_available": results.ragas_available,
            "total_questions": results.total_questions,
            "evaluated": results.evaluated,
            "skipped": results.skipped,
            "duration_seconds": round(results.duration_seconds, 1),
        },
        "aggregate": {
            "faithfulness": _opt(results.mean_faithfulness),
            "answer_relevancy": _opt(results.mean_answer_relevancy),
            "context_precision": _opt(results.mean_context_precision),
            "context_recall": _opt(results.mean_context_recall),
            "custom_answer_similarity": _opt(results.mean_custom_answer_similarity),
            "custom_keyword_overlap": _opt(results.mean_custom_keyword_overlap),
        },
        "per_domain": results.per_domain,
        "per_difficulty": results.per_difficulty,
        "per_question": [_sr_to_dict(r) for r in results.results],
        "errors": results.errors,
    }


# ---------------------------------------------------------------------------
# Summary markdown generator
# ---------------------------------------------------------------------------


def _generate_summary_md(
    ragas_dict: Optional[dict],
    tool_dict: Optional[dict],
    output_dir: Path,
    run_timestamp: str,
) -> str:
    """Generate a Markdown summary report from evaluation results."""

    lines: List[str] = []
    lines.append("# SecureMatAgent — Evaluation Summary")
    lines.append(f"\n_Generated: {run_timestamp}_\n")

    # ---- RAGAS scores ----
    lines.append("---\n")
    lines.append("## RAGAS Scores\n")

    if ragas_dict:
        meta = ragas_dict["meta"]
        agg = ragas_dict["aggregate"]
        ragas_used = meta.get("ragas_used", False)

        if ragas_used:
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for metric in [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]:
                val = agg.get(metric)
                score_str = f"{val:.3f}" if val is not None else "N/A"
                lines.append(f"| {metric.replace('_', ' ').title()} | {score_str} |")
        else:
            lines.append(
                "> **Note:** RAGAS evaluation was not run "
                "(Ollama/qwen2.5:7b unavailable or `--no-ragas` flag set). "
                "Showing custom eval scores only.\n"
            )
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            lines.append(
                f"| Answer Similarity (cosine) | {agg.get('custom_answer_similarity', 'N/A')} |"
            )
            lines.append(
                f"| Keyword Overlap (F1) | {agg.get('custom_keyword_overlap', 'N/A')} |"
            )

        lines.append(
            f"\n_Evaluated {meta['evaluated']}/{meta['total_questions']} questions "
            f"in {meta['duration_seconds']:.0f}s_\n"
        )

        # ---- Per-domain breakdown ----
        lines.append("### Per-Domain Breakdown\n")
        per_domain = ragas_dict.get("per_domain", {})
        if per_domain:
            lines.append(
                "| Domain | Count | Faithfulness | Answer Relevancy | Keyword Overlap |"
            )
            lines.append(
                "|--------|-------|-------------|-----------------|-----------------|"
            )
            for domain, metrics in per_domain.items():

                def _fmt(v):
                    return (
                        f"{v:.3f}" if isinstance(v, float) and v is not None else "N/A"
                    )

                lines.append(
                    f"| {domain} "
                    f"| {int(metrics.get('count', 0))} "
                    f"| {_fmt(metrics.get('mean_faithfulness'))} "
                    f"| {_fmt(metrics.get('mean_answer_relevancy'))} "
                    f"| {_fmt(metrics.get('mean_keyword_overlap'))} |"
                )
        else:
            lines.append("_No per-domain data available._\n")

        # ---- Top 5 worst performing questions ----
        lines.append("\n### Worst-Performing Questions (by Keyword Overlap)\n")
        per_q = ragas_dict.get("per_question", [])
        non_neg = [q for q in per_q if not q.get("negative_case")]
        sorted_q = sorted(
            non_neg, key=lambda q: (q.get("custom_keyword_overlap") or 0.0)
        )
        worst = sorted_q[:5]
        if worst:
            lines.append(
                "| ID | Domain | Difficulty | Keyword Overlap | Faithfulness |"
            )
            lines.append(
                "|----|--------|------------|-----------------|--------------|"
            )
            for q in worst:
                ko = q.get("custom_keyword_overlap")
                fa = q.get("faithfulness")
                ko_str = f"{ko:.3f}" if ko is not None else "N/A"
                fa_str = f"{fa:.3f}" if fa is not None else "N/A"
                lines.append(
                    f"| {q['id']} | {q['domain']} | {q['difficulty']} "
                    f"| {ko_str} | {fa_str} |"
                )
        else:
            lines.append("_No per-question data available._\n")

    else:
        lines.append("_RAGAS evaluation not run._\n")

    # ---- Tool accuracy ----
    lines.append("\n---\n")
    lines.append("## Tool Selection Accuracy\n")

    if tool_dict:
        summary = tool_dict.get("summary", {})
        lines.append(
            f"**Overall accuracy:** {summary.get('accuracy', 0):.1%}  "
            f"({summary.get('correct', 0)}/{summary.get('total_questions', 0)})\n"
        )
        lines.append(
            f"- Positive cases: {summary.get('positive_case_accuracy', 0):.1%}"
        )
        lines.append(
            f"- Negative cases (abstain): {summary.get('negative_case_accuracy', 0):.1%}\n"
        )

        # Per-domain accuracy
        per_domain_acc = tool_dict.get("per_domain_accuracy", {})
        if per_domain_acc:
            lines.append("| Domain | Accuracy |")
            lines.append("|--------|----------|")
            for domain, acc in per_domain_acc.items():
                lines.append(f"| {domain} | {acc:.1%} |")
            lines.append("")

        # Per-tool metrics
        ptm = tool_dict.get("per_tool_metrics", {})
        if ptm:
            lines.append("\n### Per-Tool Precision / Recall / F1\n")
            lines.append("| Tool | Precision | Recall | F1 |")
            lines.append("|------|-----------|--------|----|")
            for tool, m in ptm.items():
                lines.append(
                    f"| {tool} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
                )

        # Confusion matrix (text)
        cm = tool_dict.get("confusion_matrix", {})
        if cm:
            lines.append("\n### Tool Confusion Matrix\n")
            lines.append("```")
            all_tools_used = sorted(
                set(list(cm.keys()) + [t for v in cm.values() for t in v.keys()])
            )
            # Header
            header_row = "Expected\\Actual".ljust(24) + " ".join(
                t[:13].ljust(14) for t in all_tools_used
            )
            lines.append(header_row)
            lines.append("-" * len(header_row))
            for exp in all_tools_used:
                row = exp[:22].ljust(24)
                for act in all_tools_used:
                    count = cm.get(exp, {}).get(act, 0)
                    row += str(count).ljust(14) + " "
                lines.append(row.rstrip())
            lines.append("```\n")

    else:
        lines.append("_Tool accuracy evaluation not run._\n")

    # ---- Charts ----
    charts_dir = output_dir / "charts"
    chart_files = list(charts_dir.glob("*.png")) if charts_dir.exists() else []
    if chart_files:
        lines.append("\n---\n")
        lines.append("## Charts\n")
        for cf in sorted(chart_files):
            lines.append(f"![{cf.stem}](charts/{cf.name})")
        lines.append("")

    # ---- Improvement recommendations ----
    lines.append("\n---\n")
    lines.append("## Improvement Recommendations\n")

    recs: List[str] = []

    if tool_dict:
        acc = tool_dict.get("summary", {}).get("accuracy", 1.0)
        if acc < 0.7:
            recs.append(
                "**Tool selection accuracy is below 70%.** Consider refining tool docstrings "
                "so the LLM can distinguish tool purposes more reliably. "
                "Check if the agent is defaulting to `document_search` for calculator questions."
            )
        neg_acc = tool_dict.get("summary", {}).get("negative_case_accuracy", 1.0)
        if neg_acc < 0.5:
            recs.append(
                "**Negative case accuracy is low.** The agent is using tools for questions "
                "outside its knowledge base. Add explicit 'I don't know' instructions to the "
                "system prompt."
            )
        # Check per-tool recall
        for tool, m in tool_dict.get("per_tool_metrics", {}).items():
            if (
                m.get("recall", 1.0) < 0.5
                and (m["true_positives"] + m["false_negatives"]) > 1
            ):
                recs.append(
                    f"**Low recall for `{tool}` ({m['recall']:.1%}).** "
                    f"The agent often uses a different tool when `{tool}` is expected. "
                    f"Review the tool description and few-shot examples."
                )

    if ragas_dict:
        agg = ragas_dict.get("aggregate", {})
        faith = agg.get("faithfulness")
        if faith is not None and faith < 0.7:
            recs.append(
                f"**Faithfulness is {faith:.2f} (< 0.70).** The agent is generating answers "
                "not grounded in retrieved context. Consider reducing temperature, adding "
                "explicit grounding instructions, or increasing `top_k`."
            )
        ar = agg.get("answer_relevancy")
        if ar is not None and ar < 0.7:
            recs.append(
                f"**Answer relevancy is {ar:.2f} (< 0.70).** Answers are drifting from the "
                "question. Review the ReAct prompt template and consider adding answer "
                "format constraints."
            )
        cr = agg.get("context_recall")
        if cr is not None and cr < 0.6:
            recs.append(
                f"**Context recall is {cr:.2f} (< 0.60).** The retriever is missing relevant "
                "chunks. Consider increasing `TOP_K`, re-tuning chunk size, or using a "
                "higher-dimensional embedding model."
            )

    if not recs:
        recs.append(
            "All metrics are within acceptable ranges. Continue monitoring as the corpus grows."
        )

    for rec in recs:
        lines.append(f"- {rec}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SecureMatAgent RAGAS + Tool Accuracy Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        default="eval/test_set.json",
        help="Path to the JSON test set (default: eval/test_set.json)",
    )
    parser.add_argument(
        "--output",
        default="eval/results/",
        help="Output directory for results (default: eval/results/)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Run only the {len(QUICK_IDS)}-question quick subset",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        metavar="ID",
        help="Run only specific question IDs (e.g. --ids MS-001 CY-003)",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS evaluation (use custom metrics only — much faster)",
    )
    parser.add_argument(
        "--no-tool-eval",
        action="store_true",
        help="Skip tool accuracy evaluation",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate charts after evaluation (requires matplotlib + seaborn)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question for RAGAS context (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "charts").mkdir(exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info("SecureMatAgent Evaluation Pipeline")
    logger.info("Timestamp : %s", run_timestamp)
    logger.info("Test set  : %s", args.test_set)
    logger.info("Output    : %s", output_dir)
    logger.info("=" * 60)

    # Determine question IDs to run
    question_ids: Optional[List[str]] = None
    if args.quick:
        question_ids = QUICK_IDS
        logger.info("Quick mode: %d questions", len(QUICK_IDS))
    elif args.ids:
        question_ids = args.ids
        logger.info("Custom subset: %s", question_ids)

    ragas_dict: Optional[dict] = None
    tool_dict: Optional[dict] = None

    # ---- RAGAS Evaluation ----
    if not args.no_ragas:
        logger.info("\n[1/2] Running RAGAS + custom evaluation…")
        from eval.evaluator import run_evaluation

        ragas_results = run_evaluation(
            test_set_path=args.test_set,
            question_ids=question_ids,
            use_ragas=True,
            top_k_contexts=args.top_k,
        )
        ragas_dict = _ragas_results_to_dict(ragas_results)
        ragas_path = output_dir / "ragas_scores.json"
        with open(ragas_path, "w", encoding="utf-8") as f:
            json.dump(ragas_dict, f, indent=2, default=str)
        logger.info("Saved RAGAS scores → %s", ragas_path)
    else:
        logger.info("[1/2] RAGAS evaluation skipped (--no-ragas)")
        # Still run custom eval
        logger.info("      Running custom metrics only…")
        from eval.evaluator import run_evaluation

        ragas_results = run_evaluation(
            test_set_path=args.test_set,
            question_ids=question_ids,
            use_ragas=False,
            top_k_contexts=args.top_k,
        )
        ragas_dict = _ragas_results_to_dict(ragas_results)
        ragas_path = output_dir / "ragas_scores.json"
        with open(ragas_path, "w", encoding="utf-8") as f:
            json.dump(ragas_dict, f, indent=2, default=str)
        logger.info("Saved custom scores → %s", ragas_path)

    # ---- Tool Accuracy Evaluation ----
    if not args.no_tool_eval:
        logger.info("\n[2/2] Running tool accuracy evaluation…")
        from eval.tool_accuracy import (run_tool_accuracy_eval,
                                        tool_accuracy_to_dict)

        tool_results = run_tool_accuracy_eval(
            test_set_path=args.test_set,
            question_ids=question_ids,
        )
        tool_dict = tool_accuracy_to_dict(tool_results)
        tool_path = output_dir / "tool_accuracy.json"
        with open(tool_path, "w", encoding="utf-8") as f:
            json.dump(tool_dict, f, indent=2, default=str)
        logger.info("Saved tool accuracy → %s", tool_path)
        logger.info(
            "Tool selection accuracy: %.1f%% (%d/%d)",
            tool_results.accuracy * 100,
            tool_results.correct,
            tool_results.total_questions,
        )
    else:
        logger.info("[2/2] Tool accuracy evaluation skipped (--no-tool-eval)")

    # ---- Summary Markdown ----
    logger.info("\nGenerating summary.md…")
    summary_md = _generate_summary_md(ragas_dict, tool_dict, output_dir, run_timestamp)
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info("Saved summary → %s", summary_path)

    # ---- Charts ----
    if args.visualize:
        logger.info("\nGenerating charts…")
        try:
            from eval.visualize import generate_all_charts

            generate_all_charts(
                ragas_json=(
                    str(output_dir / "ragas_scores.json") if ragas_dict else None
                ),
                tool_json=str(output_dir / "tool_accuracy.json") if tool_dict else None,
                output_dir=str(output_dir / "charts"),
            )
            logger.info("Charts saved to %s/charts/", output_dir)
        except ImportError as e:
            logger.warning(
                "Could not generate charts: %s — install matplotlib + seaborn", e
            )

    # ---- Final summary to console ----
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info("Results saved to: %s", output_dir)
    if ragas_dict:
        agg = ragas_dict.get("aggregate", {})
        logger.info("  Faithfulness    : %s", agg.get("faithfulness", "N/A"))
        logger.info("  Answer Relevancy: %s", agg.get("answer_relevancy", "N/A"))
        logger.info("  Context Precision: %s", agg.get("context_precision", "N/A"))
        logger.info("  Context Recall  : %s", agg.get("context_recall", "N/A"))
        logger.info("  Keyword Overlap : %s", agg.get("custom_keyword_overlap", "N/A"))
    if tool_dict:
        acc = tool_dict.get("summary", {}).get("accuracy")
        if acc is not None:
            logger.info("  Tool Accuracy   : %.1f%%", acc * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
