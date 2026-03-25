"""
eval/tool_accuracy.py — Tool selection accuracy evaluation for SecureMatAgent.

Runs each test question through the agent, compares the actual tool(s) used
against the expected_tool from the test set, and computes:
  - Overall tool selection accuracy
  - Per-tool precision and recall
  - Confusion matrix
  - Negative case accuracy (agent correctly declines to use tools)

Usage:
    from eval.tool_accuracy import run_tool_accuracy_eval
    results = run_tool_accuracy_eval("eval/test_set.json")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("LOCAL_DEV", "true")

# All valid tool names in SecureMatAgent
ALL_TOOLS = [
    "document_search",
    "materials_calculator",
    "web_search",
    "data_anomaly_checker",
    "data_extractor",
    "none",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolPrediction:
    question_id: str
    question: str
    domain: str
    difficulty: str
    negative_case: bool
    expected_tool: str
    actual_tools: List[str]  # all tools the agent actually called
    primary_tool: str  # first tool called, or "none"
    correct: bool  # primary_tool == expected_tool
    latency_ms: float
    agent_answer: str
    error: Optional[str] = None


@dataclass
class PerToolMetrics:
    tool: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class ToolAccuracyResults:
    total_questions: int
    correct: int
    accuracy: float
    negative_case_accuracy: float  # accuracy on negative cases only
    positive_case_accuracy: float  # accuracy on non-negative cases only
    per_tool_metrics: Dict[str, PerToolMetrics]
    confusion_matrix: Dict[str, Dict[str, int]]  # [expected][actual] -> count
    per_domain_accuracy: Dict[str, float]
    per_difficulty_accuracy: Dict[str, float]
    predictions: List[ToolPrediction]
    duration_seconds: float
    errors: List[str] = field(default_factory=list)

    def confusion_matrix_as_table(self) -> str:
        """Return a text table of the confusion matrix."""
        tools = sorted(self.confusion_matrix.keys())
        header = "Expected\\Actual".ljust(24) + "  ".join(
            t[:12].ljust(14) for t in tools
        )
        rows = [header, "-" * len(header)]
        for expected in tools:
            row = expected[:22].ljust(24)
            for actual in tools:
                count = self.confusion_matrix.get(expected, {}).get(actual, 0)
                row += str(count).ljust(14) + "  "
            rows.append(row.rstrip())
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


def _run_agent(question: str, session_id: str) -> Dict[str, Any]:
    try:
        from src.agent.run import ask

        return ask(question, session_id=session_id)
    except Exception as exc:
        logger.error("Agent error for %r: %s", question[:60], exc)
        return {
            "answer": f"[agent_error] {exc}",
            "tools_used": [],
            "sources": [],
            "latency_ms": 0.0,
            "_error": str(exc),
        }


def _normalise_tool(tool_name: str) -> str:
    """Normalise tool name to match test-set expected_tool values."""
    name = tool_name.strip().lower()
    # Handle aliases
    aliases = {
        "knowledge_base_search": "document_search",
        "kb_search": "document_search",
        "calc": "materials_calculator",
        "calculator": "materials_calculator",
        "anomaly": "data_anomaly_checker",
        "anomaly_checker": "data_anomaly_checker",
        "extractor": "data_extractor",
        "search": "web_search",
    }
    return aliases.get(name, name)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def run_tool_accuracy_eval(
    test_set_path: str = "eval/test_set.json",
    question_ids: Optional[List[str]] = None,
) -> ToolAccuracyResults:
    """
    Run all test questions through the agent and evaluate tool selection.

    Args:
        test_set_path:  Path to the JSON test set.
        question_ids:   If provided, restrict to these IDs (for quick mode).

    Returns:
        ToolAccuracyResults with per-question predictions and aggregate metrics.
    """
    t0 = time.perf_counter()
    path = Path(test_set_path)
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    with open(path, encoding="utf-8") as f:
        test_set = json.load(f)

    if question_ids:
        test_set = [q for q in test_set if q["id"] in question_ids]

    logger.info("Tool accuracy eval: %d questions", len(test_set))

    predictions: List[ToolPrediction] = []
    errors: List[str] = []

    for entry in test_set:
        qid = entry["id"]
        question = entry["question"]
        expected = entry.get("expected_tool", "none")
        is_negative = entry.get("negative_case", False)

        logger.info("Tool eval: %s", qid)
        session_id = f"tool_eval_{qid}"
        response = _run_agent(question, session_id=session_id)

        raw_tools = response.get("tools_used", [])
        actual_tools = [_normalise_tool(t) for t in raw_tools if t]
        primary_tool = actual_tools[0] if actual_tools else "none"

        error = response.get("_error")
        if error:
            errors.append(f"{qid}: {error}")

        # For negative cases: correct if agent used "none" (called no tools)
        correct = primary_tool == expected

        predictions.append(
            ToolPrediction(
                question_id=qid,
                question=question,
                domain=entry.get("domain", "unknown"),
                difficulty=entry.get("difficulty", "unknown"),
                negative_case=is_negative,
                expected_tool=expected,
                actual_tools=actual_tools,
                primary_tool=primary_tool,
                correct=correct,
                latency_ms=response.get("latency_ms", 0.0),
                agent_answer=response.get("answer", ""),
                error=error,
            )
        )

    # ---------- Aggregate metrics ----------
    total = len(predictions)
    correct_count = sum(1 for p in predictions if p.correct)
    overall_accuracy = correct_count / total if total > 0 else 0.0

    neg_preds = [p for p in predictions if p.negative_case]
    pos_preds = [p for p in predictions if not p.negative_case]
    neg_acc = (
        sum(1 for p in neg_preds if p.correct) / len(neg_preds) if neg_preds else 0.0
    )
    pos_acc = (
        sum(1 for p in pos_preds if p.correct) / len(pos_preds) if pos_preds else 0.0
    )

    # Per-tool precision / recall
    per_tool: Dict[str, PerToolMetrics] = {t: PerToolMetrics(tool=t) for t in ALL_TOOLS}
    for pred in predictions:
        exp = pred.expected_tool if pred.expected_tool in ALL_TOOLS else "none"
        act = pred.primary_tool if pred.primary_tool in ALL_TOOLS else "none"
        if exp not in per_tool:
            per_tool[exp] = PerToolMetrics(tool=exp)
        if act not in per_tool:
            per_tool[act] = PerToolMetrics(tool=act)

        if exp == act:
            per_tool[exp].true_positives += 1
        else:
            per_tool[exp].false_negatives += 1
            per_tool[act].false_positives += 1

    # Confusion matrix: expected -> actual -> count
    cm: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred in predictions:
        exp = pred.expected_tool if pred.expected_tool in ALL_TOOLS else "none"
        act = pred.primary_tool if pred.primary_tool in ALL_TOOLS else "none"
        cm[exp][act] += 1
    # Convert to plain dicts
    cm_plain: Dict[str, Dict[str, int]] = {k: dict(v) for k, v in cm.items()}

    # Per-domain accuracy
    domain_correct: Dict[str, List[bool]] = defaultdict(list)
    for pred in predictions:
        domain_correct[pred.domain].append(pred.correct)
    per_domain_acc = {d: sum(v) / len(v) for d, v in domain_correct.items() if v}

    # Per-difficulty accuracy
    diff_correct: Dict[str, List[bool]] = defaultdict(list)
    for pred in predictions:
        diff_correct[pred.difficulty].append(pred.correct)
    per_diff_acc = {d: sum(v) / len(v) for d, v in diff_correct.items() if v}

    return ToolAccuracyResults(
        total_questions=total,
        correct=correct_count,
        accuracy=overall_accuracy,
        negative_case_accuracy=neg_acc,
        positive_case_accuracy=pos_acc,
        per_tool_metrics=per_tool,
        confusion_matrix=cm_plain,
        per_domain_accuracy=per_domain_acc,
        per_difficulty_accuracy=per_diff_acc,
        predictions=predictions,
        duration_seconds=time.perf_counter() - t0,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def tool_accuracy_to_dict(results: ToolAccuracyResults) -> Dict[str, Any]:
    """Convert ToolAccuracyResults to a JSON-serialisable dict."""

    def _metrics_dict(m: PerToolMetrics) -> Dict[str, Any]:
        return {
            "true_positives": m.true_positives,
            "false_positives": m.false_positives,
            "false_negatives": m.false_negatives,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
        }

    def _pred_dict(p: ToolPrediction) -> Dict[str, Any]:
        return {
            "id": p.question_id,
            "question": p.question[:120],
            "domain": p.domain,
            "difficulty": p.difficulty,
            "negative_case": p.negative_case,
            "expected_tool": p.expected_tool,
            "actual_tools": p.actual_tools,
            "primary_tool": p.primary_tool,
            "correct": p.correct,
            "latency_ms": round(p.latency_ms, 1),
            "error": p.error,
        }

    return {
        "summary": {
            "total_questions": results.total_questions,
            "correct": results.correct,
            "accuracy": round(results.accuracy, 4),
            "negative_case_accuracy": round(results.negative_case_accuracy, 4),
            "positive_case_accuracy": round(results.positive_case_accuracy, 4),
            "duration_seconds": round(results.duration_seconds, 1),
        },
        "per_tool_metrics": {
            tool: _metrics_dict(m)
            for tool, m in results.per_tool_metrics.items()
            if m.true_positives + m.false_positives + m.false_negatives > 0
        },
        "confusion_matrix": results.confusion_matrix,
        "per_domain_accuracy": {
            k: round(v, 4) for k, v in results.per_domain_accuracy.items()
        },
        "per_difficulty_accuracy": {
            k: round(v, 4) for k, v in results.per_difficulty_accuracy.items()
        },
        "predictions": [_pred_dict(p) for p in results.predictions],
        "errors": results.errors,
    }
