"""
eval/evaluator.py — RAGAS-based evaluation for SecureMatAgent.

Configures RAGAS to use local Ollama/Mistral as evaluation LLM and
HuggingFace all-MiniLM-L6-v2 for embeddings. No OpenAI API required.

Usage:
    from eval.evaluator import run_evaluation
    results = run_evaluation("eval/test_set.json")

Fallback:
    If RAGAS is not installed or the local LLM is unavailable, the module
    falls back to custom_eval() which uses cosine similarity and keyword
    overlap — no LLM-as-judge, fully offline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Force local dev mode so Qdrant/Ollama hit localhost
os.environ.setdefault("LOCAL_DEV", "true")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SingleResult:
    question_id: str
    question: str
    domain: str
    difficulty: str
    negative_case: bool
    expected_tool: str
    # Agent outputs
    agent_answer: str = ""
    tools_used: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    # Retrieved contexts (list of raw text chunks)
    retrieved_contexts: List[str] = field(default_factory=list)
    # RAGAS scores (None if RAGAS not run / N/A)
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    # Fallback scores
    custom_answer_similarity: Optional[float] = None
    custom_keyword_overlap: Optional[float] = None
    custom_context_coverage: Optional[float] = None
    # Error flag
    error: Optional[str] = None


@dataclass
class EvalResults:
    ragas_available: bool
    ragas_used: bool
    total_questions: int
    evaluated: int
    skipped: int
    mean_faithfulness: Optional[float]
    mean_answer_relevancy: Optional[float]
    mean_context_precision: Optional[float]
    mean_context_recall: Optional[float]
    mean_custom_answer_similarity: Optional[float]
    mean_custom_keyword_overlap: Optional[float]
    per_domain: Dict[str, Dict[str, float]]
    per_difficulty: Dict[str, Dict[str, float]]
    results: List[SingleResult]
    duration_seconds: float
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Custom fallback evaluation (no LLM, no RAGAS)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re

    return re.findall(r"\b\w+\b", text.lower())


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """TF-based cosine similarity between two texts."""
    import math
    from collections import Counter

    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)
    vocab = set(freq_a) | set(freq_b)

    dot = sum(freq_a.get(w, 0) * freq_b.get(w, 0) for w in vocab)
    mag_a = math.sqrt(sum(v**2 for v in freq_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in freq_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _keyword_overlap(answer: str, ground_truth: str) -> float:
    """F1-style unigram token overlap between answer and ground_truth."""
    pred_tokens = set(_tokenize(answer))
    gold_tokens = set(_tokenize(ground_truth))
    if not pred_tokens or not gold_tokens:
        return 0.0
    intersection = pred_tokens & gold_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _context_coverage(contexts: List[str], ground_truth: str) -> float:
    """Average similarity between contexts and ground truth (proxy for recall)."""
    if not contexts:
        return 0.0
    sims = [_cosine_similarity(ctx, ground_truth) for ctx in contexts]
    return max(sims)  # best-matching chunk


def custom_eval(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
) -> Dict[str, float]:
    """
    Lightweight evaluation without any LLM.

    Returns:
        dict with keys: answer_similarity, keyword_overlap, context_coverage
    """
    return {
        "answer_similarity": _cosine_similarity(answer, ground_truth),
        "keyword_overlap": _keyword_overlap(answer, ground_truth),
        "context_coverage": _context_coverage(contexts, ground_truth),
    }


# ---------------------------------------------------------------------------
# Context retrieval (for RAGAS — fetches raw chunks from Qdrant)
# ---------------------------------------------------------------------------


def _retrieve_contexts(question: str, top_k: int = 5) -> List[str]:
    """Query Qdrant directly to get the text chunks returned for a question."""
    try:
        from config.settings import get_settings
        from src.ingestion.vectorstore import get_retriever

        settings = get_settings()
        retriever = get_retriever(top_k=top_k)
        docs = retriever.invoke(question)
        return [doc.page_content for doc in docs]
    except Exception as exc:
        logger.warning(
            "Context retrieval failed for question: %s — %s", question[:60], exc
        )
        return []


# ---------------------------------------------------------------------------
# RAGAS setup
# ---------------------------------------------------------------------------


def _try_build_ragas_evaluator() -> Tuple[bool, Any, Any]:
    """
    Attempt to build RAGAS evaluator with local Ollama/qwen2.5:7b (RAGAS 0.4.x).

    Returns:
        (available: bool, llm, embeddings)
        embeddings is a LangchainEmbeddingsWrapper used by ResponseRelevancy only.
    """
    try:
        # RAGAS 0.4.x: LangchainLLMWrapper wraps a LangChain chat model.
        # The 'collections' metrics require OpenAI's structured output and cannot
        # be used with local LLMs. The classic metric classes (_faithfulness, etc.)
        # still work with LangchainLLMWrapper.
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        evaluator_llm = LangchainLLMWrapper(
            ChatOllama(
                model="qwen2.5:7b",
                base_url="http://localhost:11434",
                temperature=0.0,
                timeout=180,
            )
        )

        from config.settings import get_settings

        settings = get_settings()
        models_cache = str(_PROJECT_ROOT / "models_llm")

        # Embeddings used by ResponseRelevancy to measure cosine distance
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                cache_folder=models_cache,
                model_kwargs={"device": settings.embedding_device},
            )
        )

        # Smoke-test: verify the imports that _run_ragas_on_samples needs
        from ragas import EvaluationDataset  # noqa: F401
        from ragas.dataset_schema import SingleTurnSample  # noqa: F401
        from ragas.metrics._answer_relevance import \
            ResponseRelevancy  # noqa: F401
        from ragas.metrics._context_precision import \
            LLMContextPrecisionWithReference  # noqa: F401
        from ragas.metrics._context_recall import \
            LLMContextRecall  # noqa: F401
        from ragas.metrics._faithfulness import Faithfulness  # noqa: F401

        logger.info(
            "RAGAS 0.4.x: configured with Ollama/qwen2.5:7b + %s",
            settings.embedding_model,
        )
        return True, evaluator_llm, evaluator_embeddings

    except ImportError as exc:
        logger.warning(
            "RAGAS or required dependencies not installed (%s). "
            "Run: pip install -r eval/requirements-eval.txt",
            exc,
        )
        return False, None, None
    except Exception as exc:
        logger.warning("RAGAS setup failed: %s — falling back to custom_eval", exc)
        return False, None, None


def _run_ragas_on_samples(
    samples: List[Dict],
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> List[Dict[str, Optional[float]]]:
    """
    Run RAGAS metrics on a list of prepared samples (RAGAS 0.4.x API).

    Each sample dict has keys: question, answer, contexts (list[str]), ground_truth.
    Returns a list of score dicts in the same order as input.

    RAGAS 0.4.x changes from 0.2.x:
    - Use EvaluationDataset + SingleTurnSample (not HuggingFace Dataset)
    - Field names: user_input / response / retrieved_contexts / reference
    - Import metrics from ragas.metrics._* (not module-level singletons)
    - Result DataFrame columns match metric .name attributes
    """
    from ragas import EvaluationDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import \
        LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    # Instantiate metrics with the local LLM (and embeddings for ResponseRelevancy)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    relevancy_metric = ResponseRelevancy(
        llm=evaluator_llm, embeddings=evaluator_embeddings
    )
    precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)
    recall_metric = LLMContextRecall(llm=evaluator_llm)

    metrics = [faithfulness_metric, relevancy_metric, precision_metric, recall_metric]

    # Build EvaluationDataset with RAGAS 0.4.x field names
    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    logger.info(
        "Calling ragas.evaluate() on %d samples with metrics: %s",
        len(samples),
        [m.name for m in metrics],
    )

    # Local Ollama handles one request at a time — set max_workers=1 to avoid
    # concurrent timeout floods, and raise timeout to 300 s per LLM call.
    from ragas.run_config import RunConfig

    run_cfg = RunConfig(timeout=300, max_retries=2, max_workers=1)

    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_cfg,
        raise_exceptions=False,
        show_progress=True,
    )

    # Extract per-row scores; column names match metric .name attributes
    result_df = result.to_pandas()
    logger.debug("RAGAS result columns: %s", list(result_df.columns))

    # Map metric .name → our internal key
    col_map = {
        faithfulness_metric.name: "faithfulness",
        relevancy_metric.name: "answer_relevancy",
        precision_metric.name: "context_precision",
        recall_metric.name: "context_recall",
    }

    scores = []
    for _, row in result_df.iterrows():
        row_scores: Dict[str, Optional[float]] = {}
        for col, internal_key in col_map.items():
            row_scores[internal_key] = _safe_float(row.get(col))
        scores.append(row_scores)

    return scores


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return f if not (f != f) else None  # NaN check
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


def _run_agent(question: str, session_id: str) -> Dict[str, Any]:
    """Call the agent and return its response dict."""
    try:
        from src.agent.run import ask

        return ask(question, session_id=session_id)
    except Exception as exc:
        logger.error("Agent error for question %r: %s", question[:60], exc)
        return {
            "answer": f"[agent_error] {exc}",
            "tools_used": [],
            "sources": [],
            "latency_ms": 0.0,
        }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def run_evaluation(
    test_set_path: str = "eval/test_set.json",
    question_ids: Optional[List[str]] = None,
    use_ragas: bool = True,
    top_k_contexts: int = 5,
) -> EvalResults:
    """
    Run the full evaluation pipeline.

    Args:
        test_set_path:   Path to eval/test_set.json.
        question_ids:    If provided, only evaluate these IDs (for --quick mode).
        use_ragas:       Whether to attempt RAGAS evaluation. If False or if
                         RAGAS is unavailable, uses custom_eval only.
        top_k_contexts:  Number of chunks to retrieve for each question.

    Returns:
        EvalResults dataclass with per-question and aggregate scores.
    """
    t0 = time.perf_counter()
    path = Path(test_set_path)
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    with open(path, encoding="utf-8") as f:
        test_set = json.load(f)

    # Optionally filter to a subset
    if question_ids:
        test_set = [q for q in test_set if q["id"] in question_ids]

    logger.info("Evaluating %d questions from %s", len(test_set), path)

    # Attempt RAGAS setup
    ragas_available = False
    evaluator_llm = None
    evaluator_embeddings = None
    if use_ragas:
        ragas_available, evaluator_llm, evaluator_embeddings = (
            _try_build_ragas_evaluator()
        )

    # ---------- Phase 1: Run agent + retrieve contexts ----------
    single_results: List[SingleResult] = []
    ragas_samples: List[Dict] = []  # populated only for non-negative cases
    ragas_indices: List[int] = []  # index → single_results position
    errors: List[str] = []

    for entry in test_set:
        qid = entry["id"]
        question = entry["question"]
        ground_truth = entry["ground_truth_answer"]
        is_negative = entry.get("negative_case", False)

        logger.info("Running agent: %s", qid)

        # Unique session per question to avoid memory bleed
        session_id = f"eval_{qid}"
        agent_response = _run_agent(question, session_id=session_id)

        answer = agent_response.get("answer", "")
        tools_used = agent_response.get("tools_used", [])
        latency_ms = agent_response.get("latency_ms", 0.0)

        # Retrieve contexts for RAGAS (only for non-negative retrieval questions)
        contexts: List[str] = []
        if not is_negative and entry.get("expected_tool") == "document_search":
            contexts = _retrieve_contexts(question, top_k=top_k_contexts)

        sr = SingleResult(
            question_id=qid,
            question=question,
            domain=entry.get("domain", "unknown"),
            difficulty=entry.get("difficulty", "unknown"),
            negative_case=is_negative,
            expected_tool=entry.get("expected_tool", "unknown"),
            agent_answer=answer,
            tools_used=tools_used,
            latency_ms=latency_ms,
            retrieved_contexts=contexts,
        )

        # Custom eval (always run)
        custom_scores = custom_eval(question, answer, ground_truth, contexts)
        sr.custom_answer_similarity = custom_scores["answer_similarity"]
        sr.custom_keyword_overlap = custom_scores["keyword_overlap"]
        sr.custom_context_coverage = custom_scores["context_coverage"]

        # Queue for RAGAS if applicable
        if ragas_available and not is_negative and contexts:
            ragas_indices.append(len(single_results))
            ragas_samples.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                }
            )

        single_results.append(sr)

    # ---------- Phase 2: RAGAS batch evaluation ----------
    ragas_used = False
    if ragas_available and ragas_samples:
        logger.info(
            "Running RAGAS on %d samples (this may take several minutes)…",
            len(ragas_samples),
        )
        try:
            ragas_scores = _run_ragas_on_samples(
                ragas_samples, evaluator_llm, evaluator_embeddings
            )
            ragas_used = True

            for i, scores in zip(ragas_indices, ragas_scores):
                single_results[i].faithfulness = scores.get("faithfulness")
                single_results[i].answer_relevancy = scores.get("answer_relevancy")
                single_results[i].context_precision = scores.get("context_precision")
                single_results[i].context_recall = scores.get("context_recall")

            logger.info("RAGAS evaluation complete.")
        except Exception as exc:
            msg = f"RAGAS evaluation failed: {exc}"
            logger.error(msg)
            errors.append(msg)

    # ---------- Phase 3: Aggregate ----------
    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else None

    def _domain_breakdown(results: List[SingleResult]) -> Dict[str, Dict[str, float]]:
        from collections import defaultdict

        domains: Dict[str, List[SingleResult]] = defaultdict(list)
        for r in results:
            domains[r.domain].append(r)
        breakdown = {}
        for domain, items in domains.items():
            breakdown[domain] = {
                "count": len(items),
                "mean_faithfulness": _mean([r.faithfulness for r in items]),
                "mean_answer_relevancy": _mean([r.answer_relevancy for r in items]),
                "mean_keyword_overlap": _mean(
                    [r.custom_keyword_overlap for r in items]
                ),
                "mean_answer_similarity": _mean(
                    [r.custom_answer_similarity for r in items]
                ),
            }
        return breakdown

    def _difficulty_breakdown(
        results: List[SingleResult],
    ) -> Dict[str, Dict[str, float]]:
        from collections import defaultdict

        levels: Dict[str, List[SingleResult]] = defaultdict(list)
        for r in results:
            levels[r.difficulty].append(r)
        breakdown = {}
        for level, items in levels.items():
            breakdown[level] = {
                "count": len(items),
                "mean_keyword_overlap": _mean(
                    [r.custom_keyword_overlap for r in items]
                ),
                "mean_answer_similarity": _mean(
                    [r.custom_answer_similarity for r in items]
                ),
            }
        return breakdown

    eval_results = EvalResults(
        ragas_available=ragas_available,
        ragas_used=ragas_used,
        total_questions=len(test_set),
        evaluated=len(single_results),
        skipped=len(test_set) - len(single_results),
        mean_faithfulness=_mean([r.faithfulness for r in single_results]),
        mean_answer_relevancy=_mean([r.answer_relevancy for r in single_results]),
        mean_context_precision=_mean([r.context_precision for r in single_results]),
        mean_context_recall=_mean([r.context_recall for r in single_results]),
        mean_custom_answer_similarity=_mean(
            [r.custom_answer_similarity for r in single_results]
        ),
        mean_custom_keyword_overlap=_mean(
            [r.custom_keyword_overlap for r in single_results]
        ),
        per_domain=_domain_breakdown(single_results),
        per_difficulty=_difficulty_breakdown(single_results),
        results=single_results,
        duration_seconds=time.perf_counter() - t0,
        errors=errors,
    )

    return eval_results
