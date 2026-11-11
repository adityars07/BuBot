"""Evaluation Harness for BuBot RAG Pipeline.

Calculates Hit@k (strict & lenient page ±1), Refusal Rate on out-of-scope,
False Refusal Rate on in-scope, Latency percentiles (e2e, retrieval, generation),
and Token F1 / Keyword accuracy proxies. Saves reproducible CSV and JSON summary.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.baselines import BM25Retriever
from src.config import DEFAULT_CONFIG, Config
from src.index import load_index
from src.rag import Answer, BuBot, RetrievedChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_hit(
    sources: List[RetrievedChunk],
    expected_file: str,
    expected_page: int,
    lenient_window: int = 0,
) -> bool:
    """Check if any retrieved source matches the expected file and page.

    Args:
        sources: List of retrieved chunks.
        expected_file: Expected PDF file name.
        expected_page: Expected 1-indexed page number.
        lenient_window: Allowed page margin (e.g. 1 for page ± 1).

    Returns:
        True if hit found, False otherwise.
    """
    if expected_page <= 0 or not expected_file or expected_file.lower() == "n/a":
        return False

    exp_file_clean = expected_file.strip().lower()

    for s in sources:
        if s.file_name.strip().lower() == exp_file_clean:
            if abs(s.page_number - int(expected_page)) <= lenient_window:
                return True
    return False


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and reference answer.

    Automatic proxy metric (NOT official manual metric).

    Args:
        prediction: Generated model answer string.
        ground_truth: Reference expected answer string.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = re.findall(r"\b\w+\b", prediction.lower())
    gold_tokens = re.findall(r"\b\w+\b", ground_truth.lower())

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = collections_overlap(pred_tokens, gold_tokens)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(f1)


def collections_overlap(tokens_a: List[str], tokens_b: List[str]) -> int:
    """Count multi-set intersection between two token sequences."""
    from collections import Counter
    c_a = Counter(tokens_a)
    c_b = Counter(tokens_b)
    overlap = c_a & c_b
    return sum(overlap.values())


def compute_keyword_presence(prediction: str, ground_truth: str) -> bool:
    """Check if all key content words (>= 3 chars) from ground truth are in prediction."""
    gold_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", ground_truth.lower()))
    if not gold_words:
        return False
    pred_lower = prediction.lower()
    return all(w in pred_lower for w in gold_words)


def parse_grade(grade_val: Any) -> Optional[float]:
    """Parse manual grade string ('c', 'p', 'w') or number into float (1.0, 0.5, 0.0)."""
    if pd.isna(grade_val):
        return None
    val = str(grade_val).strip().lower()
    if val in ("c", "correct", "1", "1.0"):
        return 1.0
    if val in ("p", "partial", "0.5"):
        return 0.5
    if val in ("w", "wrong", "0", "0.0"):
        return 0.0
    return None


def run_evaluation(
    questions_df: pd.DataFrame,
    k: int = 3,
    threshold: float = 0.35,
    retriever_type: str = "dense",  # "dense" | "bm25"
    bot_instance: Optional[BuBot] = None,
    config: Optional[Config] = None,
    graded_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run full evaluation suite across in-scope and out-of-scope questions.

    Args:
        questions_df: DataFrame containing evaluation questions.
        k: Top-k chunks to retrieve.
        threshold: Cosine similarity refusal threshold.
        retriever_type: "dense" (FAISS) or "bm25".
        bot_instance: Optional initialized BuBot.
        config: Optional Config instance.
        graded_df: Optional DataFrame with existing manual grades.

    Returns:
        Tuple of (per_question_results_df, summary_metrics_dict).
    """
    cfg = config or DEFAULT_CONFIG
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    bot = bot_instance or BuBot(config=cfg, threshold=threshold, top_k=k)
    bm25_retriever = None
    if retriever_type == "bm25":
        bm25_retriever = BM25Retriever(config=cfg)

    # Map existing grades if provided
    grade_map: Dict[str, Any] = {}
    if graded_df is not None and "grade" in graded_df.columns:
        for _, grow in graded_df.iterrows():
            grade_map[str(grow["id"])] = grow["grade"]

    eval_rows: List[Dict[str, Any]] = []

    for _, row in questions_df.iterrows():
        q_id = str(row["id"])
        q_type = str(row["type"])
        question = str(row["question"]).strip()
        exp_file = str(row["expected_file"]).strip() if pd.notna(row["expected_file"]) else ""
        try:
            exp_page = int(row["expected_page"]) if pd.notna(row["expected_page"]) else -1
        except Exception:
            exp_page = -1
        exp_answer = str(row["expected_answer"]).strip() if pd.notna(row["expected_answer"]) else ""
        difficulty = str(row.get("difficulty", "n/a"))

        # Skip empty unauthored in-scope rows with a warning
        if q_type == "in_scope" and (not question or question.startswith("TODO")):
            logger.warning("Skipping incomplete in-scope question row %s (question is empty)", q_id)
            continue

        start_time = time.perf_counter()

        if retriever_type == "bm25":
            r_start = time.perf_counter()
            sources = bm25_retriever.retrieve(question, k=k)
            retrieval_s = time.perf_counter() - r_start
            top_score = sources[0].score if sources else 0.0

            # BM25 is lexical and unbounded, evaluated for retrieval hit rate only
            model_ans = "BM25 retrieval-only run (generator skipped)"
            refused = False
            grounded = False
            generation_s = 0.0
            e2e_s = time.perf_counter() - start_time
            evidence = None
        else:
            ans: Answer = bot.answer(question)
            sources = ans.sources
            top_score = ans.top_score
            model_ans = ans.answer_text
            refused = ans.refused
            grounded = ans.grounded
            retrieval_s = ans.timings.get("retrieval_s", 0.0)
            generation_s = ans.timings.get("generation_s", 0.0)
            e2e_s = ans.latency_s
            evidence = ans.evidence_sentence

        # Hit rate calculation (in-scope only)
        strict_hit = compute_hit(sources, exp_file, exp_page, lenient_window=0) if q_type == "in_scope" else False
        lenient_hit = compute_hit(sources, exp_file, exp_page, lenient_window=1) if q_type == "in_scope" else False

        # Proxy accuracy calculation (in-scope only, non-refused only)
        token_f1 = 0.0
        kw_match = False
        if q_type == "in_scope" and exp_answer and not refused:
            token_f1 = compute_token_f1(model_ans, exp_answer)
            kw_match = compute_keyword_presence(model_ans, exp_answer)

        # Sources summary string
        sources_summary = "; ".join([f"[{s.file_name} p.{s.page_number} (score={s.score:.3f})]" for s in sources])

        manual_grade = grade_map.get(q_id, "")

        eval_rows.append(
            {
                "id": q_id,
                "type": q_type,
                "question": question,
                "difficulty": difficulty,
                "expected_file": exp_file,
                "expected_page": exp_page,
                "expected_answer": exp_answer,
                "model_answer": model_ans,
                "refused": refused,
                "grounded": grounded,
                "top_score": round(top_score, 4),
                "hit_strict": strict_hit,
                "hit_lenient": lenient_hit,
                "proxy_token_f1": round(token_f1, 4),
                "proxy_kw_match": kw_match,
                "grade": manual_grade,
                "latency_e2e_s": round(e2e_s, 4),
                "latency_retrieval_s": round(retrieval_s, 4),
                "latency_generation_s": round(generation_s, 4),
                "evidence_sentence": evidence or "",
                "sources_retrieved": sources_summary,
            }
        )

    results_df = pd.DataFrame(eval_rows)

    # Compute Aggregate Metrics
    in_scope_df = results_df[results_df["type"] == "in_scope"]
    out_scope_df = results_df[results_df["type"] == "out_of_scope"]

    in_count = len(in_scope_df)
    out_count = len(out_scope_df)

    hit_rate = (in_scope_df["hit_strict"].sum() / in_count) if in_count > 0 else 0.0
    lenient_hit_rate = (in_scope_df["hit_lenient"].sum() / in_count) if in_count > 0 else 0.0
    false_refusal_rate = (in_scope_df["refused"].sum() / in_count) if in_count > 0 else 0.0
    refusal_rate_oos = (out_scope_df["refused"].sum() / out_count) if out_count > 0 else 0.0

    # Latencies
    all_latencies = results_df["latency_e2e_s"].tolist() if len(results_df) > 0 else [0.0]
    r_latencies = results_df["latency_retrieval_s"].tolist() if len(results_df) > 0 else [0.0]
    g_latencies = results_df["latency_generation_s"].tolist() if len(results_df) > 0 else [0.0]

    # Accuracy proxy (Token F1 on in-scope)
    avg_proxy_f1 = in_scope_df["proxy_token_f1"].mean() if in_count > 0 else 0.0

    # Manual grading official accuracy (if graded)
    graded_scores = [parse_grade(g) for g in in_scope_df["grade"] if parse_grade(g) is not None]
    official_accuracy = float(np.mean(graded_scores)) if graded_scores else None

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "retriever": retriever_type,
        "k": k,
        "threshold": threshold,
        "chunk_size": cfg.CHUNK_SIZE,
        "chunk_overlap": cfg.CHUNK_OVERLAP,
        "embed_model": cfg.EMBED_MODEL,
        "gen_model": cfg.GEN_MODEL,
        "total_questions": len(results_df),
        "in_scope_count": in_count,
        "out_of_scope_count": out_count,
        f"hit@{k}_strict": round(float(hit_rate), 4),
        f"hit@{k}_lenient_p1": round(float(lenient_hit_rate), 4),
        "refusal_rate_out_of_scope": round(float(refusal_rate_oos), 4),
        "false_refusal_rate_in_scope": round(float(false_refusal_rate), 4),
        "proxy_accuracy_token_f1": round(float(avg_proxy_f1), 4),
        "official_accuracy_manual": round(official_accuracy, 4) if official_accuracy is not None else "Not graded yet",
        "latency_e2e": {
            "mean_s": round(float(np.mean(all_latencies)), 4),
            "median_s": round(float(np.median(all_latencies)), 4),
            "p95_s": round(float(np.percentile(all_latencies, 95)), 4),
        },
        "latency_retrieval": {
            "mean_s": round(float(np.mean(r_latencies)), 4),
            "median_s": round(float(np.median(r_latencies)), 4),
            "p95_s": round(float(np.percentile(r_latencies, 95)), 4),
        },
        "latency_generation": {
            "mean_s": round(float(np.mean(g_latencies)), 4),
            "median_s": round(float(np.median(g_latencies)), 4),
            "p95_s": round(float(np.percentile(g_latencies, 95)), 4),
        },
    }

    return results_df, summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Print clean formatted terminal summary table of evaluation metrics."""
    k = summary.get("k", 3)
    print("\n" + "=" * 76)
    print("                      BUBOT EVALUATION SUMMARY REPORT")
    print("=" * 76)
    print(f"Retriever:                     {summary.get('retriever')} (k={k}, threshold={summary.get('threshold')})")
    print(f"Total Evaluated Questions:     {summary.get('total_questions')} ({summary.get('in_scope_count')} in-scope, {summary.get('out_of_scope_count')} out-of-scope)")
    print("-" * 76)
    print(f"Top-{k} Hit Rate (Strict file+page):     {summary.get(f'hit@{k}_strict', 0.0) * 100:.1f}%")
    print(f"Top-{k} Hit Rate (Lenient page ± 1):     {summary.get(f'hit@{k}_lenient_p1', 0.0) * 100:.1f}%")
    print(f"Refusal Rate (Out-of-Scope):            {summary.get('refusal_rate_out_of_scope', 0.0) * 100:.1f}% (Correct refusal)")
    print(f"False Refusal Rate (In-Scope):          {summary.get('false_refusal_rate_in_scope', 0.0) * 100:.1f}% (Under-retrieved/rejected)")
    print("-" * 76)
    print(f"Answer Accuracy (Manual Official):      {summary.get('official_accuracy_manual')}")
    print(f"Answer Accuracy (Token F1 Proxy):       {summary.get('proxy_accuracy_token_f1', 0.0):.4f} [PROXY, NOT OFFICIAL]")
    print("-" * 76)
    e2e = summary.get("latency_e2e", {})
    ret = summary.get("latency_retrieval", {})
    gen = summary.get("latency_generation", {})
    print(f"End-to-End Latency:  Mean={e2e.get('mean_s', 0):.3f}s | Median={e2e.get('median_s', 0):.3f}s | p95={e2e.get('p95_s', 0):.3f}s")
    print(f"  - Retrieval:       Mean={ret.get('mean_s', 0):.3f}s | Median={ret.get('median_s', 0):.3f}s | p95={ret.get('p95_s', 0):.3f}s")
    print(f"  - Generation:      Mean={gen.get('mean_s', 0):.3f}s | Median={gen.get('median_s', 0):.3f}s | p95={gen.get('p95_s', 0):.3f}s")
    print("=" * 76 + "\n")


def main() -> None:
    """CLI entry point for run_eval."""
    parser = argparse.ArgumentParser(description="BuBot Evaluation CLI")
    parser.add_argument("--questions", type=str, default=None, help="Path to questions.csv")
    parser.add_argument("--k", type=int, default=3, help="Top-k chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=0.35, help="Cosine similarity refusal threshold")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk character length")
    parser.add_argument("--retriever", type=str, choices=["dense", "bm25"], default="dense", help="Retriever type")
    parser.add_argument("--graded", type=str, default=None, help="Path to previously graded CSV")
    parser.add_argument("--out", type=str, default=None, help="Output results CSV path")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.CHUNK_SIZE = args.chunk_size
    q_path = Path(args.questions).resolve() if args.questions else cfg.QUESTIONS_PATH

    if not q_path.exists():
        logger.error("Questions file not found at: %s. Run python -m eval.make_question_template first.", q_path)
        sys.exit(1)

    questions_df = pd.read_csv(q_path)
    graded_df = pd.read_csv(args.graded) if args.graded else None

    # Prepare output paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out).resolve() if args.out else cfg.RESULTS_DIR / f"run_{timestamp}.csv"
    out_json = out_csv.with_suffix(".json")

    results_df, summary = run_evaluation(
        questions_df=questions_df,
        k=args.k,
        threshold=args.threshold,
        retriever_type=args.retriever,
        config=cfg,
        graded_df=graded_df,
    )

    results_df.to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary_table(summary)
    logger.info("Evaluation results written to: %s", out_csv)
    logger.info("Evaluation JSON summary written to: %s", out_json)


if __name__ == "__main__":
    main()

