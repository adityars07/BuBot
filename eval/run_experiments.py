"""Experiment Suite for BuBot RAG Pipeline.

Conducts systematic ablation studies and empirical comparisons:
1. Dense Retrieval (MiniLM + FAISS) vs BM25 Keyword Retrieval (Hit@k)
2. No-Retrieval Baseline vs Grounded RAG (Hallucination / Refusal behavior)
3. Chunk Size Sweep (300, 500, 800 chars) in isolated temporary indices
4. Retrieval k Sweep (k = 2, 3, 5)
5. Refusal Threshold Sweep (0.20 to 0.60 in steps of 0.05) with trade-off curve
   and calculation of the mathematically optimal recommended threshold.
Saves comparison tables to CSV and visual charts to PNG.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.run_eval import compute_hit, compute_token_f1
from eval.make_question_template import validate_questions_csv
from src.baselines import BM25Retriever, NoRetrievalBaseline
from src.config import DEFAULT_CONFIG, Config
from src.index import build_index, load_index
from src.ingest import chunk_pages, load_pdfs
from src.rag import BuBot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_dense_vs_bm25_experiment(
    questions_df: pd.DataFrame,
    bot: BuBot,
    bm25: BM25Retriever,
    k: int = 3,
) -> Dict[str, float]:
    """Compare Hit@k between Dense (FAISS) and Lexical (BM25) on in-scope questions."""
    in_scope = questions_df[questions_df["type"] == "in_scope"]
    if len(in_scope) == 0:
        return {"dense_hit": 0.0, "bm25_hit": 0.0, "dense_lenient": 0.0, "bm25_lenient": 0.0}

    dense_hits = 0
    dense_lenient = 0
    bm25_hits = 0
    bm25_lenient = 0

    for _, row in in_scope.iterrows():
        q = str(row["question"])
        f = str(row["expected_file"])
        p = int(row["expected_page"])

        dense_srcs = bot.retrieve(q, k=k)
        bm25_srcs = bm25.retrieve(q, k=k)

        if compute_hit(dense_srcs, f, p, lenient_window=0):
            dense_hits += 1
        if compute_hit(dense_srcs, f, p, lenient_window=1):
            dense_lenient += 1

        if compute_hit(bm25_srcs, f, p, lenient_window=0):
            bm25_hits += 1
        if compute_hit(bm25_srcs, f, p, lenient_window=1):
            bm25_lenient += 1

    total = len(in_scope)
    return {
        f"dense_hit@{k}": round(dense_hits / total, 4),
        f"dense_lenient_hit@{k}": round(dense_lenient / total, 4),
        f"bm25_hit@{k}": round(bm25_hits / total, 4),
        f"bm25_lenient_hit@{k}": round(bm25_lenient / total, 4),
    }


def run_no_retrieval_comparison(
    questions_df: pd.DataFrame,
    bot: BuBot,
    config: Config,
) -> Dict[str, Any]:
    """Compare No-retrieval generator baseline vs full grounded RAG."""
    no_ret_baseline = NoRetrievalBaseline(config=config)
    oos_df = questions_df[questions_df["type"] == "out_of_scope"]

    oos_rag_refused = 0
    oos_no_ret_refused = 0

    for _, row in oos_df.iterrows():
        q = str(row["question"])
        rag_ans = bot.answer(q)
        no_ret_ans = no_ret_baseline.answer(q)

        if rag_ans.refused:
            oos_rag_refused += 1
        if no_ret_ans.get("refused", False) or "not found" in no_ret_ans.get("answer_text", "").lower():
            oos_no_ret_refused += 1

    total_oos = max(1, len(oos_df))
    return {
        "rag_oos_refusal_rate": round(oos_rag_refused / total_oos, 4),
        "no_retrieval_oos_refusal_rate": round(oos_no_ret_refused / total_oos, 4),
    }


def run_chunk_size_sweep(
    questions_df: pd.DataFrame,
    sizes: List[int],
    config: Config,
) -> Dict[int, float]:
    """Evaluate retrieval Hit@3 across chunk sizes (300, 500, 800) in isolated temp indices."""
    in_scope = questions_df[questions_df["type"] == "in_scope"]
    results: Dict[int, float] = {}

    pages = load_pdfs(data_dir=config.DATA_DIR, config=config)

    for c_size in sizes:
        logger.info("Running chunk size sweep for size=%d...", c_size)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_cfg = Config(
                CHUNK_SIZE=c_size,
                CHUNK_OVERLAP=50,
                DATA_DIR=config.DATA_DIR,
                INDEX_DIR=temp_path,
            )
            chunks = chunk_pages(pages, chunk_size=c_size, overlap=50, config=temp_cfg)
            build_index(chunks=chunks, index_dir=temp_path, data_dir=config.DATA_DIR, config=temp_cfg)

            temp_bot = BuBot(config=temp_cfg)
            hits = 0
            for _, row in in_scope.iterrows():
                q = str(row["question"])
                f = str(row["expected_file"])
                p = int(row["expected_page"])
                srcs = temp_bot.retrieve(q, k=3)
                if compute_hit(srcs, f, p, lenient_window=0):
                    hits += 1

            rate = round(hits / max(1, len(in_scope)), 4)
            results[c_size] = rate
            logger.info("Chunk size %d: Hit@3 = %.4f", c_size, rate)

    return results


def run_k_sweep(
    questions_df: pd.DataFrame,
    k_values: List[int],
    bot: BuBot,
) -> Dict[int, Dict[str, float]]:
    """Evaluate Hit@k, token F1 accuracy proxy, and latency across k values."""
    in_scope = questions_df[questions_df["type"] == "in_scope"]
    results: Dict[int, Dict[str, float]] = {}

    for k in k_values:
        hits = 0
        f1_scores = []
        latencies = []

        for _, row in in_scope.iterrows():
            q = str(row["question"])
            f = str(row["expected_file"])
            p = int(row["expected_page"])
            gold = str(row.get("expected_answer", ""))

            ans = bot.answer(q)
            srcs = bot.retrieve(q, k=k)

            if compute_hit(srcs, f, p, lenient_window=0):
                hits += 1

            if not ans.refused and gold:
                f1_scores.append(compute_token_f1(ans.answer_text, gold))
            latencies.append(ans.latency_s)

        total = max(1, len(in_scope))
        results[k] = {
            "hit_rate": round(hits / total, 4),
            "proxy_f1": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
            "mean_latency_s": round(float(np.mean(latencies)), 4) if latencies else 0.0,
        }
        logger.info("k=%d: Hit@k=%.4f, Proxy F1=%.4f, Latency=%.3fs", k, results[k]["hit_rate"], results[k]["proxy_f1"], results[k]["mean_latency_s"])

    return results


def run_threshold_sweep(
    questions_df: pd.DataFrame,
    bot: BuBot,
    min_thresh: float = 0.20,
    max_thresh: float = 0.60,
    step: float = 0.05,
) -> Tuple[pd.DataFrame, float]:
    """Sweep threshold values to analyze Out-of-Scope Refusal vs In-Scope False Refusal.

    Computes recommended threshold that maximizes (OOS Refusal - In-Scope False Refusal).
    """
    thresholds = [round(t, 2) for t in np.arange(min_thresh, max_thresh + 1e-5, step)]

    # Precompute and cache top dense scores for all questions
    scored_questions: List[Tuple[str, float]] = []
    for _, row in questions_df.iterrows():
        q = str(row["question"])
        q_type = str(row["type"])
        srcs = bot.retrieve(q, k=1)
        score = srcs[0].score if srcs else -1.0
        scored_questions.append((q_type, score))

    rows = []
    in_scope_total = sum(1 for q_type, _ in scored_questions if q_type == "in_scope")
    oos_total = sum(1 for q_type, _ in scored_questions if q_type == "out_of_scope")

    best_utility = -999.0
    recommended_threshold = 0.35

    for t in thresholds:
        oos_refused = sum(1 for q_type, score in scored_questions if q_type == "out_of_scope" and score < t)
        in_false_refused = sum(1 for q_type, score in scored_questions if q_type == "in_scope" and score < t)

        oos_refusal_rate = round(oos_refused / max(1, oos_total), 4)
        in_false_refusal_rate = round(in_false_refused / max(1, in_scope_total), 4)
        net_utility = round(oos_refusal_rate - in_false_refusal_rate, 4)

        if net_utility > best_utility:
            best_utility = net_utility
            recommended_threshold = t

        rows.append(
            {
                "threshold": t,
                "oos_refusal_rate": oos_refusal_rate,
                "in_scope_false_refusal_rate": in_false_refusal_rate,
                "net_utility": net_utility,
            }
        )

    thresh_df = pd.DataFrame(rows)
    return thresh_df, recommended_threshold


def plot_hit_rate_comparisons(
    dense_vs_bm25: Dict[str, float],
    chunk_sweep: Dict[int, float],
    k_sweep: Dict[int, Dict[str, float]],
    output_png: Path,
) -> None:
    """Generate clean comparison bar chart of Hit@k across experiments."""
    labels = ["Dense (MiniLM)", "BM25 (Keyword)"]
    values = [dense_vs_bm25.get("dense_hit@3", 0.0) * 100, dense_vs_bm25.get("bm25_hit@3", 0.0) * 100]

    for size, rate in chunk_sweep.items():
        labels.append(f"Chunk {size}")
        values.append(rate * 100)

    for k, data in k_sweep.items():
        labels.append(f"Top-k={k}")
        values.append(data.get("hit_rate", 0.0) * 100)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#7570b3", "#7570b3", "#1b9e77", "#1b9e77", "#1b9e77"]
    bars = ax.bar(labels, values, color=colors[: len(labels)], edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Hit Rate (%)", fontsize=12)
    ax.set_title("BuBot Retrieval Hit Rate Across Baselines and Sweeps", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    logger.info("Saved hit rate bar chart to: %s", output_png)


def plot_threshold_tradeoff(thresh_df: pd.DataFrame, rec_t: float, output_png: Path) -> None:
    """Plot threshold trade-off curve and mark optimal recommended threshold."""
    fig, ax = plt.subplots(figsize=(8.5, 5))

    ax.plot(
        thresh_df["threshold"],
        thresh_df["oos_refusal_rate"] * 100,
        marker="o",
        linewidth=2,
        color="#2ca02c",
        label="Out-of-Scope Refusal Rate (Desirable)",
    )
    ax.plot(
        thresh_df["threshold"],
        thresh_df["in_scope_false_refusal_rate"] * 100,
        marker="s",
        linewidth=2,
        color="#d62728",
        label="In-Scope False Refusal Rate (Harmful)",
    )

    ax.axvline(x=rec_t, color="#1f77b4", linestyle="--", linewidth=1.5, label=f"Recommended Threshold ({rec_t:.2f})")

    ax.set_xlabel("Cosine Similarity Refusal Threshold", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("BuBot Grounding Trade-Off Curve Across Similarity Thresholds", fontsize=13, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    logger.info("Saved threshold trade-off chart to: %s", output_png)


def run_all_experiments(config: Optional[Config] = None) -> None:
    """Run all specified experiments, write CSVs, and generate plots."""
    cfg = config or DEFAULT_CONFIG
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not cfg.QUESTIONS_PATH.exists():
        logger.error("Questions file not found at: %s. Run python -m eval.make_question_template first.", cfg.QUESTIONS_PATH)
        return

    q_df = pd.read_csv(cfg.QUESTIONS_PATH)
    # Filter valid rows (non-empty questions)
    valid_df = q_df[q_df["question"].str.strip().str.len() > 0].copy()

    logger.info("Initializing BuBot and BM25 baseline for experiments...")
    bot = BuBot(config=cfg)
    bm25 = BM25Retriever(config=cfg)

    # 1. Dense vs BM25
    logger.info("--- Experiment 1: Dense Retrieval vs BM25 Keyword Baseline ---")
    dense_vs_bm25 = run_dense_vs_bm25_experiment(valid_df, bot=bot, bm25=bm25, k=3)

    # 2. No-retrieval baseline vs Grounded RAG
    logger.info("--- Experiment 2: No-Retrieval vs Grounded RAG on Out-of-Scope ---")
    no_ret_comp = run_no_retrieval_comparison(valid_df, bot=bot, config=cfg)

    # 3. Chunk size sweep
    logger.info("--- Experiment 3: Chunk Size Sweep (300, 500, 800) ---")
    chunk_sweep = run_chunk_size_sweep(valid_df, sizes=[300, 500, 800], config=cfg)

    # 4. Top-k sweep
    logger.info("--- Experiment 4: Top-k Sweep (2, 3, 5) ---")
    k_sweep = run_k_sweep(valid_df, k_values=[2, 3, 5], bot=bot)

    # 5. Threshold sweep
    logger.info("--- Experiment 5: Threshold Sweep (0.20 to 0.60) ---")
    thresh_df, rec_threshold = run_threshold_sweep(valid_df, bot=bot, min_thresh=0.20, max_thresh=0.60, step=0.05)

    print("\n" + "=" * 76)
    print("                      BUBOT EXPERIMENT RESULTS")
    print("=" * 76)
    print(f"Dense (MiniLM) Hit@3:         {dense_vs_bm25.get('dense_hit@3', 0.0) * 100:.1f}%")
    print(f"BM25 Keyword Hit@3:           {dense_vs_bm25.get('bm25_hit@3', 0.0) * 100:.1f}%")
    print(f"RAG OOS Refusal Rate:         {no_ret_comp.get('rag_oos_refusal_rate', 0.0) * 100:.1f}%")
    print(f"No-Retrieval OOS Refusal:     {no_ret_comp.get('no_retrieval_oos_refusal_rate', 0.0) * 100:.1f}%")
    print("-" * 76)
    print(f"Recommended Cosine Threshold: {rec_threshold:.2f} (Maximizes: OOS Refusal - In-Scope False Refusal)")
    print("=" * 76 + "\n")

    # Save summary tables
    summary_csv = cfg.RESULTS_DIR / "experiment_summary.csv"
    thresh_csv = cfg.RESULTS_DIR / "threshold_sweep.csv"
    hit_rate_png = cfg.RESULTS_DIR / "experiment_hit_rates.png"
    tradeoff_png = cfg.RESULTS_DIR / "threshold_tradeoff.png"

    thresh_df.to_csv(thresh_csv, index=False)

    summary_records = [
        {"experiment": "Dense vs BM25", "metric": "Dense Hit@3", "value": dense_vs_bm25.get("dense_hit@3")},
        {"experiment": "Dense vs BM25", "metric": "BM25 Hit@3", "value": dense_vs_bm25.get("bm25_hit@3")},
        {"experiment": "Grounded Refusal", "metric": "RAG OOS Refusal", "value": no_ret_comp.get("rag_oos_refusal_rate")},
        {"experiment": "Grounded Refusal", "metric": "No-Ret OOS Refusal", "value": no_ret_comp.get("no_retrieval_oos_refusal_rate")},
        {"experiment": "Chunk Size 300", "metric": "Hit@3", "value": chunk_sweep.get(300)},
        {"experiment": "Chunk Size 500", "metric": "Hit@3", "value": chunk_sweep.get(500)},
        {"experiment": "Chunk Size 800", "metric": "Hit@3", "value": chunk_sweep.get(800)},
        {"experiment": "Top-k 2", "metric": "Hit@2", "value": k_sweep.get(2, {}).get("hit_rate")},
        {"experiment": "Top-k 3", "metric": "Hit@3", "value": k_sweep.get(3, {}).get("hit_rate")},
        {"experiment": "Top-k 5", "metric": "Hit@5", "value": k_sweep.get(5, {}).get("hit_rate")},
        {"experiment": "Optimal Threshold", "metric": "Recommended Threshold", "value": rec_threshold},
    ]
    pd.DataFrame(summary_records).to_csv(summary_csv, index=False)

    # Generate visual charts
    plot_hit_rate_comparisons(dense_vs_bm25, chunk_sweep, k_sweep, hit_rate_png)
    plot_threshold_tradeoff(thresh_df, rec_threshold, tradeoff_png)

    logger.info("Experiment suite finished. Artifacts saved in %s", cfg.RESULTS_DIR)


def main() -> None:
    """CLI entry point for experiments."""
    parser = argparse.ArgumentParser(description="BuBot Experiments Runner")
    parser.parse_args()
    run_all_experiments()


if __name__ == "__main__":
    main()

