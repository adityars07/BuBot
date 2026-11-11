"""Unit tests for evaluation metrics: Hit@k, refusal rates, and token F1 proxy."""

import pytest
import pandas as pd

from eval.run_eval import (
    compute_hit,
    compute_keyword_presence,
    compute_token_f1,
    parse_grade,
    run_evaluation,
)
from src.rag import RetrievedChunk


def test_compute_hit_strict_and_lenient() -> None:
    """Verify strict hit requires exact file and page, while lenient allows margin."""
    sources = [
        RetrievedChunk("c1", "lecture1.pdf", 4, "text chunk", 0.8),
        RetrievedChunk("c2", "lecture2.pdf", 10, "text chunk", 0.7),
    ]

    # Strict (exact file and page)
    assert compute_hit(sources, "lecture1.pdf", 4, lenient_window=0) is True
    assert compute_hit(sources, "lecture1.pdf", 3, lenient_window=0) is False
    assert compute_hit(sources, "lecture2.pdf", 4, lenient_window=0) is False

    # Lenient (page ± 1)
    assert compute_hit(sources, "lecture1.pdf", 3, lenient_window=1) is True
    assert compute_hit(sources, "lecture1.pdf", 5, lenient_window=1) is True
    assert compute_hit(sources, "lecture1.pdf", 6, lenient_window=1) is False

    # Out of scope / invalid
    assert compute_hit(sources, "n/a", -1, lenient_window=1) is False


def test_compute_token_f1() -> None:
    """Verify SQuAD-style token F1 precision/recall harmonic mean calculation."""
    # Exact match -> 1.0
    assert compute_token_f1("deep neural network", "deep neural network") == 1.0

    # Completely disjoint -> 0.0
    assert compute_token_f1("cat dog fish", "apple orange banana") == 0.0

    # Partial match: pred="deep neural net" (3 tokens), gold="deep neural network" (3 tokens)
    # common = 2 ("deep", "neural") -> precision=2/3, recall=2/3 -> F1=2/3 ~ 0.6667
    f1 = compute_token_f1("deep neural net", "deep neural network")
    assert pytest.approx(f1, 0.01) == 0.6667


def test_compute_keyword_presence() -> None:
    """Verify keyword presence requires all content words (>= 3 chars) to be present."""
    ground_truth = "gradient descent algorithm"
    assert compute_keyword_presence("The gradient descent algorithm optimizes loss", ground_truth) is True
    assert compute_keyword_presence("Only gradient descent is here", ground_truth) is False


def test_parse_grade() -> None:
    """Verify manual grade parsing for c, p, w."""
    assert parse_grade("c") == 1.0
    assert parse_grade("C") == 1.0
    assert parse_grade("p") == 0.5
    assert parse_grade("w") == 0.0
    assert parse_grade(None) is None
    assert parse_grade("") is None


def test_run_evaluation_metrics_calculation() -> None:
    """Verify evaluation metric aggregation on known deterministic mock results."""
    mock_df = pd.DataFrame([
        {
            "id": "in_01",
            "type": "in_scope",
            "question": "What is backprop?",
            "expected_file": "notes.pdf",
            "expected_page": 2,
            "expected_answer": "gradient calculation",
            "difficulty": "easy",
        },
        {
            "id": "out_01",
            "type": "out_of_scope",
            "question": "Who won world cup?",
            "expected_file": "n/a",
            "expected_page": -1,
            "expected_answer": "France",
            "difficulty": "n/a",
        },
    ])

    class MockBot:
        def answer(self, q):
            from src.rag import Answer
            if "backprop" in q.lower():
                return Answer(
                    answer_text="Calculates gradient of loss",
                    grounded=True,
                    refused=False,
                    top_score=0.85,
                    sources=[RetrievedChunk("c1", "notes.pdf", 2, "Backprop text", 0.85)],
                    latency_s=0.1,
                    timings={"retrieval_s": 0.02, "generation_s": 0.08},
                )
            else:
                return Answer(
                    answer_text="Not found in your documents.",
                    grounded=False,
                    refused=True,
                    top_score=0.15,
                    sources=[],
                    latency_s=0.02,
                    timings={"retrieval_s": 0.02, "generation_s": 0.0},
                )

    results_df, summary = run_evaluation(
        questions_df=mock_df,
        k=1,
        threshold=0.35,
        bot_instance=MockBot(),
    )

    assert summary["hit@1_strict"] == 1.0
    assert summary["refusal_rate_out_of_scope"] == 1.0
    assert summary["false_refusal_rate_in_scope"] == 0.0
    assert summary["proxy_accuracy_token_f1"] > 0.0

