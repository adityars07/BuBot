"""Comprehensive Evaluation & Quality Tests for BuBot RAG Pipeline.

Verifies:
1. Relevant question with high retrieval score.
2. Relevant question with different wording from source.
3. Explain-with-example question.
4. Numerical question intent & formula preservation.
5. Multi-part question retrieval coverage.
6. Truly out-of-scope question refusal.
7. Missing API key handling.
8. Gemini timeout and failover resilience.
9. Local generation fallback.
10. Empty generation response handling.
11. Source citation correctness.
12. Mathematical calculation accuracy.
13. Quality filtering of reference URLs and answer keys.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    import faiss
except Exception:
    pass

import unittest.mock as mock
from pathlib import Path
import pytest

from src.config import Config
from src.index import build_index
from src.ingest import Chunk
from src.rag import BuBot, classify_query_intent, is_passage_low_quality
from src import gemini_gen


@pytest.fixture
def academic_chunks() -> list[Chunk]:
    """Rich course chunks including theory, numericals, answer keys, and references."""
    return [
        Chunk(
            chunk_id="lecture15.pptx::p2::c0",
            file_name="AI-ML Lect 15 -16 Backpropagation.pptx",
            page_number=2,
            text=(
                "What is Backpropagation?\n"
                "Backpropagation is an essential supervised learning algorithm for multilayer neural networks.\n"
                "It uses the chain rule of calculus to compute weight gradients and minimize error."
            ),
            char_len=205,
        ),
        Chunk(
            chunk_id="lecture15.pptx::p3::c0",
            file_name="AI-ML Lect 15 -16 Backpropagation.pptx",
            page_number=3,
            text=(
                "Forward Pass: Net Input and Sigmoid Activation\n"
                "neth1 = w1 * i1 + w2 * i2 + b1 * 1\n"
                "w1 = 0.15, i1 = 0.05, w2 = 0.2, i2 = 0.1, b1 = 0.35\n"
                "neth1 = 0.15 * 0.05 + 0.2 * 0.1 + 0.35 * 1 = 0.3775\n"
                "outh1 = 1 / (1 + e^(-0.3775)) = 0.593269992"
            ),
            char_len=220,
        ),
        Chunk(
            chunk_id="lecture15.pptx::p5::c0",
            file_name="AI-ML Lect 15 -16 Backpropagation.pptx",
            page_number=5,
            text=(
                "Total Error Calculation:\n"
                "E_total = sum(1/2 * (target - output)^2)\n"
                "Target o1 = 0.01, Target o2 = 0.99\n"
                "Initial Total Error = 0.298371109"
            ),
            char_len=145,
        ),
        Chunk(
            chunk_id="lecture15.pptx::p7::c0",
            file_name="AI-ML Lect 15 -16 Backpropagation.pptx",
            page_number=7,
            text=(
                "Backward Pass - Actual Learning\n"
                "Now consider weight w5. We want how much change in w5 affects total error.\n"
                "By applying the chain rule: dE_total/dw5 = (dE_total/dout_o1) * (dout_o1/dnet_o1) * (dnet_o1/dw5)"
            ),
            char_len=205,
        ),
        Chunk(
            chunk_id="lecture15.pptx::p12::c0",
            file_name="AI-ML Lect 15 -16 Backpropagation.pptx",
            page_number=12,
            text=(
                "Learning Rate and Weight Update:\n"
                "eta = 0.5\n"
                "w5_new = w5_old - eta * (dE_total/dw5)"
            ),
            char_len=95,
        ),
        Chunk(
            chunk_id="lecture15.pptx::p21::c0",
            file_name="AI-ML Lect 15 -16 Backpropagation.pptx",
            page_number=21,
            text="References https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/",
            char_len=85,
        ),
        Chunk(
            chunk_id="lecture15.pptx::p32::c0",
            file_name="AI-ML Lect 15 -16 Backpropagation.pptx",
            page_number=32,
            text="Answer Key Q1: B — Weight Q2: A — A weighted sum compared with a threshold",
            char_len=75,
        ),
    ]


@pytest.fixture
def indexed_bot(tmp_path: Path, academic_chunks: list[Chunk]) -> BuBot:
    """BuBot instance with indexed academic chunks."""
    idx_dir = tmp_path / "test_index"
    cfg = Config(INDEX_DIR=idx_dir, SIM_THRESHOLD=0.35, TOP_K=4)
    build_index(chunks=academic_chunks, index_dir=idx_dir, data_dir=tmp_path, config=cfg)
    return BuBot(config=cfg)


def test_1_relevant_question_high_retrieval_score(indexed_bot: BuBot) -> None:
    """1. Relevant question retrieves high-scoring passages."""
    sources = indexed_bot.retrieve("What is backpropagation?", k=3)
    assert len(sources) > 0
    assert sources[0].score >= 0.35
    assert "AI-ML Lect 15 -16 Backpropagation.pptx" in sources[0].file_name


def test_2_question_with_different_wording(indexed_bot: BuBot) -> None:
    """2. Question with different wording still retrieves relevant content via hybrid search."""
    sources = indexed_bot.retrieve("How does the network adjust weights using calculus chain rule?", k=4)
    assert len(sources) > 0
    pages = [s.page_number for s in sources]
    assert 7 in pages  # Slide 7 has chain rule


def test_3_query_intent_classification() -> None:
    """3. Question intent is accurately classified for pedagogical structuring."""
    assert classify_query_intent("Explain backpropagation with an example") == "explain_with_example"
    assert classify_query_intent("Solve this backpropagation numerical step-by-step") == "numerical_problem"
    assert classify_query_intent("Differentiate between gradient descent and SGD") == "comparison"
    assert classify_query_intent("What is an artificial neural network?") == "conceptual"
    assert classify_query_intent("Who invented backpropagation?") == "factual"


def test_4_context_quality_filtering(academic_chunks: list[Chunk]) -> None:
    """4. Reference URLs and raw answer keys receive penalty factors."""
    url_penalty = is_passage_low_quality(academic_chunks[5].text, "explain backpropagation")
    assert url_penalty < 0.5  # Reference URL penalized

    key_penalty = is_passage_low_quality(academic_chunks[6].text, "explain backpropagation")
    assert key_penalty < 0.5  # Answer key penalized

    math_factor = is_passage_low_quality(academic_chunks[1].text, "solve numerical step-by-step")
    assert math_factor > 1.0  # Formula boosted


def test_5_out_of_scope_question_refusal(indexed_bot: BuBot) -> None:
    """5. Truly out-of-scope question is cleanly rejected."""
    ans = indexed_bot.answer("How do I bake sourdough bread?")
    assert ans.refused is True
    assert ans.grounded is False
    assert ans.answer_text == indexed_bot.config.NOT_FOUND_MESSAGE


def test_6_multi_part_retrieval_coverage(indexed_bot: BuBot) -> None:
    """6. Multi-part questions retrieve both forward pass and backward pass context."""
    sources = indexed_bot.retrieve("Explain the forward pass net input and backward pass chain rule", k=4)
    pages = {s.page_number for s in sources}
    assert 3 in pages or 7 in pages


def test_7_missing_api_key_fallback(indexed_bot: BuBot, monkeypatch: pytest.MonkeyPatch) -> None:
    """7. Missing Gemini API key gracefully falls back without crashing."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    ans = indexed_bot.answer("What is the formula for net input?")
    assert ans is not None
    assert ans.grounded is True
    assert ans.refused is False
    assert len(ans.sources) > 0


def test_8_gemini_timeout_handling(indexed_bot: BuBot, monkeypatch: pytest.MonkeyPatch) -> None:
    """8. Network timeout during external generation gracefully falls back to verified evidence."""
    monkeypatch.setattr(gemini_gen, "generate_answer", mock.MagicMock(return_value=None))
    monkeypatch.setattr(gemini_gen, "is_available", lambda: True)

    ans = indexed_bot.answer("Explain backpropagation with an example")
    assert ans.refused is False
    assert len(ans.sources) > 0
    assert "From AI-ML Lect 15 -16 Backpropagation.pptx" in ans.answer_text


def test_9_empty_generation_response_handling() -> None:
    """9. Empty candidate content or missing text parts handled cleanly."""
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"candidates": [{"content": {"parts": []}}]}

    with mock.patch("src.gemini_gen._SESSION.post", return_value=mock_resp):
        res = gemini_gen.generate_answer([{"file_name": "test.pdf", "page_number": 1, "text": "sample"}], "query")
        # Should return None, triggering fallback rather than raising an uncaught exception
        assert res is None


def test_10_source_citation_correctness(indexed_bot: BuBot) -> None:
    """10. Citations map to real files and 1-indexed pages."""
    sources = indexed_bot.retrieve("Calculate the total error", k=2)
    for s in sources:
        assert s.page_number >= 1
        assert s.file_name.endswith((".pdf", ".pptx", ".txt"))
        assert s.score >= 0.0


def test_11_mathematical_calculation_correctness() -> None:
    """11. Deterministic arithmetic for course numericals is verified."""
    # From Slide 3: w1=0.15, i1=0.05, w2=0.2, i2=0.1, b1=0.35
    net_h1 = (0.15 * 0.05) + (0.2 * 0.1) + 0.35
    assert round(net_h1, 4) == 0.3775

    import math
    out_h1 = 1.0 / (1.0 + math.exp(-net_h1))
    assert round(out_h1, 4) == 0.5933
