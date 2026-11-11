"""Unit tests for BuBot RAG pipeline: refusal, strict citations, and prompt truncation."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.config import Config
from src.index import build_index
from src.ingest import Chunk
from src.rag import (
    Answer,
    BuBot,
    RetrievedChunk,
    build_prompt,
    postprocess_answer,
)


def test_rag_threshold_refusal_skips_generator(tmp_path: Path, mock_chunks: list[Chunk]) -> None:
    """Verify out-of-scope query below threshold refuses IMMEDIATELY without calling generator."""
    idx_dir = tmp_path / "index_store"
    cfg = Config(INDEX_DIR=idx_dir, SIM_THRESHOLD=0.50)
    build_index(chunks=mock_chunks, index_dir=idx_dir, data_dir=tmp_path, config=cfg)

    bot = BuBot(config=cfg)
    # Mock generator loading and generation to assert it is never called
    bot._ensure_generator_loaded = MagicMock()

    # Query unrelated to machine learning notes
    answer = bot.answer("How do I bake sourdough bread?")

    assert answer.refused is True
    assert answer.grounded is False
    assert answer.answer_text == cfg.NOT_FOUND_MESSAGE
    # Generator MUST NOT have been called
    bot._ensure_generator_loaded.assert_not_called()
    assert answer.timings["generation_s"] == 0.0


def test_rag_grounded_answer_and_citations(tmp_path: Path, mock_chunks: list[Chunk]) -> None:
    """Verify in-scope query passes threshold and includes exact citations."""
    idx_dir = tmp_path / "index_store"
    cfg = Config(INDEX_DIR=idx_dir, SIM_THRESHOLD=0.20)
    build_index(chunks=mock_chunks, index_dir=idx_dir, data_dir=tmp_path, config=cfg)

    bot = BuBot(config=cfg)
    # Answer in-scope question
    ans = bot.answer("What algorithm calculates gradients of the neural network loss?")

    assert ans.top_score >= 0.20
    assert len(ans.sources) > 0
    # Top source must be backpropagation page 2
    assert ans.sources[0].file_name == "course.pdf"
    assert ans.sources[0].page_number == 2
    assert "course.pdf" in ans.sources[0].file_name
    assert ans.sources[0].chunk_id == "course.pdf::p2::c0"


def test_prompt_truncation_keeps_question_intact() -> None:
    """Verify prompt builder drops lower-ranked chunks when exceeding token limit."""
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            # 1 word ~ 1 token for test simplicity
            return text.split()

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(token_ids)

    tokenizer = MockTokenizer()
    chunks = [
        RetrievedChunk("f::p1::c0", "file.pdf", 1, "word " * 40, 0.9),
        RetrievedChunk("f::p1::c1", "file.pdf", 1, "word " * 40, 0.8),
        RetrievedChunk("f::p1::c2", "file.pdf", 1, "word " * 40, 0.7),
    ]

    # Limit to 80 tokens so not all chunks can fit
    prompt, was_truncated = build_prompt(
        context_chunks=chunks,
        question="What is this question about?",
        tokenizer=tokenizer,
        max_input_tokens=80,
    )

    assert was_truncated is True
    assert "Question: What is this question about?" in prompt
    assert "Answer:" in prompt
    # The prompt still contains the top chunk
    assert "[file.pdf p.1]" in prompt


def test_postprocess_answer_refusal() -> None:
    """Verify postprocessing catches model outputs containing 'not found' or empty."""
    msg = "Not found in your documents."

    text, refused = postprocess_answer("The answer is not found in the context.", msg)
    assert refused is True
    assert text == msg

    text, refused = postprocess_answer("Not Found", msg)
    assert refused is True
    assert text == msg

    text, refused = postprocess_answer("", msg)
    assert refused is True
    assert text == msg

    text, refused = postprocess_answer("Supervised learning uses labeled training data.", msg)
    assert refused is False
    assert text == "Supervised learning uses labeled training data."

