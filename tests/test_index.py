"""Unit tests for FAISS and BM25 index creation, search, and stale detection."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.baselines import BM25Retriever
from src.config import Config
from src.index import (
    build_index,
    check_index_stale,
    compute_pdf_fingerprint,
    get_embedding_model,
    load_index,
)
from src.ingest import Chunk


def test_build_save_load_round_trip(tmp_path: Path, mock_chunks: list[Chunk]) -> None:
    """Verify build_index, save, and load_index reconstitute identical artifacts."""
    idx_dir = tmp_path / "index_store"
    cfg = Config(INDEX_DIR=idx_dir, EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2")

    build_index(chunks=mock_chunks, index_dir=idx_dir, data_dir=tmp_path, config=cfg)

    assert (idx_dir / "faiss.index").exists()
    assert (idx_dir / "chunks.jsonl").exists()
    assert (idx_dir / "meta.json").exists()
    assert (idx_dir / "bm25.pkl").exists()

    loaded = load_index(index_dir=idx_dir, data_dir=tmp_path, config=cfg, warn_if_stale=False)
    assert len(loaded.chunks) == len(mock_chunks)
    assert loaded.meta["num_chunks"] == 3
    assert loaded.meta["embed_model"] == cfg.EMBED_MODEL


def test_scores_within_cosine_range(tmp_path: Path, mock_chunks: list[Chunk]) -> None:
    """Verify FAISS IndexFlatIP scores over normalized vectors lie in [-1.0, 1.0]."""
    idx_dir = tmp_path / "index_store"
    cfg = Config(INDEX_DIR=idx_dir)
    build_index(chunks=mock_chunks, index_dir=idx_dir, data_dir=tmp_path, config=cfg)

    loaded = load_index(index_dir=idx_dir, config=cfg, warn_if_stale=False)
    embedder = get_embedding_model(cfg.EMBED_MODEL)

    q_vec = embedder.encode(["What is backpropagation?"], normalize_embeddings=True, convert_to_numpy=True)
    q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)

    scores, indices = loaded.faiss_index.search(q_vec, k=3)
    for score in scores[0]:
        assert -1.0 <= score <= 1.0, f"Score {score} is outside valid cosine range [-1, 1]"

    # Top match should be the backpropagation chunk
    top_chunk = loaded.chunks[indices[0][0]]
    assert "Backpropagation" in top_chunk.text


def test_stale_index_detection(tmp_path: Path, mock_chunks: list[Chunk]) -> None:
    """Verify check_index_stale alerts when files or config change."""
    idx_dir = tmp_path / "index_store"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dummy_pdf = data_dir / "lecture1.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")

    cfg = Config(INDEX_DIR=idx_dir, DATA_DIR=data_dir)
    build_index(chunks=mock_chunks, index_dir=idx_dir, data_dir=data_dir, config=cfg)

    # Initial check: not stale
    is_stale, reason = check_index_stale(idx_dir, data_dir, cfg)
    assert not is_stale

    # Modify PDF file: should become stale
    dummy_pdf.write_bytes(b"%PDF-1.4 modified content with different length")
    is_stale, reason = check_index_stale(idx_dir, data_dir, cfg)
    assert is_stale
    assert "changed" in reason.lower()

    # Change config chunk size: should report stale
    cfg_altered = Config(INDEX_DIR=idx_dir, DATA_DIR=data_dir, CHUNK_SIZE=999)
    is_stale, reason = check_index_stale(idx_dir, data_dir, cfg_altered)
    assert is_stale


def test_bm25_retriever(tmp_path: Path, mock_chunks: list[Chunk]) -> None:
    """Verify BM25 baseline retrieval ranks matching keywords at top."""
    idx_dir = tmp_path / "index_store"
    cfg = Config(INDEX_DIR=idx_dir)
    build_index(chunks=mock_chunks, index_dir=idx_dir, data_dir=tmp_path, config=cfg)

    bm25 = BM25Retriever(index_dir=idx_dir, config=cfg)
    results = bm25.retrieve("learning rate step size", k=1)
    assert len(results) == 1
    assert "Gradient descent" in results[0].text
    assert results[0].page_number == 3

