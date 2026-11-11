"""Indexing and Vector Store Management for BuBot.

Embeds document chunks using sentence-transformers (all-MiniLM-L6-v2),
L2-normalizes representations, builds a FAISS IndexFlatIP (cosine similarity)
and a BM25Okapi keyword index, and persists indices with metadata.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import DEFAULT_CONFIG, Config
from src.ingest import Chunk, chunk_pages, load_pdfs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=2)
def get_embedding_model(model_name: str):
    """Load and cache SentenceTransformer model instance (singleton pattern).

    Args:
        model_name: HuggingFace model identifier or local path.

    Returns:
        SentenceTransformer model instance.
    """
    logger.info("Loading embedding model into memory: %s", model_name)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def compute_pdf_fingerprint(data_dir: Path | str) -> str:
    """Compute a deterministic hash of all PDF filenames and sizes in data_dir.

    Args:
        data_dir: Directory containing course PDFs.

    Returns:
        SHA-256 hex digest string.
    """
    dir_path = Path(data_dir).resolve()
    if not dir_path.exists():
        return "empty_dir"

    pdf_files = sorted([p for p in dir_path.glob("*.[pP][dD][fF]") if p.is_file()])
    if not pdf_files:
        return "no_pdfs"

    hasher = hashlib.sha256()
    for f in pdf_files:
        size = f.stat().st_size
        hasher.update(f"{f.name}:{size}\n".encode("utf-8"))
    return hasher.hexdigest()


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words for BM25 retrieval.

    Args:
        text: Input string.

    Returns:
        List of lowercase string tokens.
    """
    import re
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


@dataclass
class IndexArtifacts:
    """Loaded in-memory index components ready for query retrieval."""

    faiss_index: Any
    chunks: List[Chunk]
    bm25: Any
    meta: Dict[str, Any]
    index_dir: Path

    def is_empty(self) -> bool:
        """Check if index contains zero chunks."""
        return len(self.chunks) == 0


def build_index(
    chunks: Sequence[Chunk],
    index_dir: Optional[Path | str] = None,
    data_dir: Optional[Path | str] = None,
    config: Optional[Config] = None,
) -> Path:
    """Embed chunks, build FAISS IndexFlatIP & BM25 index, and save artifacts.

    Args:
        chunks: List of Chunk objects to index.
        index_dir: Target output directory for index artifacts.
        data_dir: Source data directory to calculate PDF fingerprint.
        config: Optional Config instance.

    Returns:
        Path to the saved index directory.
    """
    cfg = config or DEFAULT_CONFIG
    out_dir = Path(index_dir).resolve() if index_dir else cfg.INDEX_DIR
    src_data_dir = Path(data_dir).resolve() if data_dir else cfg.DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks:
        logger.warning("No chunks provided to build_index. Index will be empty.")
        meta = {
            "embed_model": cfg.EMBED_MODEL,
            "chunk_size": cfg.CHUNK_SIZE,
            "chunk_overlap": cfg.CHUNK_OVERLAP,
            "num_chunks": 0,
            "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pdf_fingerprint": compute_pdf_fingerprint(src_data_dir),
            "embedding_dim": 0,
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            pass
        return out_dir

    import faiss
    from rank_bm25 import BM25Okapi

    # 1. Embed chunks in batches with progress bar and L2 normalization
    embedder = get_embedding_model(cfg.EMBED_MODEL)
    texts = [c.text for c in chunks]
    logger.info("Computing dense embeddings for %d chunks (batch_size=%d)...", len(texts), cfg.EMBED_BATCH_SIZE)

    is_interactive = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
    embeddings = embedder.encode(
        texts,
        batch_size=cfg.EMBED_BATCH_SIZE,
        show_progress_bar=is_interactive,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP with dimension %d...", dim)
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    # 2. Build BM25 index
    logger.info("Building BM25 Okapi index over chunk tokens...")
    tokenized_corpus = [tokenize_for_bm25(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. Save artifacts to out_dir
    faiss_path = out_dir / "faiss.index"
    faiss.write_index(faiss_index, str(faiss_path))
    logger.info("Saved FAISS index to %s", faiss_path)

    chunks_jsonl_path = out_dir / "chunks.jsonl"
    with open(chunks_jsonl_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    logger.info("Saved %d chunk metadata entries to %s", len(chunks), chunks_jsonl_path)

    bm25_path = out_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)
    logger.info("Saved BM25 model to %s", bm25_path)

    meta = {
        "embed_model": cfg.EMBED_MODEL,
        "chunk_size": cfg.CHUNK_SIZE,
        "chunk_overlap": cfg.CHUNK_OVERLAP,
        "num_chunks": len(chunks),
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pdf_fingerprint": compute_pdf_fingerprint(src_data_dir),
        "embedding_dim": dim,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved index metadata to %s", meta_path)

    return out_dir


def check_index_stale(index_dir: Path | str, data_dir: Path | str, config: Optional[Config] = None) -> Tuple[bool, str]:
    """Check whether the saved index is stale relative to current PDFs or config.

    Args:
        index_dir: Directory containing index artifacts.
        data_dir: Directory containing source PDFs.
        config: Optional Config instance.

    Returns:
        Tuple of (is_stale, reason_description).
    """
    cfg = config or DEFAULT_CONFIG
    idx_path = Path(index_dir).resolve()
    meta_file = idx_path / "meta.json"
    if not meta_file.exists():
        return True, "meta.json not found in index directory"

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        return True, f"Failed to read meta.json: {exc}"

    current_fingerprint = compute_pdf_fingerprint(data_dir)
    if meta.get("pdf_fingerprint") != current_fingerprint:
        return True, "PDF files in data directory have changed (added, deleted, or modified)"

    if meta.get("embed_model") != cfg.EMBED_MODEL:
        return True, f"Configured embedding model ({cfg.EMBED_MODEL}) differs from index ({meta.get('embed_model')})"

    if meta.get("chunk_size") != cfg.CHUNK_SIZE:
        return True, f"Configured chunk_size ({cfg.CHUNK_SIZE}) differs from index ({meta.get('chunk_size')})"

    if meta.get("chunk_overlap") != cfg.CHUNK_OVERLAP:
        return True, f"Configured chunk_overlap ({cfg.CHUNK_OVERLAP}) differs from index ({meta.get('chunk_overlap')})"

    return False, "Index is up to date"


def load_index(
    index_dir: Optional[Path | str] = None,
    data_dir: Optional[Path | str] = None,
    config: Optional[Config] = None,
    warn_if_stale: bool = True,
) -> IndexArtifacts:
    """Load FAISS index, chunks, BM25, and metadata into an IndexArtifacts container.

    Args:
        index_dir: Path to directory containing index artifacts.
        data_dir: Path to source PDF directory for stale check.
        config: Optional Config instance.
        warn_if_stale: Whether to log a warning if index is detected as stale.

    Returns:
        IndexArtifacts containing all loaded in-memory search components.
    """
    cfg = config or DEFAULT_CONFIG
    idx_path = Path(index_dir).resolve() if index_dir else cfg.INDEX_DIR
    src_data_dir = Path(data_dir).resolve() if data_dir else cfg.DATA_DIR

    if warn_if_stale:
        is_stale, reason = check_index_stale(idx_path, src_data_dir, cfg)
        if is_stale:
            logger.warning("Index at %s may be STALE: %s. Rebuild recommended (--rebuild).", idx_path, reason)

    import faiss

    meta_path = idx_path / "meta.json"
    chunks_path = idx_path / "chunks.jsonl"
    faiss_path = idx_path / "faiss.index"
    bm25_path = idx_path / "bm25.pkl"

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    chunks: List[Chunk] = []
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    chunks.append(Chunk(**data))

    faiss_idx = None
    if faiss_path.exists() and chunks:
        faiss_idx = faiss.read_index(str(faiss_path))

    bm25_obj = None
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            bm25_obj = bm25_data.get("bm25")

    logger.info("Loaded index from %s (%d chunks, FAISS=%s, BM25=%s)", idx_path, len(chunks), faiss_idx is not None, bm25_obj is not None)
    return IndexArtifacts(
        faiss_index=faiss_idx,
        chunks=chunks,
        bm25=bm25_obj,
        meta=meta,
        index_dir=idx_path,
    )


def build_custom_index(
    pdf_paths: Sequence[str | Path],
    output_dir: str | Path,
    config: Optional[Config] = None,
) -> Path:
    """Build an index from an arbitrary list of PDF files into a separate directory.

    Designed for incremental use and user file uploads without mutating the default index.

    Args:
        pdf_paths: Sequence of filesystem paths to PDF files.
        output_dir: Directory where the custom index will be written.
        config: Optional Config instance.

    Returns:
        Path to output directory.
    """
    cfg = config or DEFAULT_CONFIG
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from pypdf import PdfReader
    from src.ingest import clean_text, detect_repeated_headers_footers, _extract_page_with_fallback

    all_pages = []
    for p in pdf_paths:
        path = Path(p).resolve()
        if not path.exists():
            logger.warning("PDF path does not exist: %s", path)
            continue
        try:
            reader = PdfReader(str(path))
            raw_texts = []
            for i, page in enumerate(reader.pages):
                t = _extract_page_with_fallback(
                    pypdf_page=page,
                    pdf_path=path,
                    page_idx=i,
                    file_name=path.name,
                    min_page_chars=cfg.MIN_PAGE_CHARS,
                    use_ocr=cfg.USE_OCR,
                )
                raw_texts.append(t)
            headers = detect_repeated_headers_footers(raw_texts, 0.5)
            for i, raw_t in enumerate(raw_texts):
                cleaned = clean_text(raw_t, headers)
                if cleaned:
                    from src.ingest import PageDoc
                    all_pages.append(PageDoc(file_name=path.name, page_number=i + 1, text=cleaned))
        except Exception as exc:
            logger.error("Failed loading custom PDF %s: %s", path, exc)

    chunks = chunk_pages(pages=all_pages, config=cfg)
    return build_index(chunks=chunks, index_dir=out_dir, data_dir=out_dir, config=cfg)


def main() -> None:
    """CLI entry point for building/rebuilding index."""
    parser = argparse.ArgumentParser(description="BuBot FAISS & BM25 Index Builder")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild index from all PDFs in data directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to PDF documents directory",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Path to output index store directory",
    )
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    data_dir = Path(args.data_dir).resolve() if args.data_dir else cfg.DATA_DIR
    index_dir = Path(args.index_dir).resolve() if args.index_dir else cfg.INDEX_DIR

    logger.info("Starting index build process (rebuild=%s)...", args.rebuild)
    pages = load_pdfs(data_dir=data_dir, config=cfg)
    chunks = chunk_pages(pages=pages, config=cfg)
    build_index(chunks=chunks, index_dir=index_dir, data_dir=data_dir, config=cfg)
    logger.info("Index build completed successfully.")


if __name__ == "__main__":
    main()
