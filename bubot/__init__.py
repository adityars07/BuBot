"""BuBot public package interface for Streamlit UI and local RAG workflows.

Exposes:
- Settings.from_env()
- RAGPipeline.from_settings(settings)
- Source dataclass
- Answer dataclass
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config import DEFAULT_CONFIG, Config
from src.index import IndexArtifacts, build_index, get_embedding_model, load_index
from src.ingest import Chunk, PageDoc, chunk_pages, clean_text, detect_repeated_headers_footers, load_pdfs
from src.rag import BuBot

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Document passage retrieved with citation metadata and cosine match score."""

    source: str
    page: int
    text: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert source passage to dictionary."""
        return {
            "source": self.source,
            "page": self.page,
            "text": self.text,
            "score": self.score,
        }


@dataclass
class Answer:
    """Answer object returned by BuBot pipeline."""

    text: str
    sources: List[Source]
    refused: bool
    reason: Optional[str]  # "no_documents" | "below_threshold" | "model_declined" | None
    top_score: float
    latency_s: float


@dataclass
class Settings:
    """Configuration settings for BuBot RAG pipeline."""

    backend: str = "full"
    data_dir: Path = Path("data")
    index_dir: Path = Path("index_store")
    top_k: int = 3
    effective_threshold: float = 0.35

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        backend = os.getenv("BUBOT_BACKEND", "full").strip().lower()
        data_dir_str = os.getenv("BUBOT_DATA_DIR", "data")
        index_dir_str = os.getenv("BUBOT_INDEX_DIR", "index_store")
        top_k = int(os.getenv("BUBOT_TOP_K", "3"))
        threshold = float(os.getenv("BUBOT_THRESHOLD", os.getenv("BUBOT_SIM_THRESHOLD", "0.35")))
        return cls(
            backend=backend,
            data_dir=Path(data_dir_str).resolve(),
            index_dir=Path(index_dir_str).resolve(),
            top_k=top_k,
            effective_threshold=threshold,
        )


class DocumentStore:
    """Tracks indexed document sources and total chunks."""

    def __init__(self, chunks: List[Chunk]):
        self._chunks: List[Chunk] = chunks

    def sources(self) -> Dict[str, int]:
        """Return mapping of file_name -> chunk_count."""
        counts: Dict[str, int] = {}
        for c in self._chunks:
            counts[c.file_name] = counts.get(c.file_name, 0) + 1
        return counts

    def __len__(self) -> int:
        """Return total number of chunks currently indexed."""
        return len(self._chunks)

    def update_chunks(self, chunks: List[Chunk]) -> None:
        """Update internal chunks list."""
        self._chunks = list(chunks)


class EmbedderInfo:
    """Embedder metadata wrapper."""

    def __init__(self, name: str):
        self.name = name


class RAGPipeline:
    """BuBot RAG pipeline conforming to the UI contract."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.top_k = settings.top_k
        self.threshold = settings.effective_threshold
        self.embedder = EmbedderInfo(name=DEFAULT_CONFIG.EMBED_MODEL.split("/")[-1])
        self._generator: Optional[str] = None if settings.backend == "lite" else DEFAULT_CONFIG.GEN_MODEL.split("/")[-1]

        # Internal BuBot bot instance
        self._cfg = Config(
            DATA_DIR=settings.data_dir,
            INDEX_DIR=settings.index_dir,
            TOP_K=self.top_k,
            SIM_THRESHOLD=self.threshold,
        )
        self._bot: Optional[BuBot] = None
        self._chunks: List[Chunk] = []

        # Load existing index if available
        self._load_existing_store()
        self.store = DocumentStore(self._chunks)

    @property
    def generator(self) -> Optional[str]:
        """Dynamically return active generator name (Gemini if available, else local or None)."""
        try:
            from src import gemini_gen
            if gemini_gen.is_available():
                return f"Google Gemini ({gemini_gen.get_model_name()})"
        except Exception:
            pass
        return self._generator

    @generator.setter
    def generator(self, value: Optional[str]) -> None:
        self._generator = value

    def _load_existing_store(self) -> None:
        """Attempt loading existing index store from disk."""
        faiss_path = self.settings.index_dir / "faiss.index"
        chunks_path = self.settings.index_dir / "chunks.jsonl"
        if faiss_path.exists() and chunks_path.exists():
            try:
                artifacts = load_index(
                    index_dir=self.settings.index_dir,
                    data_dir=self.settings.data_dir,
                    config=self._cfg,
                    warn_if_stale=False,
                )
                self._chunks = artifacts.chunks
            except Exception as exc:
                logger.warning("Could not load existing index from %s: %s", self.settings.index_dir, exc)
                self._chunks = []
        else:
            self._chunks = []

    def _get_bot(self) -> BuBot:
        """Instantiate or update underlying BuBot."""
        self._cfg.TOP_K = self.top_k
        self._cfg.SIM_THRESHOLD = self.threshold
        if self._bot is None:
            self._bot = BuBot(config=self._cfg)
        else:
            self._bot.set_top_k(self.top_k)
            self._bot.set_threshold(self.threshold)
        return self._bot

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        """Factory method to construct pipeline from Settings."""
        return cls(settings=settings)

    def add_documents(self, list_of_paths: Sequence[Path | str]) -> Dict[str, int]:
        """Ingest, chunk, and index documents (pdf, txt, md).

        Returns:
            Dict mapping file_name -> chunk_count (0 if no extractable text).
        """
        results: Dict[str, int] = {}
        all_new_pages: List[PageDoc] = []

        for p in list_of_paths:
            path = Path(p).resolve()
            file_name = path.name
            suffix = path.suffix.lower()

            if not path.exists():
                logger.warning("File does not exist: %s", path)
                results[file_name] = 0
                continue

            if suffix == ".pdf":
                # PDF parsing using existing ingest engine
                from pypdf import PdfReader
                from src.ingest import _extract_page_with_fallback

                try:
                    reader = PdfReader(str(path))
                    raw_page_texts: List[str] = []
                    for i, page in enumerate(reader.pages):
                        raw_t = _extract_page_with_fallback(
                            pypdf_page=page,
                            pdf_path=path,
                            page_idx=i,
                            file_name=file_name,
                            min_page_chars=self._cfg.MIN_PAGE_CHARS,
                            use_ocr=self._cfg.USE_OCR,
                        )
                        raw_page_texts.append(raw_t)

                    repeated_lines = detect_repeated_headers_footers(raw_page_texts)
                    for idx, raw_t in enumerate(raw_page_texts):
                        clean_t = clean_text(raw_t, header_footer_lines=repeated_lines)
                        if clean_t.strip():
                            all_new_pages.append(
                                PageDoc(file_name=file_name, page_number=idx + 1, text=clean_t)
                            )
                except Exception as exc:
                    logger.warning("Error reading PDF %s: %s", file_name, exc)
            elif suffix in [".pptx", ".ppt"]:
                from src.ingest import extract_pptx_slides
                try:
                    slide_texts = extract_pptx_slides(path)
                    repeated_lines = detect_repeated_headers_footers(slide_texts)
                    for idx, raw_t in enumerate(slide_texts):
                        clean_t = clean_text(raw_t, header_footer_lines=repeated_lines)
                        if clean_t.strip():
                            all_new_pages.append(
                                PageDoc(file_name=file_name, page_number=idx + 1, text=clean_t)
                            )
                except Exception as exc:
                    logger.warning("Error reading presentation %s: %s", file_name, exc)
            elif suffix in [".txt", ".md"]:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    clean_t = clean_text(content)
                    if clean_t.strip():
                        all_new_pages.append(
                            PageDoc(file_name=file_name, page_number=1, text=clean_t)
                        )
                except Exception as exc:
                    logger.warning("Error reading text file %s: %s", file_name, exc)

            # Check chunks for this file
            file_pages = [p for p in all_new_pages if p.file_name == file_name]
            file_chunks = chunk_pages(pages=file_pages, config=self._cfg)
            results[file_name] = len(file_chunks)

        # Merge new chunks with existing store (replacing any prior chunks of same file)
        new_file_names = set(results.keys())
        remaining_chunks = [c for c in self._chunks if c.file_name not in new_file_names]

        # Generate all chunks for new pages
        new_chunks = chunk_pages(pages=all_new_pages, config=self._cfg)
        self._chunks = remaining_chunks + new_chunks
        self.store.update_chunks(self._chunks)

        # Rebuild index artifacts if we have any chunks
        if self._chunks:
            self.settings.index_dir.mkdir(parents=True, exist_ok=True)
            build_index(
                chunks=self._chunks,
                index_dir=self.settings.index_dir,
                data_dir=self.settings.data_dir,
                config=self._cfg,
            )
            # Invalidate cached bot artifacts so next retrieval reloads
            if self._bot is not None:
                self._bot._artifacts = None

        return results

    def remove_document(self, file_name: str) -> None:
        """Remove a document from the index and data directory."""
        # 1. Remove file from data directory if present
        target_path = self.settings.data_dir / file_name
        if target_path.exists():
            try:
                target_path.unlink()
            except Exception as exc:
                logger.warning("Could not delete file %s: %s", target_path, exc)

        # 2. Filter chunks
        self._chunks = [c for c in self._chunks if c.file_name != file_name]
        self.store.update_chunks(self._chunks)

        # 3. Rebuild or clear index
        if self._chunks:
            build_index(
                chunks=self._chunks,
                index_dir=self.settings.index_dir,
                data_dir=self.settings.data_dir,
                config=self._cfg,
            )
        else:
            # Delete index files
            for fname in ["faiss.index", "chunks.jsonl", "bm25.pkl", "meta.json"]:
                fpath = self.settings.index_dir / fname
                if fpath.exists():
                    fpath.unlink()

        if self._bot is not None:
            self._bot._artifacts = None

    def ask(self, question: str) -> Answer:
        """Answer a student query against indexed documents."""
        start_time = time.perf_counter()

        # Check if store is empty
        if len(self.store) == 0:
            return Answer(
                text="No documents indexed. Please upload your course notes first.",
                sources=[],
                refused=True,
                reason="no_documents",
                top_score=0.0,
                latency_s=0.0,
            )

        bot = self._get_bot()
        # Retrieve extra candidate chunks so reference URLs do not crowd out numerical/content slides
        candidate_k = max(self.top_k, 6)
        raw_candidates = bot.retrieve(question, k=candidate_k)
        
        # UI sources conform to user's top_k
        raw_sources = raw_candidates[:self.top_k]
        sources: List[Source] = [
            Source(source=s.file_name, page=s.page_number, text=s.text, score=float(s.score))
            for s in raw_sources
        ]
        top_score = sources[0].score if sources else 0.0

        # Check strict cosine threshold
        if not sources or top_score < self.threshold:
            latency = time.perf_counter() - start_time
            return Answer(
                text=self._cfg.NOT_FOUND_MESSAGE,
                sources=sources,
                refused=True,
                reason="below_threshold",
                top_score=round(top_score, 4),
                latency_s=round(latency, 4),
            )

        # 1. First attempt generation with Google Gemini (ChatGPT-style tutor response)
        try:
            from src import gemini_gen
            if gemini_gen.is_available():
                candidate_chunks = [
                    {"file_name": s.file_name, "page_number": s.page_number, "text": s.text}
                    for s in raw_candidates
                ]
                gemini_text = gemini_gen.generate_answer(candidate_chunks, question)
                if gemini_text:
                    latency = time.perf_counter() - start_time
                    return Answer(
                        text=gemini_text,
                        sources=sources,
                        refused=False,
                        reason=None,
                        top_score=round(top_score, 4),
                        latency_s=round(latency, 4),
                    )
        except Exception as exc:
            logger.warning("Gemini generation attempt failed: %s", exc)

        # 2. Lite backend or fallback when Gemini is unavailable/busy
        try:
            from src import gemini_gen
            gemini_configured = gemini_gen.is_available()
        except Exception:
            gemini_configured = False

        if gemini_configured or self._generator is None:
            latency = time.perf_counter() - start_time
            synthesized = self._synthesize_passages(sources, question)
            return Answer(
                text=synthesized,
                sources=sources,
                refused=False,
                reason=None,
                top_score=round(top_score, 4),
                latency_s=round(latency, 4),
            )

        # 3. Full backend -> generate with flan-t5
        bot_ans = bot.answer(question)
        final_sources = [
            Source(source=s.file_name, page=s.page_number, text=s.text, score=float(s.score))
            for s in bot_ans.sources
        ]
        reason = "model_declined" if bot_ans.refused else None

        return Answer(
            text=bot_ans.answer_text,
            sources=final_sources,
            refused=bot_ans.refused,
            reason=reason,
            top_score=round(bot_ans.top_score, 4),
            latency_s=round(bot_ans.latency_s, 4),
        )

    def _synthesize_passages(self, sources: List[Source], question: str) -> str:
        """Provide a clean, structured multi-slide summary when LLM is offline or busy."""
        lines: List[str] = []
        content_sources = [s for s in sources if not (s.text.strip().lower().startswith("references http") or s.text.strip().lower().startswith("http"))]
        ordered_sources = content_sources if content_sources else sources

        lines.append(f"Here is what your course notes and slides contain regarding **{question.strip()}**:\n")
        for s in ordered_sources:
            label = f"Slide {s.page}" if s.source.endswith((".pptx", ".ppt")) else f"Page {s.page}"
            lines.append(f"**From {s.source} ({label}) — Match {s.score:.2f}:**")
            for para in s.text.split("\n"):
                if para.strip():
                    lines.append(f"> {para.strip()}")
            lines.append("")

        try:
            from src import gemini_gen
            if gemini_gen.is_available():
                lines.append("---\n*(Note: Live AI tutor generation timed out; direct slide excerpts shown above. Please ask again.)*")
            else:
                lines.append("---\n*💡 For full ChatGPT-style conversational explanations with step-by-step math and problem solving, add your free Gemini API key in the sidebar under Settings.*")
        except Exception:
            pass
        return "\n".join(lines)

