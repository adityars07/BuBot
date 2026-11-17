"""Core RAG Pipeline and Public Interface for BuBot.

Implements strict threshold-based refusal, prompt construction with context
budgeting (512 tokens), beam-search generation with flan-t5, post-processing,
and the public BuBot class fulfilling the UI contract.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import DEFAULT_CONFIG, Config
from src.index import IndexArtifacts, get_embedding_model, load_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def classify_query_intent(question: str) -> str:
    """Classify user query into academic pedagogical intent categories."""
    q = question.lower().strip()
    if any(w in q for w in ["numerical", "solve", "calculate", "calculation", "compute", "arithmetic", "step-by-step numerical", "step by step numerical"]):
        return "numerical_problem"
    if any(w in q for w in ["with an example", "with example", "give an example", "worked example", "show an example", "illustrate with"]):
        return "explain_with_example"
    if any(w in q for w in ["difference between", "differentiate", "compare", "versus", "vs.", " vs ", "contrast"]):
        return "comparison"
    if any(w in q for w in ["what is", "define", "definition", "explain", "how does", "describe", "overview"]):
        return "conceptual"
    return "factual"


def is_passage_low_quality(text: str, q_lower: str) -> float:
    """Evaluate chunk quality and return a multiplier penalty (0.2 to 1.4)."""
    t = text.strip().lower()
    # 1. Deprioritize bare reference URLs without substantive notes
    if t.startswith(("references http", "http://", "https://")) and len(t.split()) < 10:
        return 0.25
    # 2. Deprioritize bare answer keys unless user specifically asks about quiz/answers
    if "quiz" not in q_lower and "answer key" not in q_lower and "question " not in q_lower:
        if t.startswith("answer key") or re.search(r"\bq[1-5]:\s*[a-d]\b", t):
            return 0.35
    # 3. Boost chunks containing numerical/derivation steps for numerical queries
    if any(term in q_lower for term in ["numerical", "solve", "calculate", "example", "step", "chain rule"]):
        if any(sym in t for sym in ["=", "neth1", "outh1", "e_total", "w5", "chain rule", "eta", "0."]):
            return 1.35
    return 1.0


@dataclass(frozen=True)
class RetrievedChunk:
    """Document chunk retrieved from the index with exact citation and score."""

    chunk_id: str
    file_name: str
    page_number: int  # 1-indexed
    text: str
    score: float


@dataclass
class Answer:
    """Structured response returned by BuBot."""

    answer_text: str
    grounded: bool
    refused: bool
    top_score: float
    sources: List[RetrievedChunk]
    latency_s: float
    timings: Dict[str, float] = field(default_factory=dict)
    evidence_sentence: Optional[str] = None


def find_best_evidence_sentence(chunk_text: str, question: str) -> Optional[str]:
    """Find the sentence in chunk_text most relevant to the question."""
    if not chunk_text:
        return None

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if len(s.strip()) > 10]
    if not sentences:
        return chunk_text.strip()

    q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
    if not q_words:
        return sentences[0]

    best_sent = sentences[0]
    best_overlap = -1
    for s in sentences:
        s_words = set(re.findall(r"\b\w{3,}\b", s.lower()))
        overlap = len(q_words.intersection(s_words))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sent = s

    return best_sent


def build_prompt(
    context_chunks: List[RetrievedChunk],
    question: str,
    tokenizer: Any,
    max_input_tokens: int = 512,
) -> Tuple[str, bool]:
    """Build grounded prompt adhering strictly to max input token limit.

    Drops lowest-ranked chunks or trims the tail of context if limit is exceeded,
    never truncating instructions or the question.

    Args:
        context_chunks: Ranked list of retrieved chunks.
        question: User inquiry.
        tokenizer: HuggingFace tokenizer for token counting.
        max_input_tokens: Maximum allowed prompt token length.

    Returns:
        Tuple of (formatted_prompt_string, was_truncated_boolean).
    """
    instruction = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say 'not found'.\n\n"
        "Context:\n"
    )
    suffix = f"\n\nQuestion: {question.strip()}\n\nAnswer:"

    # Calculate token cost of framing without context
    base_prompt = instruction + suffix
    base_tokens = len(tokenizer.encode(base_prompt, add_special_tokens=False))
    remaining_budget = max_input_tokens - base_tokens - 10  # safety margin

    was_truncated = False
    included_chunks: List[str] = []
    current_tokens = 0

    for chunk in context_chunks:
        chunk_entry = f"[{chunk.file_name} p.{chunk.page_number}] {chunk.text}\n\n"
        entry_tokens = len(tokenizer.encode(chunk_entry, add_special_tokens=False))

        if current_tokens + entry_tokens <= remaining_budget:
            included_chunks.append(chunk_entry)
            current_tokens += entry_tokens
        else:
            # Cannot fit full chunk
            was_truncated = True
            logger.warning("Truncation applied: dropped or trimmed lower-ranked chunk %s", chunk.chunk_id)
            if not included_chunks and remaining_budget > 30:
                # Top chunk itself is larger than budget: trim its tail
                prefix_tag = f"[{chunk.file_name} p.{chunk.page_number}] "
                tag_tokens = len(tokenizer.encode(prefix_tag, add_special_tokens=False))
                allowed_text_tokens = max(10, remaining_budget - tag_tokens - 5)
                token_ids = tokenizer.encode(chunk.text, add_special_tokens=False)[:allowed_text_tokens]
                trimmed_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                included_chunks.append(f"{prefix_tag}{trimmed_text}...\n\n")
            break

    context_str = "".join(included_chunks).strip()
    full_prompt = f"{instruction}{context_str}{suffix}"
    return full_prompt, was_truncated


def postprocess_answer(raw_answer: str, not_found_message: str) -> Tuple[str, bool]:
    """Inspect model generation for 'not found' or empty response.

    Acts as the second guard layer against ungrounded hallucinations.

    Args:
        raw_answer: Decoded string from the generator model.
        not_found_message: Default refusal message.

    Returns:
        Tuple of (final_answer_text, is_refused).
    """
    clean_ans = raw_answer.strip()
    if not clean_ans:
        return not_found_message, True

    # Check for variants of "not found", "not in the context", etc.
    lower_ans = clean_ans.lower()
    refusal_patterns = [
        "not found",
        "not in the context",
        "cannot be found",
        "is not mentioned",
        "not provided in the context",
        "unanswerable",
    ]
    for pattern in refusal_patterns:
        if pattern in lower_ans:
            return not_found_message, True

    return clean_ans, False


class BuBot:
    """BuBot: Strict Grounded Local RAG Course Assistant.

    Fulfills UI contract for Streamlit UI integration without dependencies on Streamlit.
    """

    def __init__(self, config: Optional[Config] = None, **kwargs) -> None:
        """Initialize BuBot with configuration and optional overrides."""
        self.config = config or DEFAULT_CONFIG
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        self.threshold: float = float(kwargs.get("threshold", self.config.SIM_THRESHOLD))
        self.top_k: int = int(kwargs.get("top_k", self.config.TOP_K))

        # Lazy component references
        self._artifacts: Optional[IndexArtifacts] = None
        self._tokenizer = None
        self._generator = None

    def _ensure_index_loaded(self) -> IndexArtifacts:
        """Load index artifacts into memory if not already loaded."""
        if self._artifacts is None:
            self._artifacts = load_index(
                index_dir=self.config.INDEX_DIR,
                data_dir=self.config.DATA_DIR,
                config=self.config,
                warn_if_stale=True,
            )
        return self._artifacts

    def _ensure_generator_loaded(self) -> None:
        """Load tokenizer and generator weights into memory once."""
        if self._generator is None or self._tokenizer is None:
            logger.info("Loading generation model on CPU: %s", self.config.GEN_MODEL)
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            torch.set_num_threads(min(4, torch.get_num_threads()))
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.GEN_MODEL, local_files_only=True)
                self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.config.GEN_MODEL, local_files_only=True)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.GEN_MODEL)
                self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.config.GEN_MODEL)
            self._generator.eval()

    def set_threshold(self, threshold: float) -> None:
        """Set similarity threshold for refusal (UI contract slider)."""
        self.threshold = float(threshold)
        logger.info("BuBot similarity threshold set to: %.4f", self.threshold)

    def set_top_k(self, k: int) -> None:
        """Set top-k retrieval count (UI contract slider)."""
        self.top_k = int(k)
        logger.info("BuBot top_k set to: %d", self.top_k)

    def retrieve(self, question: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """Hybrid RRF retrieval combining dense FAISS search and sparse BM25 search
        with content quality filtering and question intent awareness.

        Args:
            question: User inquiry string.
            k: Retrieval count (defaults to self.top_k).

        Returns:
            List of RetrievedChunk instances sorted descending by hybrid score.
        """
        top_k = k if k is not None else self.top_k
        artifacts = self._ensure_index_loaded()

        if artifacts.is_empty() or artifacts.faiss_index is None:
            logger.warning("Index is empty. No chunks retrieved.")
            return []

        # 1. Dense Semantic Search (all-MiniLM-L6-v2)
        embedder = get_embedding_model(self.config.EMBED_MODEL)
        q_vec = embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)

        total_chunks = len(artifacts.chunks)
        search_k = min(max(top_k * 3, 15), total_chunks)
        dense_scores, dense_indices = artifacts.faiss_index.search(q_vec, search_k)

        # 2. Sparse Lexical Search (BM25Okapi)
        from src.index import tokenize_for_bm25
        tokens = tokenize_for_bm25(question)
        bm25_scores = None
        if artifacts.bm25 is not None and tokens:
            try:
                bm25_scores = artifacts.bm25.get_scores(tokens)
            except Exception as exc:
                logger.warning("BM25 scoring failed: %s", exc)

        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_indices[0])}
        bm25_rank_map = {}
        if bm25_scores is not None:
            bm25_sorted_indices = sorted(range(total_chunks), key=lambda i: bm25_scores[i], reverse=True)
            bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_sorted_indices)}

        # 3. Context Quality Filtering & Reciprocal Rank Fusion (RRF)
        q_lower = question.lower()
        candidate_indices = set(dense_indices[0])
        if bm25_scores is not None:
            candidate_indices.update(sorted(range(total_chunks), key=lambda i: bm25_scores[i], reverse=True)[:search_k])

        scored_candidates = []
        for idx in candidate_indices:
            if idx < 0 or idx >= total_chunks:
                continue
            chunk = artifacts.chunks[idx]
            r_dense = dense_rank_map.get(idx, total_chunks)
            r_bm25 = bm25_rank_map.get(idx, total_chunks)

            # RRF calculation
            rrf_score = (0.65 / (60.0 + r_dense)) + (0.35 / (60.0 + r_bm25))
            quality_factor = is_passage_low_quality(chunk.text, q_lower)
            adjusted_score = rrf_score * quality_factor

            d_cos = float(dense_scores[0][list(dense_indices[0]).index(idx)]) if idx in dense_indices[0] else 0.0
            b_val = float(bm25_scores[idx]) if bm25_scores is not None else 0.0
            effective_score = d_cos if d_cos > 0 else min(0.95, b_val / 10.0)

            scored_candidates.append({
                "chunk": chunk,
                "adjusted_score": adjusted_score,
                "effective_score": effective_score,
                "d_cos": d_cos,
                "b_val": b_val,
            })

        # Sort by adjusted hybrid score descending
        scored_candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)

        results: List[RetrievedChunk] = []
        for item in scored_candidates[:top_k]:
            c = item["chunk"]
            results.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    file_name=c.file_name,
                    page_number=c.page_number,
                    text=c.text,
                    score=round(float(item["effective_score"]), 4),
                )
            )

        return results

    def answer(self, question: str) -> Answer:
        """Answer question with intelligent thresholding, Gemini primary generator,
        and graceful local fallback.
        """
        start_time = time.perf_counter()

        # 1. Retrieval
        retrieval_start = time.perf_counter()
        sources = self.retrieve(question, k=self.top_k)
        retrieval_time = time.perf_counter() - retrieval_start

        top_score = sources[0].score if sources else -1.0

        # 2. Answerability & Grounding Guard
        # Genuine refusal check: out-of-scope queries (e.g. sourdough bread)
        is_answerable = bool(sources and top_score >= self.threshold)
        if not is_answerable:
            total_latency = time.perf_counter() - start_time
            logger.info(
                "Refusing query '%s' (top_score=%.4f < threshold=%.4f).",
                question, top_score, self.threshold
            )
            return Answer(
                answer_text=self.config.NOT_FOUND_MESSAGE,
                grounded=False,
                refused=True,
                top_score=round(top_score, 4),
                sources=sources,
                latency_s=round(total_latency, 4),
                timings={"retrieval_s": round(retrieval_time, 4), "generation_s": 0.0},
                evidence_sentence=None,
            )

        # 3. Primary Generation: Google Gemini (ChatGPT-style tutor response)
        try:
            from src import gemini_gen
            if gemini_gen.is_available():
                gen_start = time.perf_counter()
                candidate_chunks = [
                    {"file_name": s.file_name, "page_number": s.page_number, "text": s.text}
                    for s in sources
                ]
                gemini_text = gemini_gen.generate_answer(candidate_chunks, question)
                gen_time = time.perf_counter() - gen_start
                if gemini_text:
                    total_latency = time.perf_counter() - start_time
                    return Answer(
                        answer_text=gemini_text,
                        grounded=True,
                        refused=False,
                        top_score=round(top_score, 4),
                        sources=sources,
                        latency_s=round(total_latency, 4),
                        timings={"retrieval_s": round(retrieval_time, 4), "generation_s": round(gen_time, 4)},
                        evidence_sentence=None,
                    )
        except Exception as exc:
            logger.warning("Primary Gemini generation failed: %s", exc)

        # 4. Secondary Generation: Local Model (flan-t5 on CPU)
        try:
            self._ensure_generator_loaded()
            prompt, was_truncated = build_prompt(
                context_chunks=sources,
                question=question,
                tokenizer=self._tokenizer,
                max_input_tokens=self.config.MAX_INPUT_TOKENS,
            )
            gen_start = time.perf_counter()
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output_tokens = self._generator.generate(
                    **inputs,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    num_beams=self.config.GEN_NUM_BEAMS,
                    no_repeat_ngram_size=self.config.GEN_NO_REPEAT_NGRAM_SIZE,
                    early_stopping=True,
                )
            raw_output = self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            gen_time = time.perf_counter() - gen_start
            final_text, is_model_refused = postprocess_answer(raw_output, self.config.NOT_FOUND_MESSAGE)

            if not is_model_refused and final_text.strip():
                total_latency = time.perf_counter() - start_time
                return Answer(
                    answer_text=final_text,
                    grounded=True,
                    refused=False,
                    top_score=round(top_score, 4),
                    sources=sources,
                    latency_s=round(total_latency, 4),
                    timings={"retrieval_s": round(retrieval_time, 4), "generation_s": round(gen_time, 4)},
                    evidence_sentence=None,
                )
        except Exception as exc:
            logger.warning("Local model generation failed: %s", exc)

        # 5. Fallback: Synthesize Clean Evidence Passages
        # Relevant passages exist, but LLM generation encountered an error/offline state.
        # DO NOT falsely tell the student "not found in your documents"!
        total_latency = time.perf_counter() - start_time
        lines = [f"Here is what your course notes contain regarding **{question.strip()}**:\n"]
        for s in sources:
            label = f"Slide {s.page_number}" if s.file_name.endswith((".pptx", ".ppt")) else f"Page {s.page_number}"
            lines.append(f"**From {s.file_name} ({label}) — Match {s.score:.2f}:**")
            for para in s.text.split("\n"):
                if para.strip():
                    lines.append(f"> {para.strip()}")
            lines.append("")

        lines.append("---\n*(Note: Live AI generator unavailable/timed out; verified course excerpts provided above.)*")
        fallback_text = "\n".join(lines)

        return Answer(
            answer_text=fallback_text,
            grounded=True,
            refused=False,
            top_score=round(top_score, 4),
            sources=sources,
            latency_s=round(total_latency, 4),
            timings={"retrieval_s": round(retrieval_time, 4), "generation_s": 0.0},
            evidence_sentence=None,
        )

    def index_info(self) -> Dict[str, Any]:
        """Return metadata summary for UI sidebar (UI Contract)."""
        artifacts = self._ensure_index_loaded()
        files = sorted(list({c.file_name for c in artifacts.chunks}))
        return {
            "files": files,
            "num_chunks": len(artifacts.chunks),
            "embed_model": self.config.EMBED_MODEL,
            "gen_model": self.config.GEN_MODEL,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "build_timestamp": artifacts.meta.get("build_timestamp", "N/A"),
        }

    def rebuild_index(self, pdf_paths: Optional[List[str]] = None) -> None:
        """Rebuild index and hot-swap active in-memory artifacts (UI Contract).

        Args:
            pdf_paths: Optional list of PDF file paths. If None, uses config.DATA_DIR.
        """
        from src.index import build_custom_index, build_index
        from src.ingest import chunk_pages, load_pdfs

        if pdf_paths:
            logger.info("Rebuilding index from %d custom PDF files...", len(pdf_paths))
            build_custom_index(pdf_paths=pdf_paths, output_dir=self.config.INDEX_DIR, config=self.config)
        else:
            logger.info("Rebuilding index from %s...", self.config.DATA_DIR)
            pages = load_pdfs(data_dir=self.config.DATA_DIR, config=self.config)
            chunks = chunk_pages(pages=pages, config=self.config)
            build_index(chunks=chunks, index_dir=self.config.INDEX_DIR, data_dir=self.config.DATA_DIR, config=self.config)

        # Hot-swap loaded artifacts
        self._artifacts = load_index(
            index_dir=self.config.INDEX_DIR,
            data_dir=self.config.DATA_DIR,
            config=self.config,
            warn_if_stale=False,
        )
        logger.info("Index hot-swap completed successfully.")


def main() -> None:
    """CLI entry point for querying BuBot."""
    parser = argparse.ArgumentParser(description="BuBot Command-Line Interface")
    parser.add_argument("question", type=str, help="Course question to answer")
    parser.add_argument("--threshold", type=float, default=None, help="Cosine similarity threshold")
    parser.add_argument("--top-k", type=int, default=None, help="Number of retrieved chunks")
    args = parser.parse_args()

    bot = BuBot()
    if args.threshold is not None:
        bot.set_threshold(args.threshold)
    if args.top_k is not None:
        bot.set_top_k(args.top_k)

    ans = bot.answer(args.question)

    print("\n" + "=" * 70)
    print("BUBOT RESPONSE")
    print("=" * 70)
    print(f"Question:    {args.question}")
    print(f"Answer:      {ans.answer_text}")
    print(f"Refused:     {ans.refused} (Grounded: {ans.grounded})")
    print(f"Top Score:   {ans.top_score:.4f} (Threshold: {bot.threshold:.4f})")
    print(f"Latency:     {ans.latency_s:.4f}s [retrieval: {ans.timings.get('retrieval_s', 0):.4f}s, generation: {ans.timings.get('generation_s', 0):.4f}s]")
    if ans.evidence_sentence:
        print(f"Evidence:    {ans.evidence_sentence}")

    print("\nSources Retrieved:")
    if not ans.sources:
        print("  (No chunks retrieved)")
    else:
        for idx, src in enumerate(ans.sources, 1):
            preview = src.text[:200].replace("\n", " ") + ("..." if len(src.text) > 200 else "")
            print(f"  [{idx}] {src.file_name} (Page {src.page_number}) | Score: {src.score:.4f} | ID: {src.chunk_id}")
            print(f"      Text: \"{preview}\"")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

