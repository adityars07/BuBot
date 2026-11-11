"""Baseline retrieval and generation components for BuBot.

Includes:
1. BM25Retriever: Lexical keyword retrieval baseline using Okapi BM25.
   NOTE: BM25 scores represent term-frequency ranking, NOT cosine similarity in [-1, 1].
   Therefore, similarity threshold rejection is NOT applied to BM25 runs (evaluated
   for hit rate only).
2. NoRetrievalBaseline: Direct model generation without any document context.
   Evaluates base model hallucination and pre-trained knowledge vs grounded RAG.
"""

from __future__ import annotations

import logging
import time
from src.config import DEFAULT_CONFIG, Config
from src.index import IndexArtifacts, load_index, tokenize_for_bm25
from src.rag import RetrievedChunk


class BM25Retriever:
    """Keyword-based BM25Okapi retriever matching the RAG retriever interface."""

    def __init__(
        self,
        index_artifacts: Optional[IndexArtifacts] = None,
        index_dir: Optional[Path | str] = None,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize BM25 retriever with loaded or newly loaded index artifacts."""
        self.config = config or DEFAULT_CONFIG
        self.artifacts = index_artifacts or load_index(index_dir=index_dir, config=self.config)

    def retrieve(self, question: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieve top-k chunks using BM25 Okapi lexical scoring.

        NOTE ON BM25 SCORES:
        BM25 scores are unbounded positive numbers (or zero), reflecting term
        frequency / inverse document frequency. They are NOT cosine similarity
        in [-1, 1]. Threshold-based refusal is intentionally not applied to BM25.

        Args:
            question: Search query string.
            k: Number of chunks to retrieve. Defaults to config.TOP_K.

        Returns:
            List of RetrievedChunk instances sorted by BM25 score descending.
        """
        top_k = k if k is not None else self.config.TOP_K
        if not self.artifacts.chunks or self.artifacts.bm25 is None:
            logger.warning("BM25 index or chunks empty. Returning empty retrieval list.")
            return []

        tokens = tokenize_for_bm25(question)
        if not tokens:
            return []

        scores = self.artifacts.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            chunk = self.artifacts.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    file_name=chunk.file_name,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=float(scores[idx]),
                )
            )
        return results


class NoRetrievalBaseline:
    """No-retrieval generator baseline: passes raw question directly to LLM."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize generator lazily."""
        self.config = config or DEFAULT_CONFIG
        self._tokenizer = None
        self._model = None

    def _ensure_generator_loaded(self) -> None:
        """Load generator model and tokenizer once on CPU."""
        if self._model is None or self._tokenizer is None:
            logger.info("Loading NoRetrieval baseline generator: %s", self.config.GEN_MODEL)
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            torch.set_num_threads(min(4, torch.get_num_threads()))
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.GEN_MODEL)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.GEN_MODEL)
            self._model.eval()

    def answer(self, question: str) -> dict:
        """Generate answer directly from model weights without context.

        Args:
            question: User inquiry.

        Returns:
            Dictionary matching the Answer interface format.
        """
        start_time = time.perf_counter()
        self._ensure_generator_loaded()

        import torch

        prompt = f"Answer the following question:\nQuestion: {question.strip()}\nAnswer:"
        inputs = self._tokenizer(
            prompt,
            max_length=self.config.MAX_INPUT_TOKENS,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                num_beams=self.config.GEN_NUM_BEAMS,
                no_repeat_ngram_size=self.config.GEN_NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
            )

        answer_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        latency = time.perf_counter() - start_time

        return {
            "answer_text": answer_text,
            "grounded": False,
            "refused": False,
            "top_score": 0.0,
            "sources": [],
            "latency_s": round(latency, 4),
            "timings": {"retrieval_s": 0.0, "generation_s": round(latency, 4)},
        }
