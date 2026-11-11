"""Model pre-download utility for BuBot.

Downloads and caches HuggingFace models locally so the pipeline can operate
100% offline without network access during ingestion, indexing, and generation.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from src.config import DEFAULT_CONFIG, Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_all_models(config: Optional[Config] = None) -> None:
    """Download and cache embedding and generation models to local HF cache.

    Args:
        config: Optional Config instance. Defaults to DEFAULT_CONFIG.
    """
    cfg = config or DEFAULT_CONFIG
    logger.info("Initializing pre-download of HuggingFace models for BuBot...")

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Downloading SentenceTransformer embedding model: %s", cfg.EMBED_MODEL)
        _ = SentenceTransformer(cfg.EMBED_MODEL)
        logger.info("Embedding model cached successfully.")
    except Exception as exc:
        logger.error("Failed downloading embedding model %s: %s", cfg.EMBED_MODEL, exc)
        sys.exit(1)

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        logger.info("Downloading Generator tokenizer: %s", cfg.GEN_MODEL)
        _ = AutoTokenizer.from_pretrained(cfg.GEN_MODEL)
        logger.info("Downloading Generator model: %s", cfg.GEN_MODEL)
        _ = AutoModelForSeq2SeqLM.from_pretrained(cfg.GEN_MODEL)
        logger.info("Generator model and tokenizer cached successfully.")
    except Exception as exc:
        logger.error("Failed downloading generator model %s: %s", cfg.GEN_MODEL, exc)
        sys.exit(1)

    logger.info("All BuBot models downloaded and ready for offline execution.")


if __name__ == "__main__":
    download_all_models()

