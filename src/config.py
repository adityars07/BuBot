"""Configuration management for BuBot.

All tunable hyperparameters, model checkpoints, file paths, and runtime
thresholds are defined here. Supports overriding via environment variables
(prefixed with BUBOT_).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Central configuration for BuBot RAG pipeline."""

    # Project directories
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(os.getenv("BUBOT_DATA_DIR", "data")).resolve())
    INDEX_DIR: Path = field(default_factory=lambda: Path(os.getenv("BUBOT_INDEX_DIR", "index_store")).resolve())
    RESULTS_DIR: Path = field(default_factory=lambda: Path(os.getenv("BUBOT_RESULTS_DIR", "eval/results")).resolve())
    QUESTIONS_PATH: Path = field(default_factory=lambda: Path(os.getenv("BUBOT_QUESTIONS_PATH", "eval/questions.csv")).resolve())

    # Models (100% local, offline on CPU)
    EMBED_MODEL: str = os.getenv("BUBOT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    FALLBACK_EMBED_MODEL: str = os.getenv("BUBOT_FALLBACK_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    GEN_MODEL: str = os.getenv("BUBOT_GEN_MODEL", "google/flan-t5-base")
    REMOTE_GEN_URL: str = os.getenv("BUBOT_REMOTE_GEN_URL", "")  # Optional hook for Colab remote inference

    # Chunking hyperparameters
    CHUNK_SIZE: int = int(os.getenv("BUBOT_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("BUBOT_CHUNK_OVERLAP", "50"))
    MIN_CHUNK_CHARS: int = int(os.getenv("BUBOT_MIN_CHUNK_CHARS", "30"))
    MIN_PAGE_CHARS: int = int(os.getenv("BUBOT_MIN_PAGE_CHARS", "20"))

    # Retrieval and thresholding
    TOP_K: int = int(os.getenv("BUBOT_TOP_K", "3"))
    SIM_THRESHOLD: float = float(os.getenv("BUBOT_SIM_THRESHOLD", "0.35"))
    EMBED_BATCH_SIZE: int = int(os.getenv("BUBOT_EMBED_BATCH_SIZE", "32"))

    # Generation hyperparameters
    MAX_INPUT_TOKENS: int = int(os.getenv("BUBOT_MAX_INPUT_TOKENS", "512"))
    MAX_NEW_TOKENS: int = int(os.getenv("BUBOT_MAX_NEW_TOKENS", "128"))
    GEN_NUM_BEAMS: int = int(os.getenv("BUBOT_GEN_NUM_BEAMS", "4"))
    GEN_NO_REPEAT_NGRAM_SIZE: int = int(os.getenv("BUBOT_GEN_NO_REPEAT_NGRAM_SIZE", "3"))
    ANSWER_STYLE: str = os.getenv("BUBOT_ANSWER_STYLE", "standard")  # "standard" | "extractive_fallback"
    EXTRACTIVE_MIN_WORDS: int = int(os.getenv("BUBOT_EXTRACTIVE_MIN_WORDS", "3"))

    # Grounding & safety
    NOT_FOUND_MESSAGE: str = os.getenv("BUBOT_NOT_FOUND_MESSAGE", "Not found in your documents.")
    USE_OCR: bool = os.getenv("BUBOT_USE_OCR", "false").lower() in ("true", "1", "yes")
    RANDOM_SEED: int = int(os.getenv("BUBOT_RANDOM_SEED", "42"))

    # Fine-tuning extension hyperparameters
    FT_MODEL_OUTPUT_DIR: Path = field(
        default_factory=lambda: Path(os.getenv("BUBOT_FT_MODEL_DIR", "models/bubot-minilm-ft")).resolve()
    )
    FT_EPOCHS: int = int(os.getenv("BUBOT_FT_EPOCHS", "2"))
    FT_BATCH_SIZE: int = int(os.getenv("BUBOT_FT_BATCH_SIZE", "8"))
    FT_WARMUP_STEPS: int = int(os.getenv("BUBOT_FT_WARMUP_STEPS", "10"))

    def __post_init__(self) -> None:
        """Ensure necessary parent directories exist or can be resolved."""
        # Convert string paths to Path objects if set through constructor
        if not isinstance(self.DATA_DIR, Path):
            self.DATA_DIR = Path(self.DATA_DIR).resolve()
        if not isinstance(self.INDEX_DIR, Path):
            self.INDEX_DIR = Path(self.INDEX_DIR).resolve()
        if not isinstance(self.RESULTS_DIR, Path):
            self.RESULTS_DIR = Path(self.RESULTS_DIR).resolve()
        if not isinstance(self.QUESTIONS_PATH, Path):
            self.QUESTIONS_PATH = Path(self.QUESTIONS_PATH).resolve()
        if not isinstance(self.FT_MODEL_OUTPUT_DIR, Path):
            self.FT_MODEL_OUTPUT_DIR = Path(self.FT_MODEL_OUTPUT_DIR).resolve()


# Global default instance
DEFAULT_CONFIG = Config()

