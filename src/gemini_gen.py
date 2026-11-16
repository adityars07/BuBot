"""Fast Gemini API generator for BuBot RAG pipeline.

Optimized for ultra-fast 3-5 second ChatGPT-style answers using
gemini-3.5-flash-lite and HTTP connection pooling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Persistent session for HTTP keep-alive connection reuse (saves ~1-2s per call)
_SESSION = requests.Session()

SYSTEM_PROMPT = """You are BuBot, a brilliant, fast, and thorough course tutor.
Provide a clear, high-quality, ChatGPT-grade explanation based on the provided slide/note passages.

Guidelines:
1. Direct & Punchy: Explain core concepts clearly and concisely without unnecessary filler.
2. Math & Problem Solving: If slides contain calculations (e.g. forward pass, loss, gradients, weight updates), show the formulas and step-by-step numbers clearly.
3. Quizzes & Questions: If asked about a quiz or problem from slides, state the correct answer and explain why it is correct.
4. Slide Citations: Cite exact slide numbers (e.g., [Slide 3], [Slide 7]) throughout your explanation.
5. Structure: Use bold headers, bullet points, and clean formatting for quick reading.
6. End with a 1-line list of consulted slides."""


def get_api_key() -> str:
    """Retrieve Gemini API key from environment or local .env file."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    env_file = Path(".env")
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip("\"'")
                    if k:
                        os.environ["GEMINI_API_KEY"] = k
                        return k
        except Exception:
            pass
    return ""


def set_api_key(key: str) -> None:
    """Save API key in memory and persist in .env file."""
    clean_key = key.strip()
    if clean_key:
        os.environ["GEMINI_API_KEY"] = clean_key
        try:
            env_file = Path(".env")
            lines = []
            if env_file.exists():
                lines = [l for l in env_file.read_text(encoding="utf-8").splitlines() if not l.startswith("GEMINI_API_KEY=")]
            lines.append(f"GEMINI_API_KEY={clean_key}")
            env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not persist API key to .env: %s", exc)
    else:
        os.environ.pop("GEMINI_API_KEY", None)


def is_available() -> bool:
    """Check if Gemini API key is configured."""
    return bool(get_api_key())


def get_model_name() -> str:
    """Return primary fast model name."""
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def build_gemini_prompt(context_chunks: List[Dict[str, Any]], question: str) -> str:
    """Construct prompt with concise, rich slide context."""
    context_parts = []
    for chunk in context_chunks[:4]:
        fname = chunk.get("file_name", chunk.get("source", "unknown"))
        page = chunk.get("page_number", chunk.get("page", "?"))
        text = chunk.get("text", "").strip()
        context_parts.append(f"[{fname} | Slide {page}]\n{text}\n")

    context_str = "\n".join(context_parts)
    return (
        f"Course Material Excerpts:\n{context_str}\n"
        f"Question: {question.strip()}\n\n"
        f"Answer clearly, solve any math/quizzes mentioned, and cite slide numbers."
    )


def generate_answer(
    context_chunks: List[Dict[str, Any]],
    question: str,
    preferred_model: Optional[str] = None,
) -> Optional[str]:
    """Generate detailed answer in 3-5 seconds using ultra-fast flash-lite models."""
    api_key = get_api_key()
    if not api_key:
        return None

    # Ultra-fast models ordered by speed and availability
    models_to_try = [
        preferred_model or get_model_name(),
        "gemini-3.5-flash-lite",
        "gemma-4-26b-a4b-it",
    ]
    seen = set()
    candidate_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    prompt = build_gemini_prompt(context_chunks, question)
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 500,
        },
    }

    headers = {"Content-Type": "application/json"}

    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            # 8.0s timeout gives ample headroom for cellular/Wi-Fi latency while staying responsive
            resp = _SESSION.post(url, headers=headers, json=payload, timeout=8.0)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
            elif resp.status_code in (404, 503, 429):
                logger.info("Model %s returned %d, trying fallback...", model_name, resp.status_code)
                continue
        except Exception as exc:
            logger.info("Model %s request error: %s", model_name, exc)
            continue

    return None
