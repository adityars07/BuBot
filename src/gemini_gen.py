"""Fast Gemini API generator for BuBot RAG pipeline.

Optimized for ultra-fast ChatGPT/Claude-grade academic tutoring with complete
step-by-step numerical problem solving and intuitive explanations using gemini-3.5-flash-lite.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Persistent session for HTTP keep-alive connection reuse (saves ~1-2s per call)
_SESSION = requests.Session()

SYSTEM_PROMPT = """You are BuBot, an intelligent academic tutor.

Your responsibility is to explain academic concepts clearly, accurately, and thoroughly using the user's uploaded course materials.

You must use the retrieved course material as your primary source of truth.

Do not simply copy retrieved passages. Understand the relevant information and synthesize it into a coherent educational explanation.

When the user requests an example, provide a worked example.

When the user requests a numerical solution, show the complete calculation process, including formulas, substitutions, intermediate values, and the final answer.

When the user requests a step-by-step explanation, organize the response into clearly numbered steps.

When the question involves mathematical concepts, use correctly formatted mathematical notation in LaTeX ($...$ inline, $$...$$ block).

Explain technical terminology before using it extensively.

If the uploaded material contains sufficient information, answer the question completely rather than refusing simply because the question uses different wording from the source.

You may use basic mathematical reasoning to explain or derive results from formulas and facts present in the uploaded material. Clearly distinguish derived calculations from directly stated material.

Never invent facts, formulas, numerical values, or course-specific claims that are unsupported by the retrieved context.

If essential information is genuinely missing, explain exactly what is missing and provide the parts that can be answered from the available material.

Always include relevant source citations using the original file name and page or slide number at the end of the explanation.

Your objective is to help students understand concepts, not merely retrieve text."""


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


def build_gemini_prompt(context_chunks: List[Dict[str, Any]], question: str) -> str:
    """Construct prompt with substantive, rich slide context, filtering pure link slides."""
    substantive_chunks = []
    link_chunks = []
    for chunk in context_chunks:
        text = chunk.get("text", "").strip()
        if text.lower().startswith(("references http", "http://", "https://")) and len(text.split()) < 8:
            link_chunks.append(chunk)
        else:
            substantive_chunks.append(chunk)

    # Prioritize content-rich slides, passing up to 6 chunks for complete context
    selected_chunks = (substantive_chunks + link_chunks)[:6]

    context_parts = []
    for chunk in selected_chunks:
        fname = chunk.get("file_name", chunk.get("source", "unknown"))
        page = chunk.get("page_number", chunk.get("page", "?"))
        text = chunk.get("text", "").strip()
        context_parts.append(f"[{fname} | Slide {page}]\n{text}\n")

    context_str = "\n".join(context_parts)
    intent = classify_query_intent(question)

    if intent == "numerical_problem":
        format_guide = (
            "Required Structure for Numerical Solution:\n"
            "1. Given Parameters: State all inputs, weights, biases, targets, and learning rates directly from the excerpts.\n"
            "2. Required Result: Clearly define what value/weights need to be calculated.\n"
            "3. Governing Formulas: Write each formula in clean LaTeX blocks ($$...$$).\n"
            "4. Step-by-Step Derivation & Calculations: Show full numerical substitution and compute all intermediate values step-by-step (e.g. Forward Pass net inputs and activations, Error calculation, Backward Pass chain rule gradients, and Weight updates).\n"
            "5. Final Result & Verification: Conclude with the updated values and verify error reduction.\n"
            "6. Sources: State the consulted slides at the end."
        )
    elif intent == "explain_with_example":
        format_guide = (
            "Required Structure for Explain with Example:\n"
            "1. Concept Introduction: Explain the concept with a clear, intuitive real-world analogy.\n"
            "2. Working Mechanism: Explain how the process works conceptually.\n"
            "3. Relevant Equations: State the key governing mathematical equations.\n"
            "4. Concrete Worked Example: Provide a clear worked example using the values/figures from the course excerpts.\n"
            "5. Step-by-Step Explanation: Break down each phase clearly.\n"
            "6. Final Conclusion & Key Takeaways.\n"
            "7. Sources: State the consulted slides at the end."
        )
    elif intent == "comparison":
        format_guide = (
            "Required Structure for Comparison:\n"
            "1. Brief Introduction of both concepts.\n"
            "2. Markdown Comparison Table across key criteria (purpose, mechanism, convergence, speed, practical trade-offs).\n"
            "3. Detailed Comparative Analysis.\n"
            "4. Practical Example or context from the notes.\n"
            "5. Key Differences Summary.\n"
            "6. Sources: State the consulted slides at the end."
        )
    elif intent == "conceptual":
        format_guide = (
            "Required Structure for Conceptual Question:\n"
            "1. Simple Definition.\n"
            "2. Why it is Needed (problem it addresses).\n"
            "3. How it Works (mechanics and stages).\n"
            "4. Important Terminology explained simply.\n"
            "5. Small Illustrative Example.\n"
            "6. Summary.\n"
            "7. Sources: State the consulted slides at the end."
        )
    else:
        format_guide = "Provide a direct, accurate, and concise answer grounded in the course excerpts, citing sources at the end."

    return (
        f"Course Context from Slides/Notes:\n{context_str}\n\n"
        f"Student Question: {question.strip()}\n\n"
        f"{format_guide}\n\n"
        f"Ensure all calculations are completely carried out, numbers substituted, and explanations fully self-contained."
    )


def generate_answer(
    context_chunks: List[Dict[str, Any]],
    question: str,
    preferred_model: Optional[str] = None,
) -> Optional[str]:
    """Generate detailed, complete academic answer using fast flash-lite models."""
    api_key = get_api_key()
    if not api_key:
        return None

    # Fast models ordered by capability and availability
    models_to_try = [
        preferred_model or get_model_name(),
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
    ]
    seen = set()
    candidate_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    prompt = build_gemini_prompt(context_chunks, question)
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4000,
        },
    }

    headers = {"Content-Type": "application/json"}

    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            # 25.0s timeout ensures full mathematical derivations finish safely even on high network load
            resp = _SESSION.post(url, headers=headers, json=payload, timeout=25.0)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text_parts = [p.get("text", "") for p in parts if "text" in p]
                    full_text = "".join(text_parts).strip()
                    if full_text:
                        return full_text
            elif resp.status_code in (404, 503, 429):
                logger.info("Model %s returned %d, trying fallback...", model_name, resp.status_code)
                continue
        except Exception as exc:
            logger.info("Model %s request error: %s", model_name, exc)
            continue

    return None
