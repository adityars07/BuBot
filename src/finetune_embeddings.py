"""Embedding Fine-Tuning Extension for BuBot (Optional Module).

Fine-tunes sentence-transformers (all-MiniLM-L6-v2) on domain course notes using
MultipleNegativesRankingLoss. Supports:
1. Student-supplied training pairs CSV.
2. Synthetic pair generation via flan-t5 (marked as synthetic/lower quality).
Splits pairs 80/20 by chunk (no leakage) and evaluates Hit@k before and after.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import DEFAULT_CONFIG, Config
from src.index import IndexArtifacts, load_index
from src.ingest import Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_question(chunk_text: str, tokenizer, model, max_length: int = 64) -> str:
    """Generate a synthetic question for a chunk using flan-t5."""
    import torch

    prompt = (
        "Generate a concise, specific question whose answer is directly found in the text below.\n\n"
        f"Text:\n{chunk_text[:350]}\n\n"
        "Question:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=2,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def build_synthetic_pairs(
    chunks: List[Chunk],
    config: Config,
    max_pairs: int = 120,
) -> List[Tuple[str, str, str]]:
    """Build synthetic (question, passage, chunk_id) training triplets."""
    logger.info("Generating synthetic training pairs using %s...", config.GEN_MODEL)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.GEN_MODEL)
    model.eval()

    pairs: List[Tuple[str, str, str]] = []
    sampled_chunks = chunks[:max_pairs] if len(chunks) > max_pairs else chunks

    for i, c in enumerate(sampled_chunks, 1):
        try:
            q = generate_synthetic_question(c.text, tokenizer, model)
            if q and len(q) > 10:
                pairs.append((q, c.text, c.chunk_id))
        except Exception as exc:
            logger.warning("Failed generating synthetic question for %s: %s", c.chunk_id, exc)

    logger.info("Generated %d synthetic (question, passage) pairs.", len(pairs))
    return pairs


def load_pairs_from_csv(csv_path: Path | str) -> List[Tuple[str, str, str]]:
    """Load hand-written (question, passage, chunk_id) pairs from CSV."""
    path = Path(csv_path).resolve()
    df = pd.read_csv(path)
    if "question" not in df.columns or "passage" not in df.columns:
        raise ValueError("Pairs CSV must contain 'question' and 'passage' columns.")

    pairs: List[Tuple[str, str, str]] = []
    for idx, row in df.iterrows():
        q = str(row["question"]).strip()
        p = str(row["passage"]).strip()
        c_id = str(row.get("chunk_id", f"pair_{idx}"))
        if q and p:
            pairs.append((q, p, c_id))
    return pairs


def finetune_embeddings(
    pairs: List[Tuple[str, str, str]],
    output_dir: Path | str,
    epochs: int = 2,
    batch_size: int = 8,
    base_model_name: Optional[str] = None,
    config: Optional[Config] = None,
) -> Path:
    """Fine-tune sentence-transformers embedding model with MultipleNegativesRankingLoss."""
    cfg = config or DEFAULT_CONFIG
    model_name = base_model_name or cfg.EMBED_MODEL
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    logger.info("Splitting %d pairs 80/20 by chunk_id to prevent data leakage...", len(pairs))
    # Group pairs by chunk_id
    chunk_groups: Dict[str, List[Tuple[str, str]]] = {}
    for q, p, c_id in pairs:
        chunk_groups.setdefault(c_id, []).append((q, p))

    unique_chunks = list(chunk_groups.keys())
    random.seed(cfg.RANDOM_SEED)
    random.shuffle(unique_chunks)

    split_idx = int(0.8 * len(unique_chunks))
    train_chunks = set(unique_chunks[:split_idx])
    val_chunks = set(unique_chunks[split_idx:])

    train_examples: List[InputExample] = []
    for c_id in train_chunks:
        for q, p in chunk_groups[c_id]:
            train_examples.append(InputExample(texts=[q, p]))

    val_examples: List[InputExample] = []
    for c_id in val_chunks:
        for q, p in chunk_groups[c_id]:
            val_examples.append(InputExample(texts=[q, p]))

    logger.info("Train examples: %d | Validation examples: %d", len(train_examples), len(val_examples))

    logger.info("Loading base embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    logger.info("Starting fine-tuning for %d epochs on CPU (warmup_steps=%d)...", epochs, warmup_steps)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(out_path),
        show_progress_bar=True,
    )

    logger.info("Fine-tuned model successfully saved to: %s", out_path)
    return out_path


def compare_finetuned_model(
    base_model_name: str,
    ft_model_path: Path,
    test_questions_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Evaluate Hit@3 before vs after fine-tuning on the 30 in-scope test questions."""
    from eval.run_eval import compute_hit
    from src.index import build_index
    from src.rag import BuBot
    import tempfile

    in_scope = test_questions_df[test_questions_df["type"] == "in_scope"]
    if len(in_scope) == 0:
        logger.warning("No in-scope test questions to compare fine-tuning.")
        return pd.DataFrame()

    results = []
    models_to_test = [("Baseline (MiniLM)", base_model_name), ("Fine-Tuned (Domain)", str(ft_model_path))]

    for label, m_path in models_to_test:
        with tempfile.TemporaryDirectory() as tmp_idx:
            t_path = Path(tmp_idx)
            t_cfg = Config(EMBED_MODEL=m_path, INDEX_DIR=t_path, DATA_DIR=config.DATA_DIR)
            from src.ingest import chunk_pages, load_pdfs
            pages = load_pdfs(data_dir=config.DATA_DIR, config=t_cfg)
            chunks = chunk_pages(pages, config=t_cfg)
            build_index(chunks=chunks, index_dir=t_path, data_dir=config.DATA_DIR, config=t_cfg)

            bot = BuBot(config=t_cfg)
            hits = 0
            lenient = 0
            for _, row in in_scope.iterrows():
                q = str(row["question"])
                f = str(row["expected_file"])
                p = int(row["expected_page"])
                srcs = bot.retrieve(q, k=3)
                if compute_hit(srcs, f, p, lenient_window=0):
                    hits += 1
                if compute_hit(srcs, f, p, lenient_window=1):
                    lenient += 1

            total = len(in_scope)
            results.append({
                "model": label,
                "model_identifier": m_path,
                "hit@3_strict": round(hits / total, 4),
                "hit@3_lenient": round(lenient / total, 4),
            })

    res_df = pd.DataFrame(results)
    comparison_csv = config.RESULTS_DIR / "finetune_comparison.csv"
    res_df.to_csv(comparison_csv, index=False)
    logger.info("Saved fine-tune comparison table to: %s", comparison_csv)
    return res_df


def main() -> None:
    """CLI entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="BuBot Sentence-Transformers Fine-Tuning CLI")
    parser.add_argument("--mode", choices=["synthetic", "csv"], default="synthetic", help="Pair generation mode")
    parser.add_argument("--pairs-file", type=str, default=None, help="Path to hand-written pairs CSV if mode=csv")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for fine-tuned model")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    out_dir = Path(args.out_dir).resolve() if args.out_dir else cfg.FT_MODEL_OUTPUT_DIR

    artifacts = load_index(config=cfg)
    if not artifacts.chunks:
        logger.error("No chunks available. Run python -m src.index --rebuild first.")
        return

    if args.mode == "csv":
        if not args.pairs_file:
            logger.error("--pairs-file is required when --mode=csv")
            return
        pairs = load_pairs_from_csv(args.pairs_file)
    else:
        pairs = build_synthetic_pairs(artifacts.chunks, config=cfg)

    ft_path = finetune_embeddings(
        pairs=pairs,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        config=cfg,
    )

    if cfg.QUESTIONS_PATH.exists():
        q_df = pd.read_csv(cfg.QUESTIONS_PATH)
        compare_finetuned_model(cfg.EMBED_MODEL, ft_path, q_df, cfg)


if __name__ == "__main__":
    main()

