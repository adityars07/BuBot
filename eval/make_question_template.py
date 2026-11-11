"""Generator and Validator for BuBot Evaluation Questions Template.

Creates eval/questions.csv with:
- 10 pre-filled real out-of-scope trivia questions spanning sports, cooking,
  history, geography, and entertainment (notes will not cover them).
- 30 diverse chunk samples from indexed course PDFs with source_chunk helper text
  for the student to author in-scope questions and expected answers by hand.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import DEFAULT_CONFIG, Config
from src.index import load_index
from src.ingest import Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "id",
    "type",
    "question",
    "expected_file",
    "expected_page",
    "expected_answer",
    "difficulty",
    "source_chunk",
]

PREFILLED_OUT_OF_SCOPE = [
    {
        "type": "out_of_scope",
        "question": "Who won the 2018 FIFA World Cup in Russia?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "France",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: sports",
    },
    {
        "type": "out_of_scope",
        "question": "What is the capital city of Australia?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "Canberra",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: geography",
    },
    {
        "type": "out_of_scope",
        "question": "How do I bake traditional sourdough bread at home?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "Use sourdough starter, flour, water, salt, and ferment before baking.",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: cooking",
    },
    {
        "type": "out_of_scope",
        "question": "In what year did the Apollo 11 mission land on the Moon?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "1969",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: space exploration",
    },
    {
        "type": "out_of_scope",
        "question": "Which artist painted the Mona Lisa?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "Leonardo da Vinci",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: art",
    },
    {
        "type": "out_of_scope",
        "question": "What are the rules and objective of curling in the Winter Olympics?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "Players slide stones on a sheet of ice towards a target area.",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: sports",
    },
    {
        "type": "out_of_scope",
        "question": "What is the chemical formula and preparation method for carbonara sauce?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "Eggs, hard cheese, cured pork (guanciale), and black pepper.",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: culinary",
    },
    {
        "type": "out_of_scope",
        "question": "Who directed the 1994 film The Shawshank Redemption?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "Frank Darabont",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: entertainment",
    },
    {
        "type": "out_of_scope",
        "question": "What is the highest mountain peak in North America?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "Denali (Mount McKinley)",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: geography",
    },
    {
        "type": "out_of_scope",
        "question": "Which empire built the Colosseum in Rome?",
        "expected_file": "n/a",
        "expected_page": -1,
        "expected_answer": "The Roman Empire",
        "difficulty": "n/a",
        "source_chunk": "General knowledge: ancient history",
    },
]


def validate_questions_csv(
    csv_path: Path | str,
    data_dir: Path | str,
    strict_count: bool = True,
) -> pd.DataFrame:
    """Validate questions CSV format, page integers, and PDF file existence.

    Args:
        csv_path: Path to questions.csv file.
        data_dir: Path to directory containing course PDFs.
        strict_count: Require exactly 30 in-scope and 10 out-of-scope rows.

    Returns:
        Validated pandas DataFrame.

    Raises:
        ValueError: If schema, file references, or row counts are invalid.
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Questions CSV file not found: {path}")

    df = pd.read_csv(path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column in {path}: '{col}'. Expected: {REQUIRED_COLUMNS}")

    # Check row counts
    in_scope = df[df["type"] == "in_scope"]
    out_of_scope = df[df["type"] == "out_of_scope"]

    if strict_count:
        if len(in_scope) != 30:
            raise ValueError(f"Expected exactly 30 in_scope rows, but found {len(in_scope)} in {path}")
        if len(out_of_scope) != 10:
            raise ValueError(f"Expected exactly 10 out_of_scope rows, but found {len(out_of_scope)} in {path}")

    # Check page numbers are valid integers
    for idx, row in df.iterrows():
        try:
            int(row["expected_page"])
        except (ValueError, TypeError):
            raise ValueError(
                f"Row {row['id']} has invalid non-integer expected_page: {row['expected_page']}"
            )

    # Check expected_file exists for in_scope rows
    data_path = Path(data_dir).resolve()
    for idx, row in in_scope.iterrows():
        exp_file = str(row["expected_file"]).strip()
        if not exp_file or exp_file == "n/a":
            continue
        expected_pdf = data_path / exp_file
        if not expected_pdf.exists():
            logger.warning("In-scope row %s specifies '%s' which was not found in %s", row["id"], exp_file, data_path)

    return df


def generate_template(
    output_path: Path | str,
    index_dir: Optional[Path | str] = None,
    config: Optional[Config] = None,
) -> Path:
    """Generate eval/questions.csv with 30 diverse chunk samples and 10 out-of-scope rows.

    Args:
        output_path: Path to write template CSV.
        index_dir: Path to index directory to draw sample chunks from.
        config: Optional Config instance.

    Returns:
        Path to written CSV file.
    """
    cfg = config or DEFAULT_CONFIG
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    artifacts = load_index(index_dir=index_dir, config=cfg, warn_if_stale=False)
    chunks = artifacts.chunks

    rows = []
    # 1. Generate 30 in-scope rows sampled across diverse files and pages
    if chunks:
        # Group chunks by file to ensure diversity across files
        by_file: dict[str, List[Chunk]] = {}
        for c in chunks:
            by_file.setdefault(c.file_name, []).append(c)

        sampled_chunks: List[Chunk] = []
        files = list(by_file.keys())
        stride = max(1, len(chunks) // 30)

        # Stride through all chunks to get even coverage
        for i in range(30):
            idx = (i * stride) % len(chunks)
            sampled_chunks.append(chunks[idx])

        difficulties = ["easy", "paraphrased", "multi_sentence"]
        for i, chunk in enumerate(sampled_chunks, 1):
            diff = difficulties[(i - 1) % len(difficulties)]
            rows.append(
                {
                    "id": f"in_{i:02d}",
                    "type": "in_scope",
                    "question": "",  # Student writes question by hand
                    "expected_file": chunk.file_name,
                    "expected_page": chunk.page_number,
                    "expected_answer": "",  # Student writes expected answer by hand
                    "difficulty": diff,
                    "source_chunk": chunk.text[:350].replace("\n", " "),
                }
            )
    else:
        logger.warning("No chunks available in index. Creating blank in-scope template rows.")
        for i in range(1, 31):
            rows.append(
                {
                    "id": f"in_{i:02d}",
                    "type": "in_scope",
                    "question": "",
                    "expected_file": "TODO_file.pdf",
                    "expected_page": 1,
                    "expected_answer": "",
                    "difficulty": "easy",
                    "source_chunk": "TODO: Run indexer first or inspect PDF page to formulate question.",
                }
            )

    # 2. Add 10 pre-filled real out-of-scope trivia rows
    for i, oos in enumerate(PREFILLED_OUT_OF_SCOPE, 1):
        rows.append(
            {
                "id": f"out_{i:02d}",
                "type": oos["type"],
                "question": oos["question"],
                "expected_file": oos["expected_file"],
                "expected_page": oos["expected_page"],
                "expected_answer": oos["expected_answer"],
                "difficulty": oos["difficulty"],
                "source_chunk": oos["source_chunk"],
            }
        )

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Successfully generated questions template at %s with %d rows (30 in-scope, 10 out-of-scope).", out_file, len(rows))
    return out_file


def main() -> None:
    """CLI entry point for question template generation and validation."""
    parser = argparse.ArgumentParser(description="BuBot Questions Template Generator & Validator")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: eval/questions.csv)")
    parser.add_argument("--validate", type=str, default=None, help="Validate an existing CSV file")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    if args.validate:
        val_path = Path(args.validate).resolve()
        logger.info("Validating questions CSV: %s", val_path)
        df = validate_questions_csv(val_path, data_dir=cfg.DATA_DIR)
        print(f"Validation SUCCESS: {len(df)} rows correctly formatted.")
    else:
        out_path = Path(args.out).resolve() if args.out else cfg.QUESTIONS_PATH
        generate_template(output_path=out_path, config=cfg)


if __name__ == "__main__":
    main()

