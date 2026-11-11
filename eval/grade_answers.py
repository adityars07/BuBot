"""Manual Answer Grading CLI for BuBot Evaluation.

Interactive tool that steps through an evaluation results CSV, displays the
question, expected answer, model answer, and retrieved chunks, prompts the
reviewer for a grade (c/p/w), and writes progress incrementally (resumable).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_eval_csv(results_dir: Path) -> Optional[Path]:
    """Find the most recently modified run CSV in the results directory."""
    csv_files = sorted(results_dir.glob("run_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csv_files[0] if csv_files else None


def grade_csv(csv_path: Path) -> None:
    """Run interactive terminal grading loop on evaluation CSV."""
    df = pd.read_csv(csv_path)

    if "grade" not in df.columns:
        df["grade"] = ""

    in_scope_mask = df["type"] == "in_scope"
    total_in_scope = in_scope_mask.sum()

    print("\n" + "=" * 76)
    print(f"BUBOT INTERACTIVE GRADING SESSION: {csv_path.name}")
    print(f"Total in-scope questions to review: {total_in_scope}")
    print("Grade Keys: [c] Correct (1.0) | [p] Partial (0.5) | [w] Wrong (0.0) | [s] Skip | [q] Save & Quit")
    print("=" * 76 + "\n")

    graded_count = 0
    scores = []

    try:
        for idx, row in df.iterrows():
            if row["type"] != "in_scope":
                continue

            current_grade = str(row.get("grade", "")).strip()
            if pd.notna(row.get("grade")) and current_grade:
                # Already graded
                if current_grade in ("c", "1.0", "1"):
                    scores.append(1.0)
                elif current_grade in ("p", "0.5"):
                    scores.append(0.5)
                elif current_grade in ("w", "0.0", "0"):
                    scores.append(0.0)
                graded_count += 1
                continue

            print("\n" + "-" * 76)
            print(f"Question ID:      {row['id']} (Difficulty: {row.get('difficulty', 'n/a')})")
            print(f"Question:         {row['question']}")
            print(f"Expected Answer:  {row['expected_answer']}")
            print(f"Expected Source:  {row['expected_file']} (p.{row['expected_page']})")
            print(f"Model Answer:     {row['model_answer']}")
            print(f"Refused:          {row.get('refused')} | Top Score: {row.get('top_score')}")
            print(f"Sources:          {row.get('sources_retrieved')}")
            print("-" * 76)

            while True:
                choice = input("Enter grade [c / p / w / s / q]: ").strip().lower()
                if choice in ("c", "p", "w"):
                    df.at[idx, "grade"] = choice
                    score = 1.0 if choice == "c" else (0.5 if choice == "p" else 0.0)
                    scores.append(score)
                    graded_count += 1
                    df.to_csv(csv_path, index=False, encoding="utf-8")
                    break
                elif choice == "s":
                    print("Skipping question.")
                    break
                elif choice == "q":
                    print("Saving and exiting session.")
                    df.to_csv(csv_path, index=False, encoding="utf-8")
                    return
                else:
                    print("Invalid input. Please enter 'c' (correct), 'p' (partial), 'w' (wrong), 's' (skip), or 'q' (quit).")

    except KeyboardInterrupt:
        print("\nSession interrupted. Saving current progress...")
        df.to_csv(csv_path, index=False, encoding="utf-8")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print("\n" + "=" * 76)
    print(f"Grading session complete. Graded {graded_count}/{total_in_scope} in-scope questions.")
    if scores:
        print(f"Official Manual Accuracy Score: {sum(scores) / len(scores) * 100:.1f}% ({len(scores)} answers graded)")
    print(f"Updated CSV saved to: {csv_path}")
    print("=" * 76 + "\n")


def main() -> None:
    """CLI entry point for grading."""
    parser = argparse.ArgumentParser(description="BuBot Interactive Manual Grading CLI")
    parser.add_argument("--file", type=str, default=None, help="Path to evaluation run CSV")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    if args.file:
        csv_path = Path(args.file).resolve()
    else:
        csv_path = find_latest_eval_csv(cfg.RESULTS_DIR)
        if not csv_path:
            logger.error("No evaluation run CSV found in %s. Specify --file.", cfg.RESULTS_DIR)
            return

    grade_csv(csv_path)


if __name__ == "__main__":
    main()

