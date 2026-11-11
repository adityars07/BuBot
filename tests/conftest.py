"""Test fixtures and PDF generators for BuBot test suite."""

from __future__ import annotations

from pathlib import Path
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.config import Config
from src.ingest import Chunk, PageDoc


def generate_mock_pdf(pdf_path: Path) -> Path:
    """Generate a multi-page mock PDF with deterministic content for testing."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Page 1
    c.drawString(72, 750, "Course Notes: Machine Learning Basics")
    c.drawString(72, 720, "Supervised learning uses labeled training data to learn mapping functions.")
    c.drawString(72, 700, "Common supervised algorithms include linear regression and decision trees.")
    c.drawString(72, 50, "- 1 -")
    c.showPage()

    # Page 2
    c.drawString(72, 750, "Course Notes: Machine Learning Basics")
    c.drawString(72, 720, "Backpropagation is an algorithm used to calculate gradients of the loss.")
    c.drawString(72, 700, "It applies the chain rule of calculus in reverse order through layers.")
    c.drawString(72, 50, "- 2 -")
    c.showPage()

    # Page 3
    c.drawString(72, 750, "Course Notes: Machine Learning Basics")
    c.drawString(72, 720, "Gradient descent updates parameters iteratively in the opposite gradient direction.")
    c.drawString(72, 700, "The learning rate controls the step size taken towards the local minimum.")
    c.drawString(72, 50, "- 3 -")
    c.showPage()

    c.save()
    return pdf_path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Provide a generated 3-page course sample PDF."""
    pdf_file = tmp_path / "course_notes.pdf"
    return generate_mock_pdf(pdf_file)


@pytest.fixture
def mock_chunks() -> list[Chunk]:
    """Provide hand-crafted chunks for testing indexing and retrieval."""
    return [
        Chunk(
            chunk_id="course.pdf::p1::c0",
            file_name="course.pdf",
            page_number=1,
            text="Supervised learning maps input features to target labels using training data.",
            char_len=75,
        ),
        Chunk(
            chunk_id="course.pdf::p2::c0",
            file_name="course.pdf",
            page_number=2,
            text="Backpropagation calculates gradients of the neural network loss function using the chain rule.",
            char_len=97,
        ),
        Chunk(
            chunk_id="course.pdf::p3::c0",
            file_name="course.pdf",
            page_number=3,
            text="Gradient descent optimizes loss iteratively using a step size controlled by the learning rate.",
            char_len=95,
        ),
    ]

