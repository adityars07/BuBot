"""Unit tests for document ingestion and chunking."""

from pathlib import Path
import pytest

from src.config import Config
from src.ingest import (
    PageDoc,
    chunk_pages,
    clean_text,
    detect_repeated_headers_footers,
    extract_pptx_slides,
    load_pdfs,
)


def test_page_numbers_are_one_indexed(sample_pdf: Path) -> None:
    """Verify extracted PageDoc objects use 1-indexed page numbering."""
    cfg = Config()
    pages = load_pdfs(sample_pdf.parent, config=cfg)
    assert len(pages) == 3
    assert [p.page_number for p in pages] == [1, 2, 3]


def test_chunks_never_cross_pages() -> None:
    """Verify chunks do not merge or span across distinct pages."""
    pages = [
        PageDoc(file_name="test.pdf", page_number=1, text="Page 1 sentence one. Page 1 sentence two."),
        PageDoc(file_name="test.pdf", page_number=2, text="Page 2 sentence one. Page 2 sentence two."),
    ]
    chunks = chunk_pages(pages, chunk_size=40, overlap=10, min_chunk_chars=10)

    for c in chunks:
        # Verify page number is strictly 1 or 2, matching chunk ID tag
        assert f"::p{c.page_number}::" in c.chunk_id
        if c.page_number == 1:
            assert "Page 2" not in c.text
        elif c.page_number == 2:
            assert "Page 1" not in c.text


def test_chunk_ids_deterministic() -> None:
    """Verify chunk IDs are stable, deterministic, and reproducible."""
    pages = [
        PageDoc(file_name="notes.pdf", page_number=5, text="First chunk test. Second chunk test."),
    ]
    chunks_run_1 = chunk_pages(pages, chunk_size=30, overlap=5, min_chunk_chars=10)
    chunks_run_2 = chunk_pages(pages, chunk_size=30, overlap=5, min_chunk_chars=10)

    assert len(chunks_run_1) == len(chunks_run_2)
    for c1, c2 in zip(chunks_run_1, chunks_run_2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.chunk_id == "notes.pdf::p5::c0" or "::c" in c1.chunk_id


def test_clean_text_removes_hyphenation() -> None:
    """Verify hyphenated line breaks are repaired into single words."""
    raw = "This is a back-\npropagation algorithm test with connec-\n  tion."
    cleaned = clean_text(raw)
    assert "backpropagation" in cleaned
    assert "connection" in cleaned
    assert "-\n" not in cleaned


def test_clean_text_removes_page_numbers() -> None:
    """Verify standalone page numbers and headers/footers are stripped."""
    raw = (
        "Header title\n"
        "Machine learning concepts.\n"
        "- 1 -\n"
        "Page 1 of 10\n"
        "42\n"
        "End of page text."
    )
    cleaned = clean_text(raw)
    assert "- 1 -" not in cleaned
    assert "Page 1 of 10" not in cleaned
    assert "42" not in cleaned
    assert "Machine learning concepts." in cleaned


def test_clean_text_preserves_paragraphs() -> None:
    """Verify paragraph breaks are retained as double newlines."""
    raw = "Paragraph 1 first line.\nParagraph 1 second line.\n\n\nParagraph 2 single line."
    cleaned = clean_text(raw)
    assert "\n\n" in cleaned
    assert cleaned.count("\n\n") == 1
    assert "Paragraph 1 first line. Paragraph 1 second line." in cleaned


def test_repeated_headers_footers_detection() -> None:
    """Verify detection of lines repeating across >50% of pages."""
    pages = [
        "Course CS101\nLecture 1 content here.\nFooter notice",
        "Course CS101\nLecture 2 content here.\nFooter notice",
        "Course CS101\nLecture 3 content here.\nFooter notice",
    ]
    repeated = detect_repeated_headers_footers(pages, threshold_ratio=0.5)
    assert "Course CS101" in repeated
    assert "Footer notice" in repeated
    assert "Lecture 1 content here." not in repeated


def test_empty_or_corrupt_pdf_does_not_crash(tmp_path: Path) -> None:
    """Verify ingestion gracefully handles corrupted or empty files without crashing."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4 non-valid garbage data inside")

    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    cfg = Config()
    pages = load_pdfs(tmp_path, config=cfg)
    # Should skip corrupt/empty without raising unhandled exceptions
    assert isinstance(pages, list)


def test_pptx_extraction_and_chunking(tmp_path: Path) -> None:
    """Verify PowerPoint presentations (.pptx) extract slides and chunk per slide."""
    import io
    import zipfile

    slide1_xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
    <p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Lecture 1: Convolutional Neural Networks and Filters</a:t></a:r></a:p>
        <a:p><a:r><a:t>Feature maps capture spatial hierarchies in images.</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>"""

    slide2_xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
    <p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>
      <p:cSld><p:spTree><p:sp><p:txBody>
        <a:p><a:r><a:t>Backpropagation Algorithm in Deep Learning</a:t></a:r></a:p>
        <a:p><a:r><a:t>Computes gradients of loss using calculus chain rule backwards.</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>"""

    pptx_file = tmp_path / "deep_learning.pptx"
    with zipfile.ZipFile(pptx_file, "w") as z:
        z.writestr("ppt/slides/slide1.xml", slide1_xml)
        z.writestr("ppt/slides/slide2.xml", slide2_xml)

    # 1. Test raw slide extraction
    slides = extract_pptx_slides(pptx_file)
    assert len(slides) == 2
    assert "Convolutional Neural Networks" in slides[0]
    assert "Backpropagation Algorithm" in slides[1]

    # 2. Test loading via load_pdfs directory scanner
    pages = load_pdfs(tmp_path)
    assert len(pages) == 2
    assert [p.page_number for p in pages] == [1, 2]
    assert pages[0].file_name == "deep_learning.pptx"
    assert pages[1].file_name == "deep_learning.pptx"

    # 3. Test chunking never crosses slides
    chunks = chunk_pages(pages, chunk_size=50, overlap=10, min_chunk_chars=10)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.file_name == "deep_learning.pptx"
        if c.page_number == 1:
            assert "Backpropagation" not in c.text
        elif c.page_number == 2:
            assert "Convolutional" not in c.text

