"""PDF Ingestion, Text Cleaning, and Page-Level Chunking for BuBot.

Extracts text from PDF documents with page numbers preserved (1-indexed), cleans
artifacts and repetitive headers/footers, and splits text per page using
RecursiveCharacterTextSplitter without crossing page boundaries.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from src.config import DEFAULT_CONFIG, Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageDoc:
    """Document representation of a single PDF page."""

    file_name: str
    page_number: int  # 1-indexed
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage tagged with exact file and 1-indexed page."""

    chunk_id: str
    file_name: str
    page_number: int  # 1-indexed
    text: str
    char_len: int

    def to_dict(self) -> dict:
        """Serialize chunk to a dictionary."""
        return asdict(self)


def detect_repeated_headers_footers(raw_pages_text: Sequence[str], threshold_ratio: float = 0.5) -> Set[str]:
    """Identify lines appearing on > 50% of pages in a document (headers/footers).

    Args:
        raw_pages_text: List of raw string texts for each page in a single file.
        threshold_ratio: Minimum fraction of pages a line must occur on to be flagged.

    Returns:
        Set of normalized line strings to be stripped.
    """
    total_pages = len(raw_pages_text)
    if total_pages < 2:
        return set()

    line_page_counts: Counter[str] = Counter()
    for page in raw_pages_text:
        seen_on_page: Set[str] = set()
        for raw_line in page.splitlines():
            line = raw_line.strip()
            # Ignore trivial lines or pure whitespace
            if len(line) >= 4 and not line.isdigit():
                seen_on_page.add(line)
        for line in seen_on_page:
            line_page_counts[line] += 1

    repeated: Set[str] = {
        line
        for line, count in line_page_counts.items()
        if (count / total_pages) > threshold_ratio
    }
    return repeated


def clean_text(text: str, header_footer_lines: Optional[Set[str]] = None) -> str:
    """Clean extracted page text while preserving paragraph breaks.

    Performs:
    - Unicode normalization (NFKC decomposes ligatures like 'fi' -> 'fi').
    - Repetitive header and footer line stripping.
    - Page-number-only line stripping (e.g. '12', '- 12 -', 'Page 12 of 30').
    - Hyphenated line-break repair ('net-\nwork' -> 'network').
    - Whitespace normalization while retaining paragraph breaks ('\\n\\n').

    Args:
        text: Raw page text string.
        header_footer_lines: Optional set of lines detected as recurring headers/footers.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # 1. Unicode normalization (handles ligatures like 'fi', 'fl', quotes, em-dashes)
    text = unicodedata.normalize("NFKC", text)

    # 2. Fix hyphenated line breaks (e.g. "connec-\ntion" -> "connection")
    text = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", text)

    # 3. Filter lines (page numbers, headers/footers)
    header_footer_set = header_footer_lines or set()
    cleaned_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # Strip detected repeated header/footer lines
        if line in header_footer_set:
            continue

        # Strip standalone page numbers: "1", "- 1 -", "[1]", "Page 1", "Page 1 of 10"
        if re.match(r"^(\[?-?\s*\d+\s*-?\]?|page\s+\d+(\s+of\s+\d+)?)$", line, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    filtered_text = "\n".join(cleaned_lines)

    # 4. Standardize paragraph breaks: collapse 2 or more newlines into double newline
    paragraphs = re.split(r"\n\s*\n+", filtered_text)
    processed_paragraphs: List[str] = []

    for para in paragraphs:
        # Collapse multiple spaces or single internal newlines to a single space
        para_collapsed = re.sub(r"\s+", " ", para).strip()
        if para_collapsed:
            processed_paragraphs.append(para_collapsed)

    return "\n\n".join(processed_paragraphs)


def _extract_page_with_fallback(
    pypdf_page,
    pdf_path: Path,
    page_idx: int,
    file_name: str,
    min_page_chars: int,
    use_ocr: bool = False,
) -> str:
    """Extract page text using pypdf, falling back to pdfplumber or OCR if needed."""
    page_num = page_idx + 1
    text = ""
    try:
        text = pypdf_page.extract_text() or ""
    except Exception as exc:
        logger.warning("pypdf extraction failed on %s p.%d: %s", file_name, page_num, exc)

    # Fallback to pdfplumber if extracted text is too short or garbled
    if len(text.strip()) < min_page_chars:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as plumber_pdf:
                if page_idx < len(plumber_pdf.pages):
                    plumber_text = plumber_pdf.pages[page_idx].extract_text() or ""
                    if len(plumber_text.strip()) >= min_page_chars:
                        text = plumber_text
        except Exception as exc:
            logger.warning("pdfplumber fallback failed on %s p.%d: %s", file_name, page_num, exc)

    # Scanned page check / Optional OCR fallback
    if len(text.strip()) < min_page_chars:
        if use_ocr:
            try:
                import pytesseract
                from pdf2image import convert_from_path
                images = convert_from_path(str(pdf_path), first_page=page_num, last_page=page_num)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    if len(ocr_text.strip()) >= min_page_chars:
                        text = ocr_text
            except Exception as exc:
                logger.warning("OCR fallback failed on %s p.%d: %s", file_name, page_num, exc)
        else:
            logger.warning(
                "Page %d in %s extracted fewer than %d characters (%d found). "
                "The page may be scanned, an image, or empty.",
                page_num, file_name, min_page_chars, len(text.strip())
            )

    return text


def extract_pptx_slides(pptx_path: Path | str) -> List[str]:
    """Extract text slide-by-slide from PowerPoint presentations (.pptx, .ppt).

    Args:
        pptx_path: Path to PowerPoint presentation.

    Returns:
        List of extracted slide text strings (in slide order).
    """
    path = Path(pptx_path).resolve()
    if not path.exists():
        return []

    import re
    import xml.etree.ElementTree as ET
    import zipfile

    # Standard XML-based PPTX archive
    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as z:
                slide_names = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
                def slide_key(n: str) -> int:
                    m = re.search(r"slide(\d+)\.xml", n)
                    return int(m.group(1)) if m else 0
                slide_names.sort(key=slide_key)

                slides: List[str] = []
                for name in slide_names:
                    root = ET.fromstring(z.read(name))
                    paras: List[str] = []
                    for elem in root.iter():
                        if elem.tag.endswith("}p") or elem.tag == "p":
                            p_text = "".join(
                                t.text for t in elem.iter()
                                if (t.tag.endswith("}t") or t.tag == "t") and t.text
                            )
                            if p_text.strip():
                                paras.append(p_text.strip())
                    slides.append("\n".join(paras))
                return slides
        except Exception as exc:
            logger.warning("Failed to parse PPTX XML in %s: %s", path.name, exc)

    # Fallback for binary .ppt files
    try:
        raw_bytes = path.read_bytes()
        matches = re.findall(rb"[\x20-\x7E\t\r\n]{6,}", raw_bytes)
        text = "\n".join(m.decode("ascii", errors="ignore").strip() for m in matches if len(m.decode("ascii", errors="ignore").strip()) > 10)
        if text.strip():
            return [text]
    except Exception as exc:
        logger.warning("Binary fallback extraction failed for %s: %s", path.name, exc)

    return []


def load_pdfs(data_dir: Path | str, config: Optional[Config] = None) -> List[PageDoc]:
    """Scan data_dir and load all PDF pages into PageDoc instances.

    Args:
        data_dir: Path to directory containing course PDFs.
        config: Optional Config instance for tuning parameters.

    Returns:
        List of PageDoc instances (1-indexed page numbers).
    """
    cfg = config or DEFAULT_CONFIG
    dir_path = Path(data_dir).resolve()
    if not dir_path.exists():
        logger.warning("Data directory does not exist: %s. Creating empty directory.", dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return []

    pdf_files = sorted([p for p in dir_path.glob("*.[pP][dD][fF]") if p.is_file()])
    ppt_files = sorted([p for p in dir_path.glob("*.[pP][pP][tT]*") if p.is_file()])
    if not pdf_files and not ppt_files:
        logger.warning("No PDF or presentation files found in directory: %s", dir_path)
        return []

    from pypdf import PdfReader

    all_pages: List[PageDoc] = []

    for pdf_path in pdf_files:
        file_name = pdf_path.name
        try:
            reader = PdfReader(str(pdf_path))
            if reader.is_encrypted:
                try:
                    # Attempt empty password decryption
                    reader.decrypt("")
                except Exception:
                    logger.warning("PDF %s is encrypted and password-protected. Skipping.", file_name)
                    continue

            total_pages = len(reader.pages)
            if total_pages == 0:
                logger.warning("PDF %s has 0 pages. Skipping.", file_name)
                continue

            # First pass: collect raw text to detect repetitive headers/footers across pages
            raw_page_texts: List[str] = []
            for i, page in enumerate(reader.pages):
                raw_t = _extract_page_with_fallback(
                    pypdf_page=page,
                    pdf_path=pdf_path,
                    page_idx=i,
                    file_name=file_name,
                    min_page_chars=cfg.MIN_PAGE_CHARS,
                    use_ocr=cfg.USE_OCR,
                )
                raw_page_texts.append(raw_t)

            header_footers = detect_repeated_headers_footers(raw_page_texts, threshold_ratio=0.5)
            if header_footers:
                logger.debug("Detected %d repeated header/footer lines in %s", len(header_footers), file_name)

            # Second pass: clean text and create PageDoc objects
            for i, raw_text in enumerate(raw_page_texts):
                page_number = i + 1  # 1-indexed
                cleaned = clean_text(raw_text, header_footer_lines=header_footers)
                if cleaned:
                    all_pages.append(PageDoc(file_name=file_name, page_number=page_number, text=cleaned))

            logger.info("Loaded %s: %d pages parsed.", file_name, total_pages)

        except Exception as exc:
            logger.error("Failed to parse PDF %s: %s. Skipping corrupted or unsupported file.", file_name, exc)
            continue

    # Also parse PowerPoint presentations (.pptx, .ppt)
    for ppt_path in ppt_files:
        file_name = ppt_path.name
        try:
            slide_texts = extract_pptx_slides(ppt_path)
            if not slide_texts:
                continue
            header_footers = detect_repeated_headers_footers(slide_texts, threshold_ratio=0.5)
            for i, raw_text in enumerate(slide_texts):
                cleaned = clean_text(raw_text, header_footer_lines=header_footers)
                if cleaned:
                    all_pages.append(PageDoc(file_name=file_name, page_number=i + 1, text=cleaned))
            logger.info("Loaded %s: %d slides parsed.", file_name, len(slide_texts))
        except Exception as exc:
            logger.error("Failed to parse presentation %s: %s", file_name, exc)

    logger.info("Total PageDocs loaded across all documents: %d", len(all_pages))
    return all_pages


def chunk_pages(
    pages: Sequence[PageDoc],
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_chunk_chars: Optional[int] = None,
    config: Optional[Config] = None,
) -> List[Chunk]:
    """Split PageDoc pages into retrievable chunks.

    Never merges text across different pages. Each chunk retains exact file_name
    and 1-indexed page_number metadata.

    Args:
        pages: Sequence of PageDoc instances.
        chunk_size: Max characters per chunk (default from config).
        overlap: Character overlap between chunks (default from config).
        min_chunk_chars: Discard chunks shorter than this (default from config).
        config: Optional Config instance.

    Returns:
        List of Chunk instances with stable deterministic chunk_ids.
    """
    cfg = config or DEFAULT_CONFIG
    c_size = chunk_size if chunk_size is not None else cfg.CHUNK_SIZE
    c_overlap = overlap if overlap is not None else cfg.CHUNK_OVERLAP
    min_chars = min_chunk_chars if min_chunk_chars is not None else cfg.MIN_CHUNK_CHARS

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )

    all_chunks: List[Chunk] = []

    for doc in pages:
        if not doc.text:
            continue

        raw_splits = splitter.split_text(doc.text)
        valid_chunk_idx = 0

        for split_text in raw_splits:
            cleaned_split = split_text.strip()
            if len(cleaned_split) < min_chars:
                continue

            chunk_id = f"{doc.file_name}::p{doc.page_number}::c{valid_chunk_idx}"
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    file_name=doc.file_name,
                    page_number=doc.page_number,
                    text=cleaned_split,
                    char_len=len(cleaned_split),
                )
            )
            valid_chunk_idx += 1

    logger.info(
        "Generated %d chunks from %d pages (chunk_size=%d, overlap=%d, min_chars=%d).",
        len(all_chunks), len(pages), c_size, c_overlap, min_chars
    )
    return all_chunks


def preview_chunks(chunks: List[Chunk], sample_count: int = 10, seed: int = 42) -> None:
    """Print a random sample of chunks for visual quality verification."""
    if not chunks:
        print("No chunks available to preview.")
        return

    rng = random.Random(seed)
    k = min(sample_count, len(chunks))
    sample = rng.sample(chunks, k)

    print("\n" + "=" * 80)
    print(f"BUBOT CHUNK PREVIEW: {k} SAMPLE CHUNKS (Total: {len(chunks)})")
    print("=" * 80)
    for i, c in enumerate(sample, 1):
        print(f"\n--- [Sample {i}/{k}] ID: {c.chunk_id} ---")
        print(f"File: {c.file_name} | Page: {c.page_number} | Length: {c.char_len} chars")
        print("Text Preview:")
        preview_text = c.text if len(c.text) <= 280 else c.text[:280] + "..."
        print(preview_text)
    print("\n" + "=" * 80 + "\n")


def main() -> None:
    """CLI entry point for ingest module."""
    parser = argparse.ArgumentParser(description="BuBot PDF Ingestion & Chunking CLI")
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of sample chunks to preview (default: 10)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing PDF files",
    )
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    data_dir = Path(args.data_dir).resolve() if args.data_dir else cfg.DATA_DIR
    pages = load_pdfs(data_dir=data_dir, config=cfg)
    chunks = chunk_pages(pages=pages, config=cfg)
    preview_chunks(chunks=chunks, sample_count=args.preview, seed=cfg.RANDOM_SEED)


if __name__ == "__main__":
    main()

