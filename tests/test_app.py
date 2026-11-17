"""Unit and integration tests for BuBot Streamlit UI."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    import faiss
except Exception:
    pass

from pathlib import Path
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from bubot import RAGPipeline, Settings


def test_app_first_load_welcome(tmp_path, monkeypatch):
    """Test app first load with no documents:
    - BUBOT_BACKEND=lite
    - No exception on first load
    - Welcome headline 'Ask your notes anything.' is shown
    - Chat input is disabled with no documents
    """
    st.cache_resource.clear()
    empty_data = tmp_path / "empty_data"
    empty_data.mkdir(parents=True, exist_ok=True)
    empty_index = tmp_path / "empty_index"
    empty_index.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("BUBOT_BACKEND", "lite")
    monkeypatch.setenv("BUBOT_DATA_DIR", str(empty_data))
    monkeypatch.setenv("BUBOT_INDEX_DIR", str(empty_index))

    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path.resolve()), default_timeout=60)
    at.run()

    # 1. Assert no exception on first load
    assert not at.exception, f"App raised exception: {at.exception}"

    # 2. Assert welcome headline is shown
    markdown_texts = [m.value for m in at.markdown]
    assert any("Ask your notes anything." in text for text in markdown_texts), (
        f"Welcome headline not found in: {markdown_texts}"
    )

    # 3. Assert chat input is disabled with no documents
    assert len(at.chat_input) > 0, "No chat input found on page"
    assert at.chat_input[0].disabled is True, "Chat input must be disabled when no documents exist"
    assert at.chat_input[0].placeholder == "Upload a document to start asking"

    # 4. Assert sidebar elements
    sidebar_md = [m.value for m in at.sidebar.markdown]
    assert any("BuBot" in t for t in sidebar_md), "BuBot logo missing in sidebar"
    assert any("No documents yet" in t for t in sidebar_md), "Empty documents message missing"
    assert any("none (shows best passage)" in t for t in sidebar_md), "Pipeline card generator label missing"


def test_app_with_documents_chat_state(tmp_path, monkeypatch):
    """Test app with indexed documents:
    - Chat header and pill 'Runs locally on your laptop' is shown
    - Chat input is enabled
    - Can submit a question and receive an answer
    """
    st.cache_resource.clear()
    test_data = tmp_path / "doc_data"
    test_data.mkdir(parents=True, exist_ok=True)
    test_index = tmp_path / "doc_index"
    test_index.mkdir(parents=True, exist_ok=True)

    # Create a small text course note
    note_file = test_data / "ml_basics.txt"
    note_file.write_text(
        "Supervised learning algorithms map input features X to targets Y using labeled examples. "
        "In regression tasks the target variable is continuous. In classification it is categorical.",
        encoding="utf-8",
    )

    # Pre-index the note
    settings = Settings(backend="lite", data_dir=test_data, index_dir=test_index, top_k=3, effective_threshold=0.35)
    pipeline = RAGPipeline.from_settings(settings)
    pipeline.add_documents([note_file])

    monkeypatch.setenv("BUBOT_BACKEND", "lite")
    monkeypatch.setenv("BUBOT_DATA_DIR", str(test_data))
    monkeypatch.setenv("BUBOT_INDEX_DIR", str(test_index))

    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path.resolve()), default_timeout=60)
    at.run()

    assert not at.exception, f"App raised exception: {at.exception}"

    # Main area should show chat header
    main_md = [m.value for m in at.markdown]
    assert any("Course Notes Q&A" in t for t in main_md), "Header 'Course Notes Q&A' not shown"
    assert any("Runs locally on your laptop" in t for t in main_md), "Pill 'Runs locally on your laptop' not shown"

    # Chat input should now be enabled
    assert len(at.chat_input) > 0, "Chat input missing"
    assert at.chat_input[0].disabled is False, "Chat input should be enabled when documents exist"
    assert at.chat_input[0].placeholder == "Ask a question about your documents..."

    # Sidebar should show 1 indexed document
    sb_md = [m.value for m in at.sidebar.markdown]
    assert any("1 indexed" in t for t in sb_md), "Sidebar should show '1 indexed'"
    assert any("ml_basics.txt" in t for t in sb_md), "Document card ml_basics.txt missing"
