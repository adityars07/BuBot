"""BuBot: Local Grounded Course Notes RAG Streamlit Application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import streamlit as st

from bubot import Answer, RAGPipeline, Settings, Source

st.set_page_config(page_title="BuBot - Ask your own notes", page_icon="B", layout="wide", initial_sidebar_state="expanded")

# Injected CSS (< 60 lines, Space Grotesk + Inter, deep teal #0F6E5E, no scrollbar, single-page fit)
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, .stApp { font-family: 'Inter', sans-serif; color: #16181D; }
    h1, h2, h3, h4, .stHeading { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700; color: #16181D; }
    .main .block-container { max-width: 820px; padding: 1rem 1.2rem 1.5rem 1.2rem !important; }
    .main [data-testid="stVerticalBlock"] { gap: 0.55rem !important; }
    [data-testid="stSidebar"] { background-color: #EFEDE6; }
    [data-testid="stSidebarContent"] { padding: 1rem 0.85rem !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
    [data-testid="stSidebarContent"], [data-testid="stSidebar"], [data-testid="stAppViewContainer"], body { scrollbar-width: none !important; -ms-overflow-style: none !important; }
    [data-testid="stSidebarContent"]::-webkit-scrollbar, [data-testid="stSidebar"]::-webkit-scrollbar, [data-testid="stAppViewContainer"]::-webkit-scrollbar { display: none !important; width: 0 !important; }
    .bubot-card { background: #FFFFFF; border: 1px solid #E1DED3; border-radius: 12px; padding: 0.65rem 0.8rem; margin-bottom: 0.3rem; }
    .bubot-empty-box { border: 1.5px dashed #C8C5BB; border-radius: 12px; padding: 0.75rem 0.6rem; text-align: center; color: #5B606B; font-size: 0.82rem; }
    .bubot-pill { display: inline-flex; align-items: center; background: #E2F0EC; color: #0B5448; font-size: 0.75rem; font-weight: 600; padding: 2px 9px; border-radius: 20px; }
    .bubot-pill-neutral { display: inline-flex; align-items: center; background: #EAE8E0; color: #5B606B; font-size: 0.75rem; font-weight: 600; padding: 2px 9px; border-radius: 20px; }
    [data-testid="stChatMessage"]:has([aria-label*="user"]) { flex-direction: row-reverse; text-align: right; }
    [data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] { background-color: #16181D; color: #FFFFFF; border-radius: 16px 16px 4px 16px; padding: 0.55rem 0.9rem; }
    .step-card { background: #FFFFFF; border: 1px solid #E1DED3; border-radius: 12px; padding: 0.75rem; min-height: 95px; }
    .step-num { display: inline-block; width: 20px; height: 20px; background: #16181D; color: #FFF; border-radius: 5px; text-align: center; font-size: 0.75rem; font-weight: 700; line-height: 20px; margin-bottom: 0.25rem; }
    .match-bar-bg { height: 4px; width: 100%; background: #E1DED3; border-radius: 2px; margin: 3px 0 6px 0; overflow: hidden; }
    .match-bar-fill { height: 100%; border-radius: 2px; }
    .main [data-testid="stFileUploaderDropzone"] { background: #FFFFFF !important; border: 1.5px dashed #C8C5BB !important; border-radius: 14px !important; padding: 1.3rem 1rem !important; text-align: center !important; }
    .main [data-testid="stFileUploaderDropzone"] button { background-color: #0F6E5E !important; color: #FFFFFF !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] { margin: -0.25rem 0 !important; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Load and cache RAGPipeline once with user-friendly failure instructions."""
    try:
        return RAGPipeline.from_settings(Settings.from_env())
    except Exception as exc:
        st.error(
            f"Failed to load BuBot pipeline: {exc}\n\n"
            "Please run: `python scripts/download_models.py`\n"
            "Or launch in offline lite mode: `BUBOT_BACKEND=lite streamlit run app.py`"
        )
        st.stop()


pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []


def index_files(files: List[Any]) -> None:
    """Save uploaded files to data_dir and index through the pipeline."""
    if not files:
        return
    saved = []
    pipeline.settings.data_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = pipeline.settings.data_dir / f.name
        target.write_bytes(f.getbuffer())
        saved.append(target)
    with st.spinner("Indexing notes..."):
        res = pipeline.add_documents(saved)
    for fname, count in res.items():
        if count == 0:
            st.sidebar.warning(f"{fname}: no text found, a scanned PDF needs OCR")
        else:
            st.sidebar.success(f"{fname}: {count} chunks indexed")
    st.rerun()


def render_match_bar(score: float) -> str:
    """Render match strength progress bar paired with descriptive label."""
    pct = max(0, min(100, int(score * 100)))
    color, word = ("#0F6E5E", "Strong") if score >= 0.60 else (("#2E7D32", "Good") if score >= 0.45 else ("#5B606B", "Related"))
    return (
        f'<div style="font-size:0.75rem; color:{color}; font-weight:600;">{word} ({score:.2f})</div>'
        f'<div class="match-bar-bg"><div class="match-bar-fill" style="width:{pct}%; background-color:{color};"></div></div>'
    )


# Sidebar Layout
with st.sidebar:
    st.markdown(
        """<div style="display:flex; align-items:center; gap:8px; margin-bottom:0.6rem;">
            <div style="width:32px; height:32px; background:#0F6E5E; border-radius:9px; display:flex; align-items:center; justify-content:center;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700; line-height:1.1;">BuBot</div>
                <div style="font-size:0.72rem; color:#5B606B;">Ask your own notes</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if st.button("+ New chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    docs_map = pipeline.store.sources()
    st.markdown(
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.6rem; margin-bottom:0.25rem;">'
        f'<span style="font-size:0.72rem; font-weight:700; color:#5B606B; letter-spacing:0.05em;">DOCUMENTS</span>'
        f'<span style="font-size:0.72rem; color:#5B606B;">{len(docs_map)} indexed</span></div>',
        unsafe_allow_html=True,
    )

    if not docs_map:
        st.markdown('<div class="bubot-empty-box">No documents yet. Add your notes to get started.</div>', unsafe_allow_html=True)
    else:
        for fname, count in sorted(docs_map.items()):
            c_card, c_del = st.columns([5, 1])
            with c_card:
                st.markdown(
                    f'<div class="bubot-card" style="padding:0.4rem 0.6rem; margin-bottom:0.25rem;">'
                    f'<div style="font-weight:600; font-size:0.8rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{fname}">{fname}</div>'
                    f'<div style="font-size:0.7rem; color:#5B606B;">{count} chunks</div></div>',
                    unsafe_allow_html=True,
                )
            with c_del:
                if st.button("X", key=f"del_{fname}", help=f"Remove {fname}"):
                    pipeline.remove_document(fname)
                    st.rerun()

    sb_files = st.file_uploader("Upload notes", type=["pdf", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
    if st.button("Index documents", key="sb_idx_btn", use_container_width=True):
        index_files(sb_files)

    st.markdown('<div style="font-size:0.72rem; font-weight:700; color:#5B606B; letter-spacing:0.05em; margin-top:0.6rem; margin-bottom:0.2rem;">SETTINGS</div>', unsafe_allow_html=True)
    pipeline.top_k = st.slider("Passages used (k)", 1, 6, int(pipeline.top_k), 1)
    pipeline.threshold = st.slider("Refuse below match score", 0.0, 0.9, float(pipeline.threshold), 0.01, help="Higher means BuBot refuses more often")

    gen_label = pipeline.generator if pipeline.generator else "none (shows best passage)"
    st.markdown(
        f"""<div class="bubot-card" style="margin-top:0.6rem; padding:0.6rem 0.75rem;">
            <div style="font-size:0.68rem; font-weight:700; color:#5B606B; letter-spacing:0.05em; margin-bottom:0.25rem;">PIPELINE</div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:0.15rem;"><span style="color:#5B606B;">Embeddings</span><span style="font-weight:600;">{pipeline.embedder.name}</span></div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:0.15rem;"><span style="color:#5B606B;">Generator</span><span style="font-weight:600;">{gen_label}</span></div>
            <div style="display:flex; justify-content:space-between; font-size:0.75rem;"><span style="color:#5B606B;">Chunks used (k)</span><span style="font-weight:600;">{pipeline.top_k}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

# Main Content Area
if len(pipeline.store) == 0:
    st.markdown('<h1 style="font-size:40px; margin-bottom:0.25rem; line-height:1.15;">Ask your notes anything.</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.98rem; color:#5B606B; margin-bottom:1rem;">Upload your course PDFs and BuBot answers from them only, with the exact source shown for every answer. If it isn\'t in your notes, it tells you.</p>', unsafe_allow_html=True)

    main_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="main_files",
    )
    if main_files:
        if st.button("Index documents", type="primary", key="main_idx_btn", use_container_width=True):
            index_files(main_files)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="step-card"><span class="step-num">1</span><div style="font-weight:700; margin-bottom:0.2rem;">Upload</div><div style="font-size:0.8rem; color:#5B606B; line-height:1.3;">Add notes, slides or textbook chapters as PDFs.</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="step-card"><span class="step-num">2</span><div style="font-weight:700; margin-bottom:0.2rem;">Index</div><div style="font-size:0.8rem; color:#5B606B; line-height:1.3;">BuBot splits the text and builds a searchable index.</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="step-card"><span class="step-num">3</span><div style="font-weight:700; margin-bottom:0.2rem;">Ask</div><div style="font-size:0.8rem; color:#5B606B; line-height:1.3;">Get answers grounded in your own material.</div></div>', unsafe_allow_html=True)

    st.chat_input("Upload a document to start asking", disabled=True)

else:
    st.markdown(
        '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">'
        '<h2 style="margin:0; font-size:1.35rem;">Course Notes Q&A</h2>'
        '<span class="bubot-pill">Runs locally on your laptop</span></div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="B"):
                if msg.get("refused"):
                    st.markdown('<span class="bubot-pill-neutral">Not found in your documents</span>', unsafe_allow_html=True)
                    st.write(msg["content"])
                    if msg.get("reason") == "no_documents":
                        st.info("Upload your notes to start asking.")
                    elif msg.get("sources"):
                        st.caption("CLOSEST PASSAGES")
                        for idx, s in enumerate(msg["sources"], 1):
                            with st.expander(f"{idx}. {s['source']} - page {s['page']} - match {s['score']:.2f}"):
                                st.markdown(render_match_bar(s["score"]), unsafe_allow_html=True)
                                st.write(s["text"])
                else:
                    st.write(msg["content"])
                    if msg.get("sources"):
                        st.caption("SOURCES")
                        for idx, s in enumerate(msg["sources"], 1):
                            with st.expander(f"{idx}. {s['source']} - page {s['page']} - match {s['score']:.2f}"):
                                st.markdown(render_match_bar(s["score"]), unsafe_allow_html=True)
                                st.write(s["text"])

                st.caption(f"{msg.get('latency_s', 0.0):.1f}s - best match {msg.get('top_score', 0.0):.2f}")
                st.code(msg["content"], language=None)

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching your notes..."):
            try:
                ans: Answer = pipeline.ask(prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans.text,
                    "refused": ans.refused,
                    "reason": ans.reason,
                    "sources": [s.to_dict() for s in ans.sources],
                    "top_score": ans.top_score,
                    "latency_s": ans.latency_s,
                })
            except Exception as e:
                st.error(f"Error querying BuBot: {e}")
        st.rerun()
