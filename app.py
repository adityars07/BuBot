"""BuBot: Local Grounded Course Notes RAG Streamlit Application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import streamlit as st

from bubot import Answer, RAGPipeline, Settings, Source

st.set_page_config(page_title="BuBot - Ask your own notes", page_icon="B", layout="wide", initial_sidebar_state="expanded")

# Injected CSS (< 120 lines, Space Grotesk + Inter, deep teal #0F6E5E, flat cards, no emoji)
st.markdown(
    """<style>
    /* 1. Typography & Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp, [data-testid="stSidebar"], [data-testid="stSidebar"] *, button, input, textarea, [data-testid="stChatInput"] textarea, p, span, div, label { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    h1, h2, h3, h4, .stHeading, .bubot-wordmark { font-family: 'Space Grotesk', system-ui, sans-serif !important; font-weight: 700 !important; color: #16181D !important; }
    [data-testid="stIconMaterial"], [class*="material-symbols"] { font-family: 'Material Symbols Rounded', sans-serif !important; }
    /* 2. Top Spacing & Layout */
    header[data-testid="stHeader"] { background-color: transparent !important; height: 0px !important; }
    [data-testid="stSidebarContent"] { padding-top: 16px !important; padding-bottom: 24px !important; }
    .main .block-container, [data-testid="stMain"] .block-container { padding-top: 32px !important; padding-bottom: 40px !important; max-width: 820px !important; }
    /* 3. Sidebar Layout & Overlaps */
    [data-testid="stSidebar"] { background-color: #EFEDE6; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 12px !important; }
    .bubot-empty-box { border: 1.5px dashed #9A9788; border-radius: 14px; padding: 14px 12px; text-align: center; color: #5B606B; font-size: 13px; margin-bottom: 8px; }
    .bubot-card { background: #FFFFFF; border: 1px solid #E1DED3; border-radius: 14px; padding: 10px 12px; margin-bottom: 6px; }
    .bubot-pipeline-card { background: #FFFFFF; border: 1px solid #E1DED3; border-radius: 14px; padding: 12px 14px; margin-top: 32px; }
    .pipeline-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px !important; margin-bottom: 4px; }
    .pipeline-row span { color: #5B606B !important; font-size: 12px !important; }
    .pipeline-row b { color: #16181D !important; font-size: 12px !important; font-weight: 600 !important; text-align: right !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] * { font-size: 12.5px !important; color: #5B606B !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { border: 1.5px dashed #9A9788 !important; border-radius: 12px !important; background: #FFFFFF !important; padding: 8px !important; text-align: center !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { border: 1px solid #CFCCBF !important; border-radius: 8px !important; background: #F6F5F1 !important; color: #16181D !important; font-size: 12.5px !important; font-weight: 500 !important; width: 100% !important; height: 34px !important; }
    /* 4. Main Dropzone Area */
    main [data-testid="stFileUploader"], [data-testid="stMain"] [data-testid="stFileUploader"] { padding: 0 !important; margin-bottom: 24px !important; }
    main [data-testid="stFileUploaderDropzone"], [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] { background-color: #FFFFFF !important; border: 1.5px dashed #9A9788 !important; border-radius: 20px !important; padding: 40px 24px !important; display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; text-align: center !important; }
    main [data-testid="stFileUploaderDropzone"]::before, [data-testid="stMain"] [data-testid="stFileUploaderDropzone"]::before { content: ""; display: block; width: 52px; height: 52px; background-color: #E2F0EC; border-radius: 14px; margin-bottom: 14px; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='26' height='26' viewBox='0 0 24 24' fill='none' stroke='%230F6E5E' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E"); background-position: center; background-repeat: no-repeat; order: 1; }
    main [data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"] { display: flex !important; flex-direction: column !important; align-items: center !important; order: 2 !important; margin-bottom: 18px !important; }
    main [data-testid="stFileUploaderDropzoneInstructions"]::before, [data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"]::before { content: "Drop your PDFs or slides here" !important; font-size: 17px !important; font-weight: 600 !important; color: #16181D !important; margin-bottom: 4px !important; }
    main [data-testid="stFileUploaderDropzoneInstructions"]::after, [data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"]::after { content: "or choose files from your computer" !important; font-size: 14px !important; color: #5B606B !important; }
    main [data-testid="stFileUploaderDropzoneInstructions"] > div, [data-testid="stMain"] [data-testid="stFileUploaderDropzoneInstructions"] > div { display: none !important; }
    main [data-testid="stFileUploaderDropzone"] > span:has(button), [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] > span:has(button) { order: 3 !important; }
    main [data-testid="stFileUploaderDropzone"] button, [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button { background-color: #0F6E5E !important; color: #FFFFFF !important; height: 44px !important; border-radius: 12px !important; padding: 0 24px !important; border: none !important; }
    main [data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button:hover { background-color: #0B5448 !important; }
    main [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"], [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] { display: none !important; }
    main [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p, [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p { font-size: 0 !important; margin: 0 !important; }
    main [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after, [data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after { content: "Browse files" !important; font-size: 14px !important; font-weight: 600 !important; color: #FFFFFF !important; }
    /* 5. Step Cards */
    .step-card { background: #FBFAF7; border: 1px solid #E1DED3; border-radius: 14px; padding: 16px; height: 100%; min-height: 125px; display: flex; flex-direction: column; }
    .step-num { display: inline-block; width: 24px; height: 24px; background: #16181D; color: #FFFFFF; border-radius: 6px; text-align: center; font-size: 12.5px; font-weight: 700; line-height: 24px; margin-bottom: 8px; }
    .step-title { font-size: 15px; font-weight: 700; color: #16181D; margin-bottom: 4px; }
    .step-desc { font-size: 13px; color: #5B606B; line-height: 1.4; }
    /* 6. Chat Input & Messages */
    [data-testid="stChatInput"] { background-color: #FFFFFF !important; border: 1px solid #CFCCBF !important; border-radius: 18px !important; padding: 4px 8px !important; }
    [data-testid="stChatInput"] textarea { background-color: transparent !important; font-size: 14px !important; color: #16181D !important; }
    [data-testid="stChatInput"] button:disabled svg { fill: #C9C6BA !important; color: #C9C6BA !important; }
    [data-testid="stChatInput"] button:not(:disabled) { background-color: #0F6E5E !important; border-radius: 12px !important; }
    [data-testid="stChatInput"] button:not(:disabled) svg { fill: #FFFFFF !important; color: #FFFFFF !important; }
    [data-testid="stChatMessage"]:has([aria-label*="user"]) { flex-direction: row-reverse; text-align: right; }
    [data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageAvatar"], [data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stIconMaterial"] { display: none !important; }
    [data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] { background-color: #16181D; color: #FFFFFF; border-radius: 16px 16px 4px 16px; padding: 0.65rem 1rem; }
    .bubot-pill { display: inline-flex; align-items: center; background: #E2F0EC; color: #0B5448; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; }
    .bubot-pill-neutral { display: inline-flex; align-items: center; background: #EAE8E0; color: #5B606B; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; }
    .match-bar-bg { height: 4px; width: 100%; background: #E1DED3; border-radius: 2px; margin: 4px 0 8px 0; overflow: hidden; }
    .match-bar-fill { height: 100%; border-radius: 2px; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Load and cache RAGPipeline once with user-friendly instructions."""
    try:
        return RAGPipeline.from_settings(Settings.from_env())
    except Exception as exc:
        st.error(f"Failed to load BuBot pipeline: {exc}\n\nRun: `python scripts/download_models.py`\nOr: `BUBOT_BACKEND=lite streamlit run app.py`")
        st.stop()

pipeline = get_pipeline()
if "messages" not in st.session_state: st.session_state.messages = []


def index_files(files: List[Any]) -> None:
    """Save uploaded files to data_dir and index through the pipeline (Defect 2)."""
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
    return f'<div style="font-size:0.75rem; color:{color}; font-weight:600;">{word} ({score:.2f})</div><div class="match-bar-bg"><div class="match-bar-fill" style="width:{pct}%; background-color:{color};"></div></div>'


# Sidebar Layout (Defects 2, 3, 4, 7)
with st.sidebar:
    st.markdown(
        """<div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <div style="width:36px; height:36px; background:#0F6E5E; border-radius:10px; display:flex; align-items:center; justify-content:center;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
            </div>
            <div>
                <div class="bubot-wordmark" style="font-size:1.15rem; font-weight:700; line-height:1.2;">BuBot</div>
                <div style="font-size:0.75rem; color:#5B606B;">Ask your own notes</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    if st.button("+ New chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    docs_map = pipeline.store.sources()
    # Defect 3: DOCUMENTS label left, N indexed right, 8px below
    st.markdown(
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px; margin-bottom:8px;">'
        f'<span style="font-size:12px; font-weight:700; color:#5B606B; letter-spacing:0.05em;">DOCUMENTS</span>'
        f'<span style="font-size:12px; color:#5B606B;">{len(docs_map)} indexed</span></div>',
        unsafe_allow_html=True,
    )
    if not docs_map:
        st.markdown('<div class="bubot-empty-box">No documents yet. Add your notes to get started.</div>', unsafe_allow_html=True)
    else:
        for fname, count in sorted(docs_map.items()):
            c_card, c_del = st.columns([5, 1])
            with c_card:
                st.markdown(
                    f'<div class="bubot-card" style="padding:0.45rem 0.7rem; margin-bottom:0.3rem;">'
                    f'<div style="font-weight:600; font-size:0.83rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{fname}">{fname}</div>'
                    f'<div style="font-size:0.72rem; color:#5B606B;">{count} chunks</div></div>',
                    unsafe_allow_html=True,
                )
            with c_del:
                if st.button("X", key=f"del_{fname}", help=f"Remove {fname}"):
                    pipeline.remove_document(fname)
                    st.rerun()
        # Defect 2: Sidebar uploader only once documents exist
        sb_files = st.file_uploader("Add documents", type=["pdf", "pptx", "ppt", "txt", "md"], accept_multiple_files=True, key="sidebar_uploader", label_visibility="collapsed")
        if sb_files:
            index_files(sb_files)

    # Defect 3: Explicit headings without overlap
    st.markdown('<div style="font-size:12px; font-weight:700; color:#5B606B; letter-spacing:0.05em; margin-top:20px; margin-bottom:8px;">SETTINGS</div>', unsafe_allow_html=True)
    pipeline.top_k = st.slider("Passages used (k)", 1, 6, int(pipeline.top_k), 1)
    pipeline.threshold = st.slider("Refuse below match score", 0.0, 0.9, float(pipeline.threshold), 0.01, help="Higher means BuBot refuses more often")

    # Defect 7: Pipeline card at bottom
    gen_label = pipeline.generator if pipeline.generator else "none (shows best passage)"
    st.markdown(
        f"""<div class="bubot-pipeline-card">
            <div style="font-size:11px; font-weight:700; color:#5B606B; letter-spacing:0.05em; margin-bottom:6px;">PIPELINE</div>
            <div class="pipeline-row"><span>Embeddings</span><b>{pipeline.embedder.name}</b></div>
            <div class="pipeline-row"><span>Generator</span><b>{gen_label}</b></div>
            <div class="pipeline-row"><span>Chunks used (k)</span><b>{pipeline.top_k}</b></div>
        </div>""",
        unsafe_allow_html=True,
    )

# Main Area (Defects 1, 8, 9)
if len(pipeline.store) == 0:
    st.markdown('<h1 style="font-size:44px; margin-bottom:6px; line-height:1.15;">Ask your notes anything.</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:15px; color:#5B606B; margin-bottom:24px;">Upload your course PDFs or slides and BuBot answers from them only, with the exact source shown for every answer. If it isn\'t in your notes, it tells you.</p>', unsafe_allow_html=True)
    main_files = st.file_uploader("Upload files", type=["pdf", "pptx", "ppt", "txt", "md"], accept_multiple_files=True, key="main_uploader", label_visibility="collapsed")
    if main_files: index_files(main_files)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown('<div class="step-card"><span class="step-num">1</span><div class="step-title">Upload</div><div class="step-desc">Add notes, slides or textbook chapters as PDFs or PPTs.</div></div>', unsafe_allow_html=True)
    with col2: st.markdown('<div class="step-card"><span class="step-num">2</span><div class="step-title">Index</div><div class="step-desc">BuBot splits the text and builds a searchable index.</div></div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="step-card"><span class="step-num">3</span><div class="step-title">Ask</div><div class="step-desc">Get answers grounded in your own material.</div></div>', unsafe_allow_html=True)

    # Defect 8: Disabled chat input
    st.chat_input("Upload a document to start asking", disabled=True)

else:
    st.markdown(
        '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">'
        '<h2 style="margin:0; font-size:1.5rem;">Course Notes Q&A</h2>'
        '<span class="bubot-pill">Runs locally on your laptop</span></div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
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
