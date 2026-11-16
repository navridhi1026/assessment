import os
from datetime import datetime

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
NO_ANSWER = "I couldn't find relevant information in the uploaded documents."

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#172033; --muted:#758099; --line:#e6eaf2; --accent:#5947dc; }
    html, body, [class*="css"] { font-family:Inter,Segoe UI,sans-serif; color:var(--ink); }
    .stApp { background:linear-gradient(180deg,#f8f9fd 0%,#f2f4fa 100%); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.6rem; }
    .brand { display:flex; align-items:center; gap:12px; margin:0 0 7px; }
    .brand-icon { display:grid; place-items:center; width:42px; height:42px; border-radius:14px;
      background:linear-gradient(140deg,#735ff1,#4333bd); color:#fff; font-size:22px;
      box-shadow:0 7px 18px #5947dc38; }
    .brand-name { font:800 16px Inter,Segoe UI,sans-serif; letter-spacing:-.4px; color:#20263a; }
    .brand-caption { margin:0 0 25px 54px; color:#8992a7; font-size:11px; }
    .eyebrow { color:#6656d8; font-size:11px; font-weight:700; letter-spacing:1.3px;
      text-transform:uppercase; margin-bottom:8px; }
    .hero-title { color:#20263a; font:800 clamp(28px,3vw,40px) Inter,Segoe UI,sans-serif;
      letter-spacing:-1.6px; line-height:1.16; margin:0 0 10px; }
    .hero-copy { color:#778198; font-size:15px; line-height:1.6; margin-bottom:24px; }
    .status-card { padding:13px 15px; border:1px solid var(--line); border-radius:12px;
      background:#fff; margin:12px 0 20px; color:#546078; font-size:12px; }
    .answer-card { background:#fff; border:1px solid var(--line); border-radius:16px;
      padding:20px 22px; box-shadow:0 8px 26px #2633540a; }
    .source-card { background:#f7f8fc; border:1px solid #e8ebf3; border-radius:10px;
      padding:12px 14px; margin:8px 0; color:#59647c; font-size:13px; line-height:1.55; }
    .source-label { font-weight:700; color:#414b63; margin-bottom:4px; }
    .section-label { color:#414b63; font-size:12px; font-weight:700;
      letter-spacing:.7px; text-transform:uppercase; margin:22px 0 8px; }
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button {
      border:0; border-radius:9px; color:#fff; background:linear-gradient(120deg,#6653e8,#4936c4);
      font-weight:700; min-height:42px; box-shadow:0 6px 15px #5947dc24;
    }
    div.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button:hover {
      border:0; color:#fff; background:#4936c4; }
    [data-testid="stFileUploader"] { border:1px dashed #cdd2e1; border-radius:11px;
      background:#fafaff; padding:8px; }
    div[data-testid="stChatMessage"] { border:1px solid #e9ecf3; background:#fff;
      border-radius:13px; padding:14px 17px; }
    textarea, input { border-radius:10px !important; }
    .footer-note { color:#9aa2b3; font-size:11px; text-align:center; margin-top:28px; }
    @media (max-width: 760px) { .hero-title { letter-spacing:-.8px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "api_error" not in st.session_state:
    st.session_state.api_error = ""


def api_request(path, method="get", **kwargs):
    response = requests.request(
        method,
        f"{API_URL}{path}",
        timeout=(5, 180),
        **kwargs,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API error ({response.status_code}): {detail}")
    if response.status_code == 204:
        return None
    return response.json()


def refresh_documents():
    try:
        st.session_state.documents = api_request("/documents")
        st.session_state.api_error = ""
    except (requests.RequestException, RuntimeError) as error:
        st.session_state.api_error = str(error)


refresh_documents()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">✦</div>'
        '<div class="brand-name">Knowledge<br>Assistant</div></div>'
        '<div class="brand-caption">PRIVATE DOCUMENT INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Add to your knowledge base</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a PDF here or browse",
        type=["pdf"],
        help="Searchable PDFs up to 20 MiB. Scanned PDFs require OCR.",
        label_visibility="collapsed",
    )
    if st.button(
        "Upload and index PDF",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        try:
            with st.spinner("Extracting text and creating semantic embeddings…"):
                result = api_request(
                    "/upload",
                    method="post",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                )
            st.toast(
                "This PDF is already indexed."
                if result.get("already_present")
                else f"Indexed {result['chunk_count']} passages."
            )
            refresh_documents()
        except (requests.RequestException, RuntimeError) as error:
            st.error(str(error))

    st.markdown('<div class="section-label">Your documents</div>', unsafe_allow_html=True)
    documents = st.session_state.documents
    if documents:
        st.caption(f"{len(documents)} document{'s' if len(documents) != 1 else ''} indexed")
        for document in documents:
            with st.container(border=True):
                st.markdown(f"**{document['filename']}**")
                st.caption(
                    f"{document['page_count']} pages · {document['chunk_count']} passages"
                )
                if st.button(
                    "Remove document",
                    key=f"delete-{document['id']}",
                    use_container_width=True,
                ):
                    try:
                        api_request(f"/documents/{document['id']}", method="delete")
                        st.session_state.messages = []
                        refresh_documents()
                        st.rerun()
                    except (requests.RequestException, RuntimeError) as error:
                        st.error(str(error))
    else:
        st.caption("Upload a PDF to start building your knowledge base.")

    if st.session_state.api_error:
        st.error(f"Backend unavailable: {st.session_state.api_error}")
    else:
        st.markdown(
            '<div class="status-card">● &nbsp; Backend connected &nbsp;·&nbsp; '
            'Local embeddings &amp; Ollama</div>',
            unsafe_allow_html=True,
        )

    st.caption("Your files and embeddings stay in your configured local storage.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

left, center, right = st.columns([1, 5, 1])
with center:
    st.markdown('<div class="eyebrow">Enterprise knowledge, at your fingertips</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Get answers from<br>your documents.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-copy">Ask a question in plain language. The assistant searches '
        'your PDFs and responds with page-level sources.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.documents:
        with st.container(border=True):
            st.markdown("### Start with your knowledge base")
            st.write(
                "Upload a text-based PDF from the sidebar. Its content is chunked, "
                "embedded locally, and stored in persistent ChromaDB."
            )
            st.caption(
                "No paid API key required · Sentence Transformers · Ollama qwen2.5:1.5b"
            )
    else:
        st.markdown(f"**{len(st.session_state.documents)} source documents ready**")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"View {len(message['sources'])} retrieved source passages"):
                    for source in message["sources"]:
                        with st.container(border=True):
                            st.caption(
                                f'{source["filename"]} · Page {source["page_number"]}'
                            )
                            st.write(source["excerpt"])

    with st.form("question-form", clear_on_submit=True):
        question = st.text_area(
            "Ask your documents",
            placeholder="For example: What is our policy on data retention?",
            height=82,
            label_visibility="collapsed",
        )
        ask = st.form_submit_button(
            "Ask the assistant  →",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.documents) or bool(st.session_state.api_error),
        )

    if ask and question.strip():
        st.session_state.messages.append({"role": "user", "content": question.strip()})
        try:
            with st.spinner("Searching your documents and preparing a cited answer…"):
                result = api_request("/ask", method="post", json={"question": question.strip()})
            answer = result["answer"]
            if answer.strip() == NO_ANSWER:
                answer = f"**I couldn't find this in the uploaded documents.**\n\n{answer}"
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": result["sources"]}
            )
        except (requests.RequestException, RuntimeError) as error:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**Request failed:** {error}", "sources": []}
            )
        st.rerun()

    st.markdown(
        f'<div class="footer-note">Local-first document assistant · '
        f'{datetime.now().year}</div>',
        unsafe_allow_html=True,
    )
