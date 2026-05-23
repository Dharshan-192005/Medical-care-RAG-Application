from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from ollama_client import OLLAMA_MODEL, generate_with_ollama, is_ollama_enabled
from rag import HealthcareRAG


APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "data" / "health_docs"
TOP_K = 3


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --chat-bg: #f7f7f8;
            --chat-panel: #ffffff;
            --chat-ink: #202123;
            --chat-muted: #6b7280;
            --chat-line: #e5e7eb;
            --chat-user: #eff6ff;
            --chat-user-line: #bfdbfe;
            --chat-accent: #10a37f;
        }

        .stApp {
            background: var(--chat-bg);
            color: var(--chat-ink);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--chat-line);
        }

        .block-container {
            max-width: 860px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        .chat-header {
            border-bottom: 1px solid var(--chat-line);
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            text-align: center;
        }

        .chat-header h1 {
            font-size: 1.85rem;
            line-height: 1.2;
            margin: 0;
            letter-spacing: 0;
        }

        .chat-header p {
            color: var(--chat-muted);
            font-size: .98rem;
            margin: .4rem auto 0;
            max-width: 620px;
        }

        .safety-strip {
            align-items: center;
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 8px;
            color: #713f12;
            display: flex;
            gap: .7rem;
            margin: 0 0 1.2rem;
            padding: .75rem .9rem;
        }

        .source-card {
            border: 1px solid var(--chat-line);
            border-radius: 8px;
            background: #ffffff;
            margin: .65rem 0;
            padding: .85rem .95rem;
        }

        .source-title {
            align-items: center;
            display: flex;
            justify-content: space-between;
            gap: .75rem;
            font-weight: 800;
            margin-bottom: .45rem;
        }

        .score-pill {
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            border-radius: 999px;
            color: #047857;
            font-size: .78rem;
            font-weight: 800;
            padding: .15rem .5rem;
            white-space: nowrap;
        }

        .source-text {
            color: #374151;
            font-size: .92rem;
            line-height: 1.55;
        }

        .sidebar-note {
            background: #f9fafb;
            border: 1px solid var(--chat-line);
            border-radius: 8px;
            color: #4b5563;
            padding: .8rem;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--chat-line);
            border-radius: 10px;
            background: var(--chat-panel);
            margin: .75rem 0;
            padding: .35rem;
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid var(--chat-line);
            background: rgba(247, 247, 248, .96);
        }

        .stButton button {
            border-radius: 8px;
        }

        .stButton button[kind="primary"],
        .stButton button:hover {
            border-color: var(--chat-accent);
            color: var(--chat-accent);
        }

        @media (max-width: 720px) {
            .block-container {
                padding-top: 1rem;
            }

            .safety-strip {
                align-items: flex-start;
                flex-direction: column;
                gap: .25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_source_cards(sources: list) -> None:
    for index, source in enumerate(sources, start=1):
        snippet = source.text[:650].strip()
        if len(source.text) > 650:
            snippet = f"{snippet}..."
        safe_source = escape(source.source)
        safe_snippet = escape(snippet)
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">
                    <span>{index}. {safe_source}</span>
                    <span class="score-pill">{source.score:.2f} match</span>
                </div>
                <div class="source-text">{safe_snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def initial_messages() -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": (
                "Hi, I can help with general healthcare education using the local "
                "knowledge base. Ask a focused question and I will show the sources I used."
            ),
            "sources": [],
        }
    ]


@st.cache_resource
def load_rag() -> HealthcareRAG:
    return HealthcareRAG(DOCS_DIR)


def answer_question(rag: HealthcareRAG, question: str) -> tuple[str, list]:
    retrieved = rag.retrieve(question, top_k=TOP_K)
    prompt = rag.build_prompt(question, retrieved)
    answer = generate_with_ollama(prompt)

    if not answer:
        answer = rag.retrieval_only_answer(question, retrieved)

    return answer, retrieved


st.set_page_config(page_title="CareGuide RAG", page_icon="+", layout="wide")
inject_styles()

docs = sorted(path.name for path in DOCS_DIR.glob("*.txt"))
ollama_ready = is_ollama_enabled()

st.markdown(
    """
    <section class="chat-header">
        <h1>Medical Care Assistant</h1>
        <p>
            Ask a health education question. Answers are generated from your local
            documents and include source snippets for checking.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="safety-strip">
        <strong>Safety note</strong>
        <span>This app is for learning only. For urgent symptoms, contact emergency services immediately.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Medical Assistant")
    st.markdown(
        """
        <div class="sidebar-note">
            Answers are limited to local text documents and should be checked
            against trusted clinical guidance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Knowledge Base")
    st.write(f"{len(docs)} documents indexed")
    for doc in docs:
        st.write(f"+ {doc}")

    st.subheader("Answer Mode")
    if ollama_ready:
        st.success(f"Ollama enabled: {OLLAMA_MODEL}")
    else:
        st.info("Retrieval-only fallback active")

    st.caption("Retrieval: TF-IDF")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = initial_messages()
        st.rerun()

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = initial_messages()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Review source snippets", expanded=False):
                render_source_cards(message["sources"])

question = st.chat_input("Ask a healthcare education question")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the local knowledge base..."):
            answer, sources = answer_question(rag, question)
        st.markdown(answer)
        if sources:
            with st.expander("Review source snippets", expanded=True):
                render_source_cards(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
