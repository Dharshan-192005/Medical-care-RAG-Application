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
            --care-ink: #102027;
            --care-muted: #64757d;
            --care-line: #d7e2e5;
            --care-soft: #eef7f5;
            --care-teal: #0f766e;
            --care-coral: #d85f45;
            --care-gold: #b7791f;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, .10), transparent 28rem),
                linear-gradient(180deg, #f9fcfc 0%, #f3f7f8 100%);
            color: var(--care-ink);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--care-line);
        }

        .block-container {
            max-width: 1040px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        .care-hero {
            border: 1px solid var(--care-line);
            border-left: 6px solid var(--care-teal);
            border-radius: 8px;
            background: #ffffff;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1rem;
        }

        .care-kicker {
            color: var(--care-teal);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }

        .care-hero h1 {
            font-size: clamp(2rem, 4vw, 3.3rem);
            line-height: 1.04;
            margin: 0 0 .55rem 0;
            letter-spacing: 0;
        }

        .care-hero p {
            color: var(--care-muted);
            font-size: 1.02rem;
            margin: 0;
            max-width: 760px;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
            margin: .75rem 0 1.1rem;
        }

        .care-metric {
            border: 1px solid var(--care-line);
            border-radius: 8px;
            background: #ffffff;
            padding: .85rem 1rem;
        }

        .care-metric span {
            color: var(--care-muted);
            display: block;
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .care-metric strong {
            display: block;
            font-size: 1.35rem;
            margin-top: .2rem;
        }

        .safety-strip {
            align-items: center;
            background: #fff8ef;
            border: 1px solid #f2d3a5;
            border-radius: 8px;
            color: #5b3a08;
            display: flex;
            gap: .7rem;
            margin: .3rem 0 1.2rem;
            padding: .75rem .9rem;
        }

        .source-card {
            border: 1px solid var(--care-line);
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
            background: var(--care-soft);
            border: 1px solid #b9d9d5;
            border-radius: 999px;
            color: var(--care-teal);
            font-size: .78rem;
            font-weight: 800;
            padding: .15rem .5rem;
            white-space: nowrap;
        }

        .source-text {
            color: #405158;
            font-size: .92rem;
            line-height: 1.55;
        }

        .sidebar-note {
            background: #f6faf9;
            border: 1px solid var(--care-line);
            border-radius: 8px;
            padding: .8rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 8px;
            border: 1px solid rgba(215, 226, 229, .8);
            background: rgba(255, 255, 255, .72);
        }

        @media (max-width: 720px) {
            .metric-row {
                grid-template-columns: 1fr;
            }

            .block-container {
                padding-top: 1rem;
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
    <section class="care-hero">
        <div class="care-kicker">Local document assistant</div>
        <h1>CareGuide RAG</h1>
        <p>
            Ask health education questions and get grounded answers from the
            project knowledge base, with source snippets kept visible for review.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-row">
        <div class="care-metric"><span>Documents</span><strong>{len(docs)}</strong></div>
        <div class="care-metric"><span>Retrieval</span><strong>TF-IDF</strong></div>
        <div class="care-metric"><span>Generation</span><strong>{"Ollama" if ollama_ready else "Retrieval"}</strong></div>
    </div>
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
    st.header("CareGuide")
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
