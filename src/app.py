"""
Streamlit application for the Manufacturing Document Q&A RAG system.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Ensure src/ is on sys.path when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import ingest_documents, ingest_uploaded_file
from qa_chain import ManufacturingQAChain
from retriever import ManufacturingRetriever

load_dotenv()
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Manufacturing Doc Q&A",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain: ManufacturingQAChain | None = None

if "ingested" not in st.session_state:
    st.session_state.ingested: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_qa_chain(provider: str, model: str) -> ManufacturingQAChain:
    chain = st.session_state.qa_chain
    if chain is None:
        retriever = ManufacturingRetriever()
        chain = ManufacturingQAChain(retriever=retriever)
        st.session_state.qa_chain = chain

    chain.change_provider(provider, model)
    return chain


def check_vector_store_ready() -> bool:
    retriever = ManufacturingRetriever()
    return retriever.is_ready()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏭 Manufacturing Q&A")
    st.caption("RAG-powered document assistant")

    st.divider()

    # Model selection
    st.subheader("Model Settings")
    provider = st.selectbox(
        "Provider",
        options=["anthropic", "openai"],
        index=0,
        help="Select the LLM backend. Ensure the corresponding API key is set.",
    )

    model_options = {
        "anthropic": ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-3-5"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    }
    selected_model = st.selectbox(
        "Model",
        options=model_options[provider],
        index=0,
    )

    st.divider()

    # Document management
    st.subheader("Document Management")

    vs_ready = check_vector_store_ready()
    if vs_ready:
        st.success("Vector store ready", icon="✅")
    else:
        st.warning("Vector store empty. Load documents below.", icon="⚠️")

    # Load sample documents
    sample_docs_path = Path(__file__).parent.parent / "data" / "sample_docs"
    if st.button("Load Sample Documents", use_container_width=True, type="primary"):
        with st.spinner("Ingesting sample documents..."):
            try:
                count = ingest_documents(sample_docs_path)
                st.session_state.ingested = True
                st.session_state.qa_chain = None
                st.success(f"Loaded {count} chunks from sample docs!")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    # File uploader
    st.caption("Or upload your own documents:")
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Ingest Uploaded Files", use_container_width=True):
            total_chunks = 0
            with st.spinner(f"Ingesting {len(uploaded_files)} file(s)..."):
                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        chunks = ingest_uploaded_file(tmp_path)
                        total_chunks += chunks
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

            st.session_state.qa_chain = None
            st.success(f"Added {total_chunks} chunks from {len(uploaded_files)} file(s).")
            st.rerun()

    st.divider()

    # Clear conversation
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.lc_history = []
        st.rerun()

    st.divider()
    st.caption(
        "Built with LangChain · ChromaDB · Streamlit\n\n"
        "Source: [GitHub](https://github.com/)"
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Manufacturing Document Q&A")
st.caption(
    "Ask questions about maintenance procedures, error codes, cutting parameters, "
    "and safety requirements from your uploaded documents."
)

# Example queries
with st.expander("Example queries", expanded=not bool(st.session_state.chat_history)):
    example_queries = [
        "What is the daily maintenance procedure for the cutting head nozzle?",
        "What does error code ERR-201 mean and how do I fix it?",
        "What laser power and speed should I use for 5mm stainless steel?",
        "How often should the chiller coolant be replaced?",
        "What PPE is required when working on the laser system?",
        "How do I perform a nozzle alignment calibration?",
    ]
    cols = st.columns(2)
    for i, query in enumerate(example_queries):
        if cols[i % 2].button(query, key=f"example_{i}", use_container_width=True):
            st.session_state._prefill_query = query

# Chat display
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    with st.expander(f"Sources ({len(sources)} document section(s))"):
                        for src in sources:
                            page_info = f" — page {src['page']}" if src.get("page") else ""
                            st.markdown(
                                f"**{src['file']}**{page_info} "
                                f"*(relevance: {src['relevance']:.2f})*"
                            )
                            st.caption(src["snippet"])

# Chat input
prefill = st.session_state.pop("_prefill_query", "")
user_input = st.chat_input(
    "Ask a question about your manufacturing documents...",
    disabled=not check_vector_store_ready(),
)

if not check_vector_store_ready() and not user_input:
    st.info(
        "No documents loaded yet. Use the sidebar to load the sample documents "
        "or upload your own PDFs/TXT files.",
        icon="ℹ️",
    )

query = prefill or user_input
if query:
    if not check_vector_store_ready():
        st.error("Please load documents before asking questions.")
        st.stop()

    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()

        try:
            chain = get_qa_chain(provider, selected_model)
            full_response = ""
            final_metadata: dict = {}

            with st.spinner("Retrieving relevant sections..."):
                stream = chain.stream_answer(
                    query, chat_history=st.session_state.lc_history
                )

            for item in stream:
                if isinstance(item, dict):
                    final_metadata = item
                else:
                    full_response += item
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            sources = final_metadata.get("sources", [])
            if sources:
                with sources_placeholder.expander(
                    f"Sources ({len(sources)} document section(s))"
                ):
                    for src in sources:
                        page_info = f" — page {src['page']}" if src.get("page") else ""
                        st.markdown(
                            f"**{src['file']}**{page_info} "
                            f"*(relevance: {src['relevance']:.2f})*"
                        )
                        st.caption(src["snippet"])

        except EnvironmentError as e:
            full_response = f"Configuration error: {e}\n\nPlease check your `.env` file."
            sources = []
            message_placeholder.error(full_response)
        except RuntimeError as e:
            full_response = str(e)
            sources = []
            message_placeholder.warning(full_response)
        except Exception as e:
            full_response = f"An unexpected error occurred: {e}"
            sources = []
            message_placeholder.error(full_response)

    # Update history
    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )
    st.session_state.lc_history.extend(
        [HumanMessage(content=query), AIMessage(content=full_response)]
    )
    # Keep conversation history bounded
    st.session_state.lc_history = st.session_state.lc_history[-20:]
