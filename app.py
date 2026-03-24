# app.py - Frontend UI

# ── IMPORTS ────────────────────────────────────────────────────────────
import streamlit as st
from main import (
    get_embeddings,
    get_llm,
    build_vector_store,
    build_rag_chain,
    ask_question
)

# ── PAGE CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

# ── CACHED RESOURCES ────────────────────────────────────────────────────
# Runs once — not on every Streamlit rerun
@st.cache_resource
def load_embeddings():
    return get_embeddings()

@st.cache_resource
def load_llm():
    return get_llm()

# ── SESSION STATE ────────────────────────────────────────────────────────
# Persists data across Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ── UI ───────────────────────────────────────────────────────────────────
st.title("📄 RAG Chatbot")
st.caption("Upload any PDF and ask questions about it")

# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF..."):
            embeddings = load_embeddings()
            llm = load_llm()
            vector_store, num_pages, num_chunks = build_vector_store(
                uploaded_file,
                embeddings
            )
            st.session_state.rag_chain = build_rag_chain(vector_store, llm)
            st.session_state.pdf_processed = True
            st.session_state.messages = []

        st.success("✅ Ready!")
        st.info(f"📄 {num_pages} pages → {num_chunks} chunks")

    if st.session_state.pdf_processed:
        if st.button("Upload New PDF"):
            st.session_state.pdf_processed = False
            st.session_state.rag_chain = None
            st.session_state.messages = []
            st.rerun()

# ── CHAT AREA ────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF from the sidebar to get started")
else:
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(f"📄 Page {source['page']}: {source['text']}")

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Generate and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask_question(
                    st.session_state.rag_chain,
                    question
                )
            st.markdown(answer)
            with st.expander("Sources"):
                for source in sources:
                    st.caption(f"📄 Page {source['page']}: {source['text']}")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
