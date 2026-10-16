"""
DSA RAG Chatbot - Streamlit Application
An AI-powered Data Structures & Algorithms tutor grounded in curated PDF documents.
Built with Streamlit, LangChain, Google Gemini, Pinecone, and PyPDF.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, List

import config
from src.pdf_loader import list_available_pdfs
from src.embeddings import get_embedding_model
from src.vector_store import get_vector_store, get_index_stats
from src.retriever import get_dsa_retriever
from src.rag_chain import get_llm, execute_rag_query

# Streamlit Page Configuration
st.set_page_config(
    page_title="DSA AI Tutor - RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #854D0E;
    }
    .badge-danger {
        background-color: #FDE8E8;
        color: #9B1C1C;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.88rem;
    }
    .empty-state-card {
        border: 1px dashed #CBD5E1;
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        background-color: #F8FAFC;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    """
    Cache and initialize the LangChain RAG pipeline components.
    Returns (retriever, llm, status_dict).
    """
    missing_creds = config.get_missing_credentials()
    if missing_creds:
        return None, None, {
            "ready": False,
            "error": f"Missing credentials: {', '.join(missing_creds)}",
        }

    try:
        # Check Pinecone status
        stats = get_index_stats()
        if not stats.get("exists", False):
            return None, None, {
                "ready": False,
                "error": f"Pinecone index '{config.PINECONE_INDEX_NAME}' was not found. Please create and ingest it first.",
            }

        if stats.get("total_vector_count", 0) == 0:
            return None, None, {
                "ready": False,
                "error": f"Pinecone index '{config.PINECONE_INDEX_NAME}' contains 0 vectors. Please run ingestion to index your DSA PDFs.",
            }

        # Initialize Embeddings, VectorStore, Retriever, and LLM
        embedding_model = get_embedding_model()
        vector_store = get_vector_store(embedding_model=embedding_model)
        retriever = get_dsa_retriever(vector_store=vector_store, top_k=config.TOP_K_RESULTS)
        llm = get_llm()

        return retriever, llm, {
            "ready": True,
            "vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 768),
        }
    except Exception as exc:
        return None, None, {
            "ready": False,
            "error": str(exc),
        }


def render_sidebar():
    """Render the application sidebar with status, metrics, and topic suggestions."""
    with st.sidebar:
        st.header("⚙️ System Status")
        
        missing = config.get_missing_credentials()
        
        # Credentials Status
        if not missing:
            st.markdown('<span class="status-badge badge-success">✓ Credentials Configured</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-danger">✗ Incomplete Credentials</span>', unsafe_allow_html=True)
            st.caption(f"Missing in `.env`: {', '.join(missing)}")
            
        st.divider()
        
        # Knowledge Base Status
        st.subheader("📚 Knowledge Base")
        retriever, llm, pipeline_status = load_rag_pipeline()
        
        if pipeline_status.get("ready"):
            st.markdown('<span class="status-badge badge-success">✓ Knowledge Base Connected</span>', unsafe_allow_html=True)
            st.caption(f"**Index:** `{config.PINECONE_INDEX_NAME}`")
            st.caption(f"**Total Vectors:** {pipeline_status.get('vector_count', 0)}")
            st.caption(f"**Dimension:** {pipeline_status.get('dimension', 768)} (Cosine)")
        else:
            st.markdown('<span class="status-badge badge-warning">⚠ Knowledge Base Not Ready</span>', unsafe_allow_html=True)
            st.info(pipeline_status.get("error", "Not initialized"))
            
        # PDF Files Overview
        st.divider()
        st.subheader("📄 Available PDFs")
        available_pdfs = list_available_pdfs(config.PDF_DIR)
        if available_pdfs:
            for pdf in available_pdfs:
                st.text(f"• {pdf['name']} ({pdf['pages']} pgs, {pdf['size_kb']} KB)")
        else:
            st.caption("No PDFs found in `data/pdfs/`. Add your DSA PDF files to ingest them.")

        # Quick Starter Prompts
        st.divider()
        st.subheader("💡 Suggested Topics")
        topic_prompts = [
            "Explain the Two-Pointer technique with Python code",
            "What is the time complexity of QuickSort vs MergeSort?",
            "How does Breadth-First Search (BFS) work on graphs?",
            "Explain Dynamic Programming memoization vs tabulation",
            "How does a Binary Search Tree handle node deletion?",
            "Explain Kadane's algorithm for maximum subarray sum",
        ]
        
        for p in topic_prompts:
            if st.button(p, key=f"btn_{p}", use_container_width=True):
                st.session_state.pending_prompt = p

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_empty_state():
    """Render a helpful empty-state UI when no messages exist."""
    st.markdown("""
    <div class="empty-state-card">
        <h2 style="color: #1E293B; margin-bottom: 0.5rem;">Welcome to the DSA AI Tutor 🧠</h2>
        <p style="color: #64748B; max-width: 650px; margin: 0 auto 1.5rem auto;">
            I am a specialized Data Structures & Algorithms assistant grounded exclusively in your curated course PDFs.
            I explain concepts step-by-step, provide clean Python implementations, analyze Big-O complexities, and cite exact page numbers.
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 1rem;">
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem 1.2rem; text-align: left; width: 260px;">
                <strong>📚 Grounded Answers</strong>
                <p style="font-size: 0.85rem; color: #64748B; margin: 0.2rem 0 0 0;">Strictly derived from your indexed DSA PDFs.</p>
            </div>
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem 1.2rem; text-align: left; width: 260px;">
                <strong>⚡ Big-O Complexity</strong>
                <p style="font-size: 0.85rem; color: #64748B; margin: 0.2rem 0 0 0;">Time and space complexity breakdowns for every algorithm.</p>
            </div>
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem 1.2rem; text-align: left; width: 260px;">
                <strong>🔍 Verifiable Citations</strong>
                <p style="font-size: 0.85rem; color: #64748B; margin: 0.2rem 0 0 0;">Direct references to source files and page numbers.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    render_sidebar()

    st.markdown('<div class="main-header">DSA RAG Tutor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-powered Data Structures & Algorithms guidance powered by Google Gemini, LangChain & Pinecone</div>',
        unsafe_allow_html=True
    )

    retriever, llm, pipeline_status = load_rag_pipeline()

    # Informative Banner when knowledge base is not connected
    if not pipeline_status.get("ready"):
        st.warning(
            f"**⚠️ Knowledge Base Status: Not Connected**\n\n"
            f"{pipeline_status.get('error')}\n\n"
            "**Steps to activate your tutor:**\n"
            "1. Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME`.\n"
            "2. Add your DSA PDF documents to `data/pdfs/`.\n"
            "3. Run the ingestion script (`python -m src.ingest`) to create the 768-dimension index and store document vectors.\n"
            "4. Refresh this page to begin chatting!"
        )

    # Render Empty State or Message History
    if not st.session_state.messages:
        render_empty_state()
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Render source citations if available
                if msg.get("sources"):
                    with st.expander(f"📚 Source References ({len(msg['sources'])} cited)"):
                        for idx, source in enumerate(msg["sources"], 1):
                            st.markdown(
                                f"**[{idx}] {source.get('source')} — Page {source.get('page')}**"
                            )
                            if source.get("preview"):
                                st.caption(f'"{source.get("preview")}"')

    # Handle incoming user query
    user_query = st.chat_input("Ask a DSA question (e.g., 'How does Dijkstra's algorithm work?')...")
    
    # Check if a prompt was triggered via sidebar suggested buttons
    if st.session_state.pending_prompt:
        user_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_query:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response
        with st.chat_message("assistant"):
            if not pipeline_status.get("ready"):
                fallback_msg = (
                    "**Cannot answer query yet.** The knowledge base is not connected. "
                    "Please configure your `.env` credentials, place your DSA PDFs in `data/pdfs/`, "
                    "and run the ingestion pipeline."
                )
                st.markdown(fallback_msg)
                st.session_state.messages.append({"role": "assistant", "content": fallback_msg, "sources": []})
            else:
                with st.spinner("Searching DSA knowledge base and generating grounded answer..."):
                    try:
                        result = execute_rag_query(
                            question=user_query,
                            retriever=retriever,
                            llm=llm,
                        )
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"📚 Source References ({len(sources)} cited)"):
                                for idx, source in enumerate(sources, 1):
                                    st.markdown(
                                        f"**[{idx}] {source.get('source')} — Page {source.get('page')}**"
                                    )
                                    if source.get("preview"):
                                        st.caption(f'"{source.get("preview")}"')

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    except Exception as err:
                        error_msg = f"❌ **An error occurred while answering your question:**\n\n`{str(err)}`"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": [],
                        })


if __name__ == "__main__":
    main()
