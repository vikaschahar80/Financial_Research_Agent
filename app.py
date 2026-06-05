"""
Financial Research Agent — Streamlit UI
Stack: Groq (Llama 3) + HuggingFace Embeddings (local) + ChromaDB
"""

import os
import tempfile
import streamlit as st
from pathlib import Path

from src.rag_pipeline import FinancialRAGPipeline
from src.agent import FinancialResearchAgent
from src.summarizer import FinancialSummarizer

st.set_page_config(page_title="Financial Research Agent", page_icon="📈", layout="wide")

# ── Session state ──────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    groq_key = st.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama3-8b-8192"], index=0)

    st.divider()
    st.subheader("📂 Upload Document")

    company_name = st.text_input("Company Name", placeholder="e.g., Reliance Industries")
    doc_type = st.selectbox("Document Type", ["Earnings Call Transcript", "10-K", "10-Q", "Annual Report", "RBI Report", "Other"])
    doc_year = st.text_input("Year", placeholder="e.g., 2024")

    uploaded_files = st.file_uploader("Upload PDF or TXT", type=["pdf", "PDF", "txt"], accept_multiple_files=True)

    if st.button("🔄 Ingest Documents", type="primary", disabled=not uploaded_files):
        if not groq_key:
            st.error("Please enter your Groq API key first.")
        else:
            with st.spinner("Loading embedding model and ingesting documents..."):
                rag = FinancialRAGPipeline()
                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    if suffix.lower() == ".pdf":
                        docs = rag.load_pdf(tmp_path)
                    else:
                        docs = rag.load_text(tmp_path)

                    docs = rag.add_metadata(docs, doc_type=doc_type, company=company_name, year=doc_year)
                    rag.ingest(docs)

                st.session_state.rag = rag
                st.session_state.agent = FinancialResearchAgent(rag=rag, model=model)
                st.session_state.docs_loaded = True

            st.success(f"✅ {len(uploaded_files)} file(s) ingested!")

    st.divider()
    st.caption("🔵 Chat: Groq Llama 3  |  ⚙️ Embeddings: local (free)")

# ── Main UI ────────────────────────────────────
st.title("📈 Financial Research Agent")
st.caption("Upload SEC filings, earnings transcripts, or RBI reports — then ask anything.")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Full Report", "🔍 Raw Retrieval"])

with tab1:
    if not st.session_state.docs_loaded:
        st.info("👈 Upload and ingest a document using the sidebar to get started.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the financials..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    result = st.session_state.agent.run(prompt)
                    answer = result["answer"]
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    if not st.session_state.docs_loaded:
        st.info("👈 Upload documents first.")
    else:
        st.subheader("🏦 Generate Full Investment Report")
        report_company = st.text_input("Company name", value=company_name)
        if st.button("🚀 Generate Report", type="primary"):
            with st.spinner("Generating report... (~30 seconds)"):
                summarizer = FinancialSummarizer(model=model)
                docs = st.session_state.rag.retrieve(f"{report_company} overview financials risks", k=10)
                context = "\n\n".join([d.page_content for d in docs])
                memo = summarizer.generate_investment_memo(company=report_company, context=context)
                markdown_report = summarizer.format_report_markdown(memo)
            st.markdown(markdown_report)
            st.download_button("⬇️ Download Report", data=markdown_report,
                               file_name=f"{report_company}_memo.md", mime="text/markdown")

with tab3:
    if not st.session_state.docs_loaded:
        st.info("👈 Upload documents first.")
    else:
        st.subheader("🔍 Inspect Retrieved Chunks")
        raw_query = st.text_input("Query to inspect")
        k = st.slider("Chunks to retrieve", 2, 10, 4)
        if st.button("Retrieve") and raw_query:
            docs = st.session_state.rag.retrieve(raw_query, k=k)
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Chunk {i} — {doc.metadata}"):
                    st.text(doc.page_content)
