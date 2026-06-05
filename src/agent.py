"""
Financial Research Agent — Groq (Llama 3) + local embeddings
"""

from typing import Any, Dict, List
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from src.rag_pipeline import FinancialRAGPipeline
from src.summarizer import FinancialSummarizer


class FinancialResearchAgent:

    SYSTEM_PROMPT = """You are an expert financial research analyst. You have been given 
excerpts from financial documents (SEC filings, earnings call transcripts, RBI reports, annual reports).

Answer the user's question using ONLY the provided document context.
- Be precise and use specific numbers/metrics when available
- Highlight key financial indicators (revenue, margins, EPS, guidance)
- Identify risks and opportunities explicitly
- If the context doesn't contain enough information, say so clearly
- Never fabricate financial figures

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Provide a thorough, structured answer based on the above context."""

    def __init__(self, rag: FinancialRAGPipeline, model: str = "llama-3.3-70b-versatile"):
        self.rag = rag
        self.summarizer = FinancialSummarizer(model=model)
        self.llm = ChatGroq(model=model, temperature=0.1)
        self.prompt = ChatPromptTemplate.from_template(self.SYSTEM_PROMPT)

    def run(self, query: str) -> Dict[str, Any]:
        docs: List[Document] = self.rag.retrieve(query, k=6)
        if not docs:
            return {"query": query, "answer": "No relevant information found in the uploaded documents."}

        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source = f"{meta.get('company', 'Unknown')} | {meta.get('doc_type', 'doc')} | {meta.get('year', '')}"
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        messages = self.prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)
        return {"query": query, "answer": response.content}

    def generate_full_report(self, company: str) -> Dict[str, str]:
        sections = {
            "Executive Summary": f"Provide a 3-sentence executive summary of {company}'s current financial position and outlook.",
            "Revenue & Profitability": f"Analyze {company}'s revenue growth, gross margins, and operating income trends.",
            "Key Risks": f"What are the top 3-5 risks facing {company} based on their filings?",
            "Management Guidance": f"What forward-looking guidance has {company}'s management provided?",
            "Investment Thesis": f"What is the bull and bear case for {company}?",
        }
        report = {}
        for section, query in sections.items():
            result = self.run(query)
            report[section] = result["answer"]
        return report
