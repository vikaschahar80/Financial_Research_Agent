"""
RAG Pipeline — uses local HuggingFace embeddings (free, no API key needed)
"""

import time
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class FinancialRAGPipeline:

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", " "],
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        return PyPDFLoader(file_path).load()

    def load_text(self, file_path: str) -> List[Document]:
        return TextLoader(file_path, encoding="utf-8").load()

    def load_from_url(self, url: str) -> List[Document]:
        return WebBaseLoader(url).load()

    def add_metadata(self, docs: List[Document], doc_type: str, company: str, year: str) -> List[Document]:
        for doc in docs:
            doc.metadata.update({"doc_type": doc_type, "company": company, "year": year})
        return docs

    def ingest(self, documents: List[Document], batch_size: int = 50) -> None:
        chunks = self.text_splitter.split_documents(documents)
        total = len(chunks)
        print(f"[RAG] Ingesting {total} chunks from {len(documents)} pages...")

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            print(f"[RAG] Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            else:
                self.vectorstore.add_documents(batch)

        if self.vectorstore:
            self.vectorstore.persist()
        print(f"[RAG] ✅ Done. Vector store saved to {self.persist_directory}")

    def load_existing(self) -> None:
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized.")
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self, k: int = 6):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
