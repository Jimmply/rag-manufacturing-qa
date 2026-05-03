"""
Retrieval layer for the Manufacturing Q&A RAG system.

Queries the ChromaDB vector store and returns relevant document chunks
along with source metadata for citation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingestion import COLLECTION_NAME, CHROMA_PERSIST_DIR, get_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))


class ManufacturingRetriever:
    """
    Thin wrapper around a ChromaDB collection that provides semantic search
    and formatted context for downstream LLM generation.
    """

    def __init__(self, persist_dir: str | None = None, top_k: int = TOP_K) -> None:
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.top_k = top_k
        self._vectorstore: Chroma | None = None

    def _get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            persist_path = Path(self.persist_dir)
            if not persist_path.exists():
                raise RuntimeError(
                    f"ChromaDB directory not found at '{persist_path}'. "
                    "Run ingestion first: python src/ingestion.py"
                )
            self._vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=str(persist_path),
            )
            count = self._vectorstore._collection.count()
            if count == 0:
                raise RuntimeError(
                    "Vector store is empty. Run ingestion first: python src/ingestion.py"
                )
            logger.info("Connected to ChromaDB: %d chunks available", count)
        return self._vectorstore

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Search the vector store for the most relevant chunks.

        Returns a list of (Document, relevance_score) tuples sorted by score
        descending (higher = more similar for cosine similarity).
        """
        vectorstore = self._get_vectorstore()
        results = vectorstore.similarity_search_with_relevance_scores(
            query, k=self.top_k
        )
        logger.debug("Retrieved %d chunks for query: %r", len(results), query[:80])
        return results

    def format_context(self, results: List[Tuple[Document, float]]) -> str:
        """
        Format retrieved chunks into a structured context block for the LLM prompt.
        """
        if not results:
            return "No relevant documents found."

        sections: List[str] = []
        for i, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_str = f", page {page + 1}" if page != "" else ""
            header = f"[Source {i}: {Path(source).name}{page_str} | Relevance: {score:.2f}]"
            sections.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(sections)

    def get_source_citations(self, results: List[Tuple[Document, float]]) -> List[dict]:
        """
        Return structured source metadata for display in the UI.
        """
        citations = []
        seen = set()
        for doc, score in results:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            key = (source, page)
            if key not in seen:
                seen.add(key)
                citations.append(
                    {
                        "file": Path(source).name,
                        "page": int(page) + 1 if page != "" else None,
                        "relevance": round(score, 3),
                        "snippet": doc.page_content[:200].strip() + "...",
                    }
                )
        return citations

    def is_ready(self) -> bool:
        """Return True if the vector store exists and contains documents."""
        try:
            vs = self._get_vectorstore()
            return vs._collection.count() > 0
        except RuntimeError:
            return False
