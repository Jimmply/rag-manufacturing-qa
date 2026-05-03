"""
Document ingestion pipeline for the Manufacturing Q&A RAG system.

Loads PDF and text files from a source directory, splits them into chunks,
generates embeddings, and stores them in a persistent ChromaDB collection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
COLLECTION_NAME = "manufacturing_docs"


def load_documents(source_dir: str | Path) -> List[Document]:
    """Load all PDF and TXT documents from a directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Document directory not found: {source_path}")

    documents: List[Document] = []

    pdf_loader = DirectoryLoader(
        str(source_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    txt_loader = DirectoryLoader(
        str(source_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents.extend(pdf_loader.load())
    documents.extend(txt_loader.load())

    logger.info("Loaded %d document pages/sections from %s", len(documents), source_path)
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a sentence-transformers embedding model (runs locally, no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_documents(source_dir: str | Path) -> int:
    """
    Full ingestion pipeline: load -> split -> embed -> store.

    Returns the number of chunks stored.
    """
    documents = load_documents(source_dir)
    if not documents:
        logger.warning("No documents found in %s", source_dir)
        return 0

    chunks = split_documents(documents)

    embeddings = get_embeddings()

    from langchain_chroma import Chroma

    persist_path = Path(CHROMA_PERSIST_DIR)
    persist_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )

    existing_count = vectorstore._collection.count()
    if existing_count > 0:
        logger.info(
            "Collection already contains %d chunks. Clearing before re-ingestion.",
            existing_count,
        )
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )

    vectorstore.add_documents(chunks)
    final_count = vectorstore._collection.count()
    logger.info("Stored %d chunks in ChromaDB at %s", final_count, persist_path)
    return final_count


def ingest_uploaded_file(file_path: str | Path) -> int:
    """
    Ingest a single uploaded file into the existing vector store.

    Returns the number of new chunks added.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Only PDF and TXT are supported.")

    documents = loader.load()
    chunks = split_documents(documents)

    embeddings = get_embeddings()

    from langchain_chroma import Chroma

    persist_path = Path(CHROMA_PERSIST_DIR)
    persist_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )

    vectorstore.add_documents(chunks)
    logger.info("Added %d chunks from %s to the vector store", len(chunks), file_path.name)
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    default_docs = Path(__file__).parent.parent / "data" / "sample_docs"
    count = ingest_documents(default_docs)
    print(f"Ingestion complete. {count} chunks stored.")
