"""
Unit tests for the document ingestion pipeline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the src/ directory is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a temporary .txt file with realistic content."""
    content = (
        "LASER CUTTER MAINTENANCE MANUAL\n\n"
        "Section 1: Daily Checks\n"
        "Check the nozzle for debris before each shift.\n"
        "Verify nitrogen pressure is at 20 bar.\n\n"
        "Section 2: Error Codes\n"
        "ERR-101: Laser source interlock open.\n"
        "ERR-102: Chiller temperature out of range.\n"
    )
    file = tmp_path / "test_manual.txt"
    file.write_text(content, encoding="utf-8")
    return file


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple sample text files."""
    (tmp_path / "manual_a.txt").write_text(
        "MANUAL A\nMaintenance section content here.\n" * 10,
        encoding="utf-8",
    )
    (tmp_path / "manual_b.txt").write_text(
        "MANUAL B\nOperations and safety procedures.\n" * 10,
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------

class TestLoadDocuments:
    def test_loads_txt_files(self, sample_docs_dir: Path):
        from ingestion import load_documents

        docs = load_documents(sample_docs_dir)
        assert len(docs) >= 2
        sources = [d.metadata.get("source", "") for d in docs]
        assert any("manual_a.txt" in s for s in sources)
        assert any("manual_b.txt" in s for s in sources)

    def test_raises_on_missing_directory(self):
        from ingestion import load_documents

        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/to/docs")

    def test_returns_empty_for_empty_directory(self, tmp_path: Path):
        from ingestion import load_documents

        docs = load_documents(tmp_path)
        assert docs == []


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------

class TestSplitDocuments:
    def test_chunks_are_produced(self, sample_docs_dir: Path):
        from ingestion import load_documents, split_documents

        docs = load_documents(sample_docs_dir)
        chunks = split_documents(docs)
        assert len(chunks) >= len(docs)

    def test_chunk_size_is_respected(self, sample_docs_dir: Path):
        from ingestion import load_documents, split_documents

        docs = load_documents(sample_docs_dir)
        chunks = split_documents(docs)
        for chunk in chunks:
            # Allow a small buffer above CHUNK_SIZE due to splitter behavior
            assert len(chunk.page_content) <= 1200, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )

    def test_metadata_is_preserved(self, sample_docs_dir: Path):
        from ingestion import load_documents, split_documents

        docs = load_documents(sample_docs_dir)
        chunks = split_documents(docs)
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_empty_input_returns_empty(self):
        from ingestion import split_documents

        assert split_documents([]) == []


# ---------------------------------------------------------------------------
# get_embeddings
# ---------------------------------------------------------------------------

class TestGetEmbeddings:
    @patch.dict(os.environ, {"EMBEDDING_MODEL": "all-MiniLM-L6-v2"})
    def test_returns_embedding_instance(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        from ingestion import get_embeddings

        embeddings = get_embeddings()
        assert isinstance(embeddings, HuggingFaceEmbeddings)

    @patch.dict(os.environ, {"EMBEDDING_MODEL": "all-MiniLM-L6-v2"})
    def test_embedding_model_name(self):
        from ingestion import get_embeddings

        embeddings = get_embeddings()
        assert "MiniLM" in embeddings.model_name


# ---------------------------------------------------------------------------
# ingest_uploaded_file
# ---------------------------------------------------------------------------

class TestIngestUploadedFile:
    def test_raises_on_unsupported_extension(self, tmp_path: Path):
        from ingestion import ingest_uploaded_file

        bad_file = tmp_path / "report.docx"
        bad_file.write_bytes(b"fake docx content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_uploaded_file(bad_file)

    @patch("ingestion.Chroma")
    @patch("ingestion.get_embeddings")
    def test_txt_file_is_ingested(
        self,
        mock_embeddings: MagicMock,
        mock_chroma_cls: MagicMock,
        sample_txt_file: Path,
        tmp_path: Path,
    ):
        from ingestion import ingest_uploaded_file

        mock_vs = MagicMock()
        mock_chroma_cls.return_value = mock_vs

        with patch.dict(os.environ, {"CHROMA_PERSIST_DIR": str(tmp_path / "chroma")}):
            count = ingest_uploaded_file(sample_txt_file)

        assert count > 0
        mock_vs.add_documents.assert_called_once()
        added_docs = mock_vs.add_documents.call_args[0][0]
        assert len(added_docs) == count


# ---------------------------------------------------------------------------
# ingest_documents (integration-level, mocked ChromaDB)
# ---------------------------------------------------------------------------

class TestIngestDocuments:
    @patch("ingestion.Chroma")
    @patch("ingestion.get_embeddings")
    def test_returns_chunk_count(
        self,
        mock_embeddings: MagicMock,
        mock_chroma_cls: MagicMock,
        sample_docs_dir: Path,
        tmp_path: Path,
    ):
        from ingestion import ingest_documents

        mock_vs = MagicMock()
        mock_vs._collection.count.side_effect = [0, 5]
        mock_chroma_cls.return_value = mock_vs

        with patch.dict(os.environ, {"CHROMA_PERSIST_DIR": str(tmp_path / "chroma")}):
            count = ingest_documents(sample_docs_dir)

        assert count == 5

    @patch("ingestion.Chroma")
    @patch("ingestion.get_embeddings")
    def test_clears_existing_collection(
        self,
        mock_embeddings: MagicMock,
        mock_chroma_cls: MagicMock,
        sample_docs_dir: Path,
        tmp_path: Path,
    ):
        from ingestion import ingest_documents

        mock_vs = MagicMock()
        # First count() call returns > 0 to trigger delete, subsequent calls return 0 then 3
        mock_vs._collection.count.side_effect = [10, 0, 3]
        mock_chroma_cls.return_value = mock_vs

        with patch.dict(os.environ, {"CHROMA_PERSIST_DIR": str(tmp_path / "chroma")}):
            ingest_documents(sample_docs_dir)

        mock_vs.delete_collection.assert_called_once()

    def test_returns_zero_for_empty_directory(self, tmp_path: Path):
        from ingestion import ingest_documents

        count = ingest_documents(tmp_path)
        assert count == 0
