"""
Ingest manufacturing documents into ChromaDB vector store.

Usage
-----
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --docs-path data/my_docs/ --collection my_collection
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    p.add_argument("--docs-path",   type=str, default="data/sample_docs/")
    p.add_argument("--collection",  type=str, default="manufacturing_docs")
    p.add_argument("--chunk-size",  type=int, default=500)
    p.add_argument("--chunk-overlap", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    docs_path = Path(args.docs_path)

    if not docs_path.exists():
        logger.error("Documents path does not exist: %s", docs_path)
        sys.exit(1)

    files = list(docs_path.glob("*.txt")) + list(docs_path.glob("*.md")) + list(docs_path.glob("*.pdf"))
    logger.info("Found %d documents in %s", len(files), docs_path)

    try:
        from ingestion import DocumentIngester
        ingester = DocumentIngester(
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        ingester.ingest_directory(str(docs_path))
        logger.info("Ingestion complete. Collection: %s", args.collection)
    except ImportError:
        logger.warning("LangChain not installed. Install with: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
