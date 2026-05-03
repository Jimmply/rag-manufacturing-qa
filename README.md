# Manufacturing Document Q&A — RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-quality Retrieval-Augmented Generation (RAG) system that lets engineers and operators query manufacturing documents — maintenance manuals, operational procedures, and equipment guides — using plain natural language.

Built with LangChain, ChromaDB, sentence-transformers, and a Streamlit chat interface. Supports both Anthropic Claude and OpenAI GPT-4o as the generation backend with a single config change.

---

## Demo

![Demo GIF placeholder](assets/demo.gif)

> *Upload a maintenance manual, ask "What does error ERR-201 mean and how do I fix it?" — get a grounded answer with source citations in seconds.*

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                      │
│   ┌──────────────────┐          ┌───────────────────────────────┐  │
│   │  Sidebar         │          │  Chat Interface               │  │
│   │  • Doc upload    │          │  • Conversation history       │  │
│   │  • Model select  │          │  • Streaming response         │  │
│   │  • Load samples  │          │  • Source citations           │  │
│   └──────────────────┘          └───────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                     ManufacturingQAChain (qa_chain.py)             │
│                                                                    │
│   1. Receive user question + conversation history                  │
│   2. Retrieve top-k relevant document chunks                       │
│   3. Build prompt: System (manufacturing expert) + Context + Query │
│   4. Stream response from configured LLM                           │
└────────────────────────────────────────────────────────────────────┘
         │  retrieve()                         │  invoke/stream
         ▼                                     ▼
┌─────────────────────────┐      ┌─────────────────────────────────┐
│  ManufacturingRetriever │      │  LLM Backend (qa_chain.py)      │
│      (retriever.py)     │      │                                 │
│                         │      │  ┌──────────────────────────┐   │
│  • similarity_search    │      │  │ Anthropic claude-sonnet  │   │
│  • relevance scoring    │      │  │ OpenAI gpt-4o            │   │
│  • source formatting    │      │  └──────────────────────────┘   │
└────────────┬────────────┘      │  Selected via MODEL_PROVIDER env │
             │                   └─────────────────────────────────┘
             ▼
┌─────────────────────────┐
│  ChromaDB Vector Store  │
│  (Persistent on disk)   │
│                         │
│  Collection:            │
│  manufacturing_docs     │
└────────────┬────────────┘
             │  populated by
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Ingestion Pipeline (ingestion.py)              │
│                                                                 │
│  PDF / TXT files  ──►  RecursiveCharacterTextSplitter           │
│                    ──►  HuggingFace sentence-transformers        │
│                    ──►  ChromaDB (persisted to ./chroma_db/)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain 0.3 |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost) |
| Generation | Anthropic `claude-sonnet-4-6` / OpenAI `gpt-4o` |
| UI | Streamlit |
| Document loaders | LangChain PyPDFLoader, TextLoader |
| Text splitting | RecursiveCharacterTextSplitter |
| Testing | pytest + pytest-mock |

---

## Project Structure

```
rag-manufacturing-qa/
├── src/
│   ├── app.py           # Streamlit chat application
│   ├── ingestion.py     # Document loading, chunking, and embedding pipeline
│   ├── retriever.py     # ChromaDB semantic search and context formatting
│   └── qa_chain.py      # LangChain RAG chain with multi-provider LLM support
├── data/
│   └── sample_docs/
│       └── sample_manual.txt   # Laseronics LX-5000 maintenance manual (demo)
├── tests/
│   └── test_ingestion.py       # Unit tests for the ingestion pipeline
├── .env.example         # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/rag-manufacturing-qa.git
cd rag-manufacturing-qa

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your API key for the provider you want to use:

```env
MODEL_PROVIDER=anthropic          # or "openai"
ANTHROPIC_API_KEY=sk-ant-...      # required if MODEL_PROVIDER=anthropic
OPENAI_API_KEY=sk-...             # required if MODEL_PROVIDER=openai
```

### 3. Run the Streamlit app

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Load documents and start querying

1. Click **"Load Sample Documents"** in the sidebar to ingest the included Laseronics LX-5000 maintenance manual.
2. Ask a question in the chat box.
3. To use your own documents, upload PDFs or TXT files via the sidebar uploader.

---

## Configuration Reference

All settings are controlled via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `MODEL_PROVIDER` | `anthropic` | LLM backend: `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic model name |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Path to ChromaDB storage |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers model |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve per query |
| `CHUNK_SIZE` | `1000` | Max characters per document chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |

---

## Example Queries

The included sample manual covers a laser cutting machine. Try these:

| Query | What it tests |
|---|---|
| *"What is the daily maintenance procedure for the cutting head nozzle?"* | Procedural retrieval |
| *"What does error ERR-201 mean and how do I fix it?"* | Troubleshooting lookup |
| *"What laser power and cutting speed should I use for 5mm stainless steel with nitrogen?"* | Parameter table retrieval |
| *"How often should the chiller coolant be replaced and what is the spec?"* | Scheduled maintenance |
| *"What PPE is required when working on the laser during maintenance?"* | Safety information |
| *"How do I perform a nozzle alignment calibration step by step?"* | Multi-step procedure |

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers ingestion pipeline unit tests with mocked ChromaDB calls (no API keys or GPU required).

---

## Ingestion from the Command Line

To re-ingest documents without launching the UI:

```bash
python src/ingestion.py
```

This processes all `.pdf` and `.txt` files in `data/sample_docs/` by default. The vector store is cleared and rebuilt on each run.

To ingest a custom directory:

```python
from src.ingestion import ingest_documents
ingest_documents("/path/to/your/manuals/")
```

---

## Design Decisions

**Local embeddings** — `sentence-transformers` runs on CPU with no API cost. For production at scale, swap to OpenAI `text-embedding-3-small` via `langchain-openai` for higher throughput.

**ChromaDB** — Zero-config local persistence is ideal for a portfolio demo and single-machine deployment. Replace with Pinecone, Weaviate, or pgvector for multi-user / cloud deployments.

**Provider abstraction** — The `ManufacturingQAChain.change_provider()` method lets the UI hot-swap between Anthropic and OpenAI without rebuilding the retriever or vector store.

**Conversation history** — The chain passes the last 6 messages as context, giving the LLM enough conversational grounding without inflating token costs on long sessions.

---

## License

MIT — see [LICENSE](LICENSE).
