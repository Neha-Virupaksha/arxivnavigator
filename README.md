# 🔬 ArXivNavigator — Self-Healing RAG for AI Research Papers

A production-style RAG system that answers natural language questions about AI research by retrieving real ArXiv paper abstracts, generating grounded answers, **evaluating its own output quality**, and **autonomously retrying with smarter retrieval strategies** when quality falls below threshold — running entirely on free, local infrastructure.

## Why "self-healing"?

Standard RAG systems have no idea when they're wrong — they retrieve, generate, and return an answer with no quality check. ArXivNavigator adds two extra steps:

1. **Evaluate** — a second LLM (acting as a judge) scores every answer on **Faithfulness**, **Answer Relevancy**, and **Context Relevance** using [Ragas](https://github.com/explodinggradients/ragas).
2. **Heal** — if the score falls below threshold, the system automatically retries with a different retrieval strategy, re-evaluates, and keeps whichever attempt scored best. It never returns something worse than the original attempt.

## Results (real data, not projected)

Across 16 healed queries in local testing: average Ragas composite score improved from **0.33 → 0.51** after healing.

## Architecture
User Query
│
▼
Query Classifier (Technique / Comparison / Recent / Paper / Application)
│
▼
Retriever ──► Pinecone (top-k chunks from category namespace)
│
▼
Generator ──► Ollama (llama3.2:3b) generates grounded answer
│
▼
Evaluator ──► Ragas + Ollama judge (llama3.2:1b)
│ scores: Faithfulness (40%) | Answer Relevancy (40%) | Context Relevance (20%)
│
▼
Composite Score ≥ threshold?
│
├── YES → Return answer, log to Postgres
│
└── NO → Self-Healing Loop
tries up to 2 of 5 strategies, re-evaluates each,
keeps the best-scoring attempt seen


## The 5 Healing Strategies

| Strategy | What it does | Best for |
|---|---|---|
| **Query Expansion** | LLM rewrites the query with full technical terms and synonyms | Vague/abbreviated queries, faithfulness failures |
| **Date-Aware Retrieval** | Filters to papers published in the last ~12 months | "Latest research on X" style queries |
| **Category Filtering** | Detects the research domain and restricts to the matching namespace | Cross-category retrieval noise |
| **Abstract Reranking** | Retrieves a larger pool, reranks by title + abstract similarity separately | Papers with a relevant title but dense abstract |
| **Expand K with MMR Diversity** | Widens the candidate pool, then selects a diverse (non-redundant) subset | Comparison queries needing multiple perspectives |

The strategy tried first is chosen based on *why* the original answer failed (which metric failed, and whether the query has recency language) — not a fixed order.

## Tech Stack

- **Orchestration:** LangChain
- **LLM Inference:** Ollama (`llama3.2:3b` for generation, `llama3.2:1b` as judge) — fully local, zero API cost
- **Vector DB:** Pinecone (namespace per ArXiv category: cs.AI, cs.LG, cs.CL, cs.CV, cs.IR)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Evaluation:** Ragas (Faithfulness, Answer Relevancy, Context Relevance)
- **API:** FastAPI
- **Database:** PostgreSQL (query + healing-attempt logging)
- **Dashboard:** Streamlit + Plotly
- **Testing:** Pytest
- **Deployment:** Docker Compose (Postgres + API + Dashboard, one command)

## Dashboard

- Live query interface with real-time scoring
- RAG Health Gauge (composite score, color-coded)
- Healing trigger rate and per-category performance
- Strategy effectiveness heatmap (by query type)
- Healing Journey chart — per-attempt score progression with winning attempt marked
- Before/after healing impact comparison
- Filterable query log

## Getting Started

### Prerequisites
- Python 3.11
- [Ollama](https://ollama.com) installed and running locally, with `llama3.2:3b`, `llama3.2:1b`, and `nomic-embed-text` pulled
- Docker Desktop
- A free [Pinecone](https://www.pinecone.io) account and index (384 dimensions, cosine metric)

### Setup

```bash
git clone https://github.com/Neha-Virupaksha/arxivnavigator.git
cd arxivnavigator
cp .env.example .env   # fill in your real Pinecone credentials and Postgres password
```

### Run everything with one command

```bash
docker compose up -d --build
```

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

### Ingest papers (first-time setup)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_ingestion.py
```

### Run tests

```bash
pytest tests/ -v
```

## Known Limitations (documented honestly)

- **Judge model calibration:** the local `llama3.2:1b` judge trends conservative on Faithfulness — verified via repeated evaluation of identical, well-grounded inputs. The Faithfulness threshold was recalibrated from 0.6 → 0.4 to account for this. As a result, the healing trigger rate on this dataset (~90-100%) is well above the originally targeted 20-30% — this reflects judge strictness, not system malfunction.
- **Query classification** is keyword-based, not ML-based — fast and effective for the 5 defined categories, but not as robust as a trained classifier would be.
- **No automatic paper refresh scheduler** yet — ingestion is run manually via `run_ingestion.py`.

## Project Structure

arxivnavigator/
├── src/
│ ├── ingestion/ # ArXiv fetching, chunking, embedding
│ ├── retrieval/ # Pinecone retrieval, MMR, reranking
│ ├── generation/ # Ollama-based answer generation
│ ├── evaluation/ # Ragas scoring, healer, strategies, composite scoring
│ ├── classification/ # Query type classifier
│ ├── api/ # FastAPI app
│ └── utils/ # DB models and connection
├── dashboard/ # Streamlit dashboard
├── tests/ # Pytest suite
├── docker-compose.yml
├── Dockerfile
└── requirements.txt


## License

MIT
