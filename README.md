# Rare Disease Diagnosis Assistant

An explainable, RAG-based AI assistant for rare disease diagnosis powered by **Anthropic Claude**. Retrieves real-time data from PubMed, Orphanet, and FDA, builds a knowledge graph, and generates source-cited answers with confidence scoring.

**Build4SC Hackathon 2026 | Track: Create (Intelligent Systems)**

## How It Works

```
User Query → Entity Extraction (Claude) → Data Collection (PubMed + Orphanet + FDA)
    → Knowledge Graph (NetworkX) → RAG Engine (FAISS + Claude) → Cited Answer + Confidence Score
```

## Claude Integration

| Use | Description |
|-----|-------------|
| Entity Extraction | Extracts diseases, drugs, symptoms from free text |
| RAG Generation | Generates evidence-based answers with source citations |
| Confidence Scoring | Self-assesses confidence (0-1), generates follow-ups if < 0.7 |
| Graph Unification | Merges entities from PubMed, Orphanet, FDA into unified graph |

## Tech Stack

Anthropic Claude · FastAPI · FAISS · Sentence Transformers · NetworkX · spaCy · Docker

## Quick Start

```bash
git clone https://github.com/namrathavpatil/Rare-Disease-Diagnosis-Assistant.git
cd Rare-Disease-Diagnosis-Assistant

pip install -r requirements.txt

# Add your ANTHROPIC_API_KEY to docker.env
export $(grep -v '^#' docker.env | grep '=' | xargs)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/rag-ready/simple-query` | POST | Ask a question (builds graph + answers) |
| `/search/diseases` | POST | Search PubMed + Orphanet |
| `/search/drugs` | POST | Search FDA drug data |
| `/extract-entities-llm` | POST | Claude entity extraction |
| `/rag-ready/build-graph` | POST | Build knowledge graph |
| `/knowledge-graph/stats` | GET | Graph statistics |
| `/health` | GET | Health check |

## Team

**Namratha Patil** — USC
