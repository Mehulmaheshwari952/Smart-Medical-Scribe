# Smart Medical Scribe and Patient Dashboard

**A Human-in-the-Loop Hybrid RAG Framework for Clinical Documentation and Decision Support**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-brightgreen.svg)](#running-tests)
[![TRL 4](https://img.shields.io/badge/Readiness-TRL%204-informational.svg)](#overview)

---

## Overview

Clinicians spend a significant portion of every consultation manually documenting notes while simultaneously cross-checking fragmented Electronic Health Records (EHRs) for allergies, active medications, laboratory trends, and prior treatment failures. Standalone generative Large Language Models (LLMs) risk factual hallucinations, whereas rigid rule-based Clinical Decision Support Systems (CDSSs) ignore unstructured narrative notes and contribute to alert fatigue.

The **Smart Medical Scribe and Patient Dashboard** bridges this gap through a modular, **human-in-the-loop (HITL)** pipeline operating at **Technology Readiness Level (TRL) 4**:

1. **Incremental Audio Transcription:** Captures live 30-second consultation audio chunks and transcribes them via **Whisper Large V3** (Groq API), conditioning each segment on prior transcript context.
2. **Schema-Enforced SOAP Structuring:** Transforms raw consultation transcripts into validated **Subjective, Objective, Assessment, and Plan (SOAP)** JSON records using **Llama 3.1** (zero fine-tuning) with strict `"unknown"` abstention defaults for unmentioned fields.
3. **Hybrid Longitudinal EHR Retrieval:** Combines dense semantic vectors (`BAAI/bge-small-en-v1.5`) and sparse lexical search (`BM25`) via **Reciprocal Rank Fusion (RRF)**, followed by a zero-keyword **`SemanticAdaptiveReranker`**.
4. **Citation-Verified Clinical Decision Support:** Synthesizes ranked clinical suggestions and plain-language patient summaries grounded strictly in retrieved EHR chunks, deterministically filtering out any uncited claims.
5. **Clinician Review & Approval Gate:** Presents all outputs on a unified clinician dashboard where drafts remain editable until explicitly approved and locked into the longitudinal EHR store.

> **Base Paper Reference:** Extends the RAG-LLM clinical prescribing architecture introduced by L. Garza, A. Kotal, M. A. Grasso, and E. Umucu, *"Retrieval-Augmented Framework for LLM-Based Clinical Decision Support,"* arXiv:2510.01363v1, 2025.

---

## System Architecture

```text
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────────┐
│  Live Audio / File  │────▶│  Whisper Large V3    │────▶│  Llama 3.1 SOAP Structuring │
│  (30s Chunks)       │     │  (Context-Conditioned)│     │  (Pydantic JSON + Abstain)  │
└─────────────────────┘     └──────────────────────┘     └──────────────┬──────────────┘
                                                                        │ retrieval_query
                                                                        ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────────┐
│  Clinician Dashboard│◀────│  Citation Validator  │◀────│  Hybrid EHR Retrieval       │
│  (HITL Edit/Approve)│     │  & Patient Summary   │     │  Dense + BM25 ─▶ RRF        │
└──────────┬──────────┘     └──────────────────────┘     └──────────────┬──────────────┘
           │ On Explicit Approval                                       │
           ▼                                                            ▼
┌─────────────────────┐                                  ┌─────────────────────────────┐
│  SQLite + ChromaDB  │◀─────────────────────────────────│  SemanticAdaptiveReranker   │
│  Longitudinal Store │                                  │  (Temporal + Safety + Gate) │
└─────────────────────┘                                  └─────────────────────────────┘
```

### Zero-Keyword `SemanticAdaptiveReranker`

Rather than relying on static half-lives or brittle keyword dictionaries, our reranker (`app/eval/custom_reranker.py`) uses three prototype semantic anchor embeddings encoded in the shared `BAAI/bge-small-en-v1.5` space:

* **Continuous Semantic Temporal Weighting:** Measures query alignment against acute biomarker anchors ($\mathbf{a}_{\text{acute}}$) versus longitudinal history anchors ($\mathbf{a}_{\text{chronic}}$) to smoothly scale record decay half-lives ($H_{\text{adapt}} = H_0 \cdot 2^{-\delta}$).
* **Cross-Intent Clinical Safety Alignment:** Elevates allergy, contraindication, and adverse-event evidence proportionally when both the clinical query and EHR chunk align with safety-risk semantics ($\mathbf{a}_{\text{safety}}$).
* **Relative Distribution Gating:** Filters out low-signal distractor visits ($S_i < 0.45 \cdot \max_j S_j$) before LLM prompt construction to reduce context noise.

---

## Repository Structure

```text
Smart-Medical-Scribe/
├── app/
│   ├── config.py                  # Pydantic settings (.env loader, model IDs, decay defaults)
│   ├── database.py                # SQLAlchemy engine, SessionLocal, init_db(), get_db()
│   ├── models.py                  # ORM models: Patient, StructuredFact, EHRRecord, Consultation
│   ├── schemas.py                 # Strict Pydantic schemas: SOAPNote, SuggestionItem, RetrievedChunk
│   ├── main.py                    # FastAPI app, ServiceContainer DI, REST routes, static mount
│   ├── seed_demo.py               # Seeds demo patients, longitudinal histories, and cohort cases
│   ├── services/
│   │   ├── stt.py                 # Audio validation, Groq Whisper V3 STT, transcript stitching
│   │   ├── structuring.py         # Single-pass transcript -> SOAP JSON + retrieval_query
│   │   ├── ingestion.py           # Section-aware EHR chunking, SQL fact storage, Chroma upsert
│   │   ├── retrieval.py           # Hybrid Dense + BM25 search, RRF fusion, baseline reranker
│   │   ├── generation.py          # Grounded clinical suggestions, citation filter, patient summary
│   │   └── similar_cases.py       # De-identified cohort ingestion (PHI guard) & semantic search
│   ├── eval/
│   │   ├── custom_reranker.py     # Zero-keyword SemanticAdaptiveReranker
│   │   ├── synthetic_data.py      # 20 labeled MTSamples cases + 12 longitudinal EHR patients
│   │   └── runner.py              # CLI evaluation harness (Recall@k, Precision@k, MRR, F1, latency)
│   └── static/
│       └── index.html             # Clinician review & patient dashboard UI (HTML + Tailwind + JS)
├── tests/                         # 24 automated unit and integration tests
├── data/                          # SQLite DB, ChromaDB files, and benchmark JSON snapshots
└── requirements.txt               # Python dependencies
```

---

## Getting Started

### Prerequisites

* Python **3.11+**
* A **Groq API Key** (for Whisper Large V3 STT and Llama 3.1 inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Smart-Medical-Scribe.git
   cd Smart-Medical-Scribe
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   DATABASE_URL=sqlite:///./data/scribe.db
   CHROMA_PERSIST_DIR=./data/chroma
   EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
   ```

5. **Seed the demo database and vector store:**
   ```bash
   python -m app.seed_demo
   ```

6. **Launch the application:**
   ```bash
   uvicorn app.main:app --reload
   ```
   Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to access the dashboard (or **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** for interactive Swagger API documentation).

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check |
| `GET` | `/patients` | List all registered patients |
| `POST` | `/patients` | Register a new patient with optional allergies, medications, and baseline notes |
| `GET` | `/patients/{id}/history` | Retrieve structured facts (`allergies`, `medications`, `diagnoses`) and EHR notes |
| `POST` | `/consultations` | Create a new consultation session (`draft`) for a patient |
| `POST` | `/consultations/{id}/audio` | Upload an audio chunk (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`, `.flac`) for STT |
| `POST` | `/consultations/{id}/finalize` | Generate SOAP JSON, run Hybrid RAG + `SemanticAdaptiveReranker`, and produce cited suggestions |
| `PATCH` | `/consultations/{id}` | Edit draft SOAP fields or patient summary prior to approval |
| `POST` | `/consultations/{id}/approve` | Lock consultation as `approved` and index SOAP note into longitudinal EHR |
| `GET` | `/similar-cases/search` | Search de-identified reference cohort cases by clinical query |

---

## Evaluation & Benchmark Results

The evaluation harness (`app/eval/runner.py`) benchmarks retrieval quality across 12 longitudinal patient EHR profiles (containing timestamped labs, allergies, chronic conditions, and routine distractor notes) and 20 labeled MTSamples-style clinical cases.

### Run the Benchmark

```bash
# Run offline retrieval comparison (Baseline vs. SemanticAdaptiveReranker)
python -m app.eval.runner --mode retrieval --save-json data/eval_improved.json

# Run full benchmark including Groq LLM SOAP structuring F1
python -m app.eval.runner --mode all
```

### Retrieval Performance Comparison (`BAAI/bge-small-en-v1.5`)

| Evaluation Metric | Baseline (`baseline_time_aware_rerank`) | Improved (`SemanticAdaptiveReranker`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Recall@1** | `0.4861` | `0.4861` | `0.0000` |
| **Recall@3** | `1.0000` | `1.0000` | `0.0000` (Zero recall loss) |
| **Recall@5** | `1.0000` | `1.0000` | `0.0000` (Zero recall loss) |
| **Precision@3** | `0.7500` | **`0.9583`** | **`+20.83%`** |
| **Precision@5** | `0.6250` | **`0.9583`** | **`+33.33%`** |
| **MRR** | `1.0000` | `1.0000` | `0.0000` |
| **Rerank Latency** | `0.014 ms` | `8.767 ms` | `+8.753 ms` |
| **Total CPU Retrieval** | `9.289 ms` | `18.102 ms` | `+8.813 ms` |

---

## Running Tests

The repository includes 24 automated unit and integration tests covering database schemas, audio validation, SOAP abstention rules, RRF retrieval, citation enforcement, approval immutability, PHI guards, and semantic adaptive reranking:

```bash
pytest -v
```

---

## Team & Contributors

**Department of Artificial Intelligence & Data Engineering (AIDE)**  
**Malaviya National Institute of Technology (MNIT), Jaipur**

* **Mehul Maheshwari** (`2023UAI1828` • `2023uai1828@mnit.ac.in`)  
  *Backend architecture, Groq Whisper V3 audio chunking pipeline, zero-fine-tuning SOAP JSON structuring, hybrid RRF retrieval, and `SemanticAdaptiveReranker`.*
* **Parth Agrawal** (`2023UAI1892` • `2023uai1892@mnit.ac.in`)  
  *Clinician review & patient dashboard UI, human-in-the-loop approval & indexing workflow, privacy-guarded cohort search, automated test suite, and evaluation harness.*

---

## License

This project is developed for academic and research demonstration purposes at MNIT Jaipur. Not intended for unvalidated clinical deployment.