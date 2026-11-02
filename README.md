---
title: Enterprise RAG Platform
emoji: 🤗
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Enterprise RAG Platform

> Production-grade Retrieval-Augmented Generation system with hybrid retrieval, intelligent reranking, conversation memory, and RAGAS evaluation.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit-blue?style=flat)](https://sohel2309-enterprise-rag.hf.space)
[![GitHub](https://img.shields.io/badge/Code-Repository-black?style=flat)](https://github.com/Sohel2309/Enterprise-RAG)

## Overview

Enterprise RAG Platform combines **BM25 sparse retrieval + BGE-large dense embeddings** with **Cohere cross-encoder reranking** to deliver accurate, context-aware answers from uploaded documents.

The platform supports multi-turn conversation memory, real-time PDF/DOCX ingestion, hybrid retrieval, and automated RAGAS evaluation.

## 📊 Key Results

| Metric | Score |
|---|---:|
| Faithfulness | **0.842** |
| Answer Relevancy | **0.900** ⭐ |
| Context Precision | **0.877** |
| Context Recall | **0.721** |
| Indexed Chunks | **7,536** |
| Evaluated QA Pairs | **45** |

## 🏗️ Architecture

```text
Documents (PDF/DOCX)
        ↓
Parsing + Chunking
        ↓
BGE-large Embeddings (Redis-cached)
        ↓
        ├── FAISS (dense search)
        └── BM25 (sparse search)
                ↓
        Reciprocal Rank Fusion (RRF)
                ↓
        Cohere Cross-Encoder Reranking
                ↓
        LLM Generation (Groq gpt-oss-120b)
        + Conversation Memory
                ↓
            Answer + Sources
```

### Retrieval Pipeline

- **BM25** sparse retrieval
- **BGE-large** dense embeddings
- **Reciprocal Rank Fusion (RRF)** to combine both
- **Cohere cross-encoder reranking** on top candidates
- Top retrieved results are passed to the generation pipeline

Each stage is its own module behind a small interface (`BaseRetriever`, `VectorStore`, `BaseReranker`) and `HybridRetriever` only composes them. Component responsibilities, contracts, vector backends and the deterministic-ID scheme: [docs/architecture.md](docs/architecture.md).

### Generation

- **openai/gpt-oss-120b** through the Groq API
- Multi-turn conversation memory
- Automatic session management
- Context truncation for token safety
- Retry/fallback handling

### Evaluation

- **RAGAS-style LLM-as-judge evaluation**
- Faithfulness
- Answer relevancy
- Context precision
- Context recall

## 🚀 Tech Stack

**Backend:** FastAPI
**Frontend:** Streamlit
**Retrieval:** BM25 + FAISS + RRF
**Embeddings:** BAAI/bge-large (local, Redis-cached)
**Reranking:** Cohere
**LLM:** openai/gpt-oss-120b via Groq
**Evaluation:** RAGAS-style LLM-as-judge
**Deployment:** Docker · Hugging Face Spaces

## ⚡ Quick Start

### 1. Clone the repo

This repo uses **Git LFS** for pre-built index files (`data/processed/*.bin`, `*.pkl`). Without LFS, those files download as small pointer files and the app starts with an empty index.

```bash
git lfs install
git clone https://github.com/Sohel2309/Enterprise-RAG.git
cd Enterprise-RAG
git lfs pull
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_key          # required — generation & evaluation
COHERE_API_KEY=your_cohere_key      # optional — reranking falls back to
                                     # rank-order without it
REDIS_URL=redis://localhost:6379    # optional — embedding cache; app runs
                                     # fine without Redis, just uncached
```

Get a free `GROQ_API_KEY` at [console.groq.com](https://console.groq.com) and a free `COHERE_API_KEY` at [dashboard.cohere.com](https://dashboard.cohere.com).

### 4. (Optional) Start Redis for embedding caching

```bash
docker compose up -d
```

### 5. Run the app

```bash
uvicorn api.rag_api:app --reload --port 8000   # Terminal 1 — backend
streamlit run ui/streamlit_app.py              # Terminal 2 — frontend
```

Then visit:

```text
http://localhost:8501
```

### Or run with Docker

The app also ships as a single container (this is how the live demo is deployed) that runs the FastAPI backend and Streamlit frontend together:

```bash
docker build -t enterprise-rag .
docker run -p 7860:7860 --env-file .env enterprise-rag
```

Then visit `http://localhost:7860`.

## 🧭 Adaptive Routing & Evidence Gating

`/query` routes each question before retrieval and checks the evidence before generating:

```text
question -> Router (deterministic, no LLM)
   LOW  : hybrid retrieval
   HIGH : split into sub-queries -> hybrid retrieval each -> merge (RRF)
        -> Cohere rerank once, against the original question
        -> Evidence evaluator (top-k reranker scores vs configured thresholds)
   weak -> ONE reformulated retrieval -> rerank -> evaluate again
   still weak -> refuse: "Insufficient evidence to answer accurately."
   otherwise  -> generate
```

A question is routed HIGH when it contains multiple questions, an enumeration, a
comparison, a multi-document cue, or is very long. Cohere scores are **ranking
signals, not probabilities**, so the evidence check compares raw scores against
configurable thresholds. The default is an **uncalibrated placeholder**, to be tuned
on the Dimension 1 benchmark. This is a mechanism; its benefit has not yet been
measured.

Responses gain `refused`, `refusal_reason`, `routing` and `evidence` fields. Configure via
environment variables (see `.env.example`); `ADAPTIVE_ROUTING_ENABLED=false` restores the
previous retrieval path and response shape exactly.

## 🛡️ Security & Adversarial Guardrails

Prompt-injection protection for `/query`, implemented as four small deterministic layers (input guard → retrieved-text sanitizer → prompt hardening → output guard) with no extra service or model call. It is on by default; `SECURITY_GUARDRAILS_ENABLED=false` restores the previous behavior exactly.

An **80-case adversarial benchmark** (direct injection, system-prompt override, prompt/secret extraction, indirect injection via retrieved documents, instructions embedded in relevant documents) plus 20 benign controls lives in `data/security/`, separate from the retrieval benchmark. **Attack Success Rate (ASR)** is decided by a string oracle on the final answer and reported before/after guardrails:

```bash
python -m pytest tests/test_security_*.py -q
python scripts/evaluate_security.py --mode offline --ablation      # worst-case simulated model, no keys
python scripts/evaluate_security.py --mode live --arms baseline    # real retrieval + Groq (resumable)
python scripts/evaluate_security.py --mode live --arms guarded
```

The committed offline run (worst-case simulated LLM, so baseline = 100% by construction) takes ASR from 100% to 18.8% (95% CI 12–29%) at a 5% false-block rate on benign controls; the survivors are paraphrased and non-English attacks the rules cannot see. Live numbers are produced by the commands above. Threat model, attack categories, defense design, methodology and limitations: [docs/security.md](docs/security.md).

## ⏱️ Performance & Cost Measurement (D5-Lite)

Per-stage latency (dense, BM25, fusion, rerank, routing, reformulation,
generation), provider-reported token usage, and an **estimated** LLM cost per
query, reported in an opt-in `performance` block (`"include_performance": true`
in the `/query` request). The default response is unchanged.

```bash
python scripts/benchmark_performance.py --label adaptive --limit 20 --exclude-categories adversarial
python scripts/analyze_performance.py data/performance/benchmark_adaptive.json
```

Cost is an estimate (reported tokens x a price you configure in
`config/llm_pricing.json` or the environment), **not billing data**. See
[docs/performance.md](docs/performance.md) for what is measured, the
methodology and the limitations. No measurements are quoted here: run the
benchmark on your own deployment.

## 📁 Project Structure

```text
Enterprise-RAG/
├── api/                 # FastAPI backend
├── ingestion/           # PDF/DOCX parsing and chunking
├── retrieval/           # Retriever interfaces, dense/BM25, RRF, reranking, adaptive routing
├── generation/          # LLM client and prompt templates
├── security/            # Prompt-injection guardrails (Dimension 3)
├── performance/         # Latency / token / cost measurement (D5-Lite)
├── config/              # LLM pricing used for cost estimates
├── evaluation/          # RAGAS-style evaluation
├── ui/                  # Streamlit application
├── scripts/             # Data ingestion and evaluation scripts
├── app.py               # Combined FastAPI + Streamlit entrypoint (Docker/HF Spaces)
├── Dockerfile
├── docker-compose.yml    # Local Redis for embedding cache
└── data/
    ├── processed/        # Pre-built indexes (Git LFS)
    └── security/         # Adversarial benchmark + security evaluation reports
```

## 🎯 Features

- ✅ Hybrid BM25 + dense vector retrieval
- ✅ Reciprocal Rank Fusion
- ✅ Cohere intelligent reranking
- ✅ Multi-turn conversation memory
- ✅ Real-time PDF/DOCX document upload
- ✅ Automatic document indexing
- ✅ RAGAS evaluation dashboard
- ✅ Interactive Streamlit interface
- ✅ FastAPI backend
- ✅ Session management
- ✅ Pre-indexed document collection (50 arXiv papers, ~7,500 chunks)

## 📈 Benchmark

Fixed-size chunking outperformed sentence-based chunking by **17% on context precision** in the project benchmark.

The evaluation dataset was generated from **50 ArXiv papers** with a larger QA set during experimentation. The currently reported production evaluation contains **45 valid samples**.

## 🔍 Evaluation Methodology & Limitations

The metrics reported above come from an **LLM-as-judge evaluation**, not a human-labeled benchmark.

The evaluation uses `openai/gpt-oss-120b` against a held-out QA set using the same retrieval configuration as production:

```text
BM25 + Dense Retrieval
        ↓
RRF Fusion
        ↓
Cohere Reranking
        ↓
LLM Generation
        ↓
RAGAS Evaluation
```

The current reported run contains **45 valid samples**.

### Limitations

**Self-judging bias:**
The same model family (`openai/gpt-oss-120b`) is used for generation and evaluation. Therefore, the scores should be interpreted as internal project-level evidence rather than an independent industry benchmark.

**Sample size:**
45 samples provide directional evidence of system performance but are not sufficient for statistically rigorous benchmarking.

The purpose of the evaluation is to demonstrate that retrieval and generation quality were **measured, analyzed, and iterated on**, rather than assumed to work.

## 🔗 Links

- **Live Demo:** https://sohel2309-enterprise-rag.hf.space
- **GitHub:** https://github.com/Sohel2309/Enterprise-RAG

---

**MIT License**
