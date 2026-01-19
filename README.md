# Mini RAG System

A minimal Retrieval-Augmented Generation system built for a 4-hour assessment. Prioritizes correctness, clarity, and free-tier compatibility over feature completeness.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│  │  INGEST  │     │ RETRIEVE │     │  RERANK  │     │ GENERATE │          │
│  │          │     │          │     │          │     │          │          │
│  │ Text     │     │ Pinecone │     │ Cohere   │     │ Groq     │          │
│  │ → Chunk  │────►│ top-20   │────►│ top-5    │────►│ Llama-3  │          │
│  │ → Embed  │     │ vectors  │     │ reranked │     │ + cite   │          │
│  │ → Store  │     │          │     │          │     │          │          │
│  └──────────┘     └──────────┘     └──────────┘     └──────────┘          │
│       │                │                │                │                 │
│   ~2s/doc          ~100ms           ~200ms           ~500ms                │
│                                                                             │
│                    Total query latency: ~800ms                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Provider | Free Tier | Why This Choice |
|-----------|----------|-----------|-----------------|
| **Embeddings** | `all-MiniLM-L6-v2` | Unlimited (local) | Fast, 384-dim, no API costs |
| **Vector DB** | Pinecone Serverless | 100K vectors | Hosted, no infra, good SDK |
| **Reranker** | Cohere Rerank v3 | 1K calls/mo | Best free reranker available |
| **LLM** | Groq (Llama-3.1-8B) | 14.4K req/day | Fastest inference, free |
| **Backend** | FastAPI | — | Async, typed, minimal |
| **Frontend** | React + Vite | — | Simple, fast HMR |

## Chunking Strategy

```python
TokenChunker(chunk_size=1000, chunk_overlap=120)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Size** | 1000 tokens | ~750 words. Balances retrieval precision vs semantic completeness |
| **Overlap** | 120 tokens (12%) | Prevents context loss at boundaries. 5-6 sentences of bleed |
| **Tokenizer** | tiktoken `cl100k_base` | Matches LLM tokenization for predictable context windows |
| **Splitter** | LangChain Recursive | Respects paragraph → sentence → word boundaries |

**Tradeoff**: Smaller chunks (256) improve precision but require more reranking compute. 1000 is the empirical sweet spot for general-purpose RAG.

## Retrieval Pipeline

### Stage 1: Vector Search (Recall-focused)
- Embed query with same model as documents
- Cosine similarity search in Pinecone
- Retrieve top-20 candidates
- **Purpose**: Cast wide net, ensure answer is in candidate set

### Stage 2: Reranking (Precision-focused)
- Cohere cross-encoder scores (query, chunk) pairs
- Full transformer attention between query and each chunk
- Keep top-5 most relevant
- **Purpose**: Vector search finds *related* text; reranker finds *answering* text

```
Vector Search: "Python frameworks" → "Python is a language..." (topically related)
After Rerank:  "Python frameworks" → "FastAPI and Flask are..." (actually answers)
```

## Citation Implementation

**Prompt Engineering**:
```
STRICT RULES:
1. Use ONLY information from numbered context chunks
2. Cite with inline brackets: [1], [2]
3. If context insufficient, say "I cannot answer..."
```

**Response Format**:
```json
{
  "answer": "Climate change is caused by greenhouse gases [1]. CO2 levels reached 421 ppm [2].",
  "sources": [
    {"index": 1, "text": "...", "title": "IPCC Report", "score": 0.89},
    {"index": 2, "text": "...", "title": "NASA Data", "score": 0.84}
  ],
  "has_answer": true,
  "tokens_used": 547
}
```

## Evaluation

### Test Document
Ingested 3 pages of Wikipedia content on "Climate Change" (~4000 tokens, 5 chunks).

### QA Pairs

| # | Question | Expected | Actual | Citations | Pass |
|---|----------|----------|--------|-----------|------|
| 1 | What is the main cause of climate change? | Human GHG emissions | "Human activities, particularly burning fossil fuels, are the dominant cause [1][3]" | ✓ Correct | ✅ |
| 2 | What is the current CO2 level? | ~420 ppm | "CO2 levels reached 421 ppm in 2023 [2]" | ✓ Correct | ✅ |
| 3 | How much has temperature risen? | ~1.1°C | "Global temperatures have risen 1.1°C since pre-industrial times [1]" | ✓ Correct | ✅ |
| 4 | What is the Paris Agreement target? | 1.5°C limit | "The Paris Agreement aims to limit warming to 1.5°C [4]" | ✓ Correct | ✅ |
| 5 | What is quantum computing? | Should decline | "I cannot answer this based on the provided documents." | N/A | ✅ |

**Result**: 5/5 pass. System correctly answers from context and declines out-of-scope questions.

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/ingest` | Chunk, embed, store text |
| `POST` | `/query` | Retrieve → Rerank → Generate |
| `GET` | `/documents` | List ingested docs |
| `DELETE` | `/documents/{id}` | Remove doc and chunks |
| `GET` | `/health` | Liveness check |

## Tradeoffs & Limitations

| Decision | Tradeoff | Mitigation |
|----------|----------|------------|
| **Local embeddings** | Slower than API (~50ms vs 10ms) | Batched encoding, acceptable for demo scale |
| **1000-token chunks** | May split related content | 12% overlap preserves boundary context |
| **Top-20 → Top-5 pipeline** | More Cohere API calls | Single rerank call for 20 docs is fine |
| **Groq Llama-3-8B** | Less capable than GPT-4 | Sufficient for extractive QA with good context |
| **No auth** | Public access | Add API key middleware for production |
| **No persistence** | Docs lost on Pinecone reset | Acceptable for assessment scope |

### Known Limitations

1. **No PDF parsing** — Text paste only (add PyMuPDF for PDF support)
2. **No conversation memory** — Single-turn QA only
3. **English only** — Switch to `multilingual-MiniLM` for other languages
4. **Cold start** — Render free tier sleeps after 15min (~30s wake)

## Local Development

```bash
# Backend
cd backend
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env  # Add API keys
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install && npm run dev
```

## Deployment

| Component | Platform | URL |
|-----------|----------|-----|
| Frontend | Vercel | `https://mini-rag.vercel.app` |
| Backend | Render | `https://mini-rag-api.onrender.com` |

**Total cost**: $0/month (all free tiers)

## Project Structure

```
MiniProjectRAG/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── schemas.py           # Pydantic models
│   │   └── services/
│   │       ├── chunker.py       # Token-based splitting
│   │       ├── embedder.py      # Sentence-transformers
│   │       ├── vector_store.py  # Pinecone operations
│   │       ├── retriever.py     # Vector search
│   │       ├── reranker.py      # Cohere rerank
│   │       ├── llm.py           # Groq generation
│   │       └── pipeline.py      # Unified RAG pipeline
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx              # Main UI component
    │   └── index.css            # Minimal styling
    └── package.json
```

## Time Breakdown

| Task | Time |
|------|------|
| Architecture design | 20 min |
| Chunking + embeddings | 30 min |
| Vector store integration | 30 min |
| Retrieval + reranking | 30 min |
| LLM + citations | 30 min |
| FastAPI endpoints | 30 min |
| Frontend UI | 40 min |
| Testing + docs | 30 min |
| **Total** | **~4 hours** |

---

*Built for a Mini RAG assessment. Optimized for clarity and correctness over feature completeness.*
