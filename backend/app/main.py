"""
Mini RAG - FastAPI Backend
═══════════════════════════════════════════════════════════════════════════════

Endpoints:
- POST /ingest    - Ingest text into vector store
- POST /upload    - Upload and ingest file (PDF, DOCX, TXT)
- POST /query     - Query with RAG pipeline
- GET  /documents - List all documents
- DELETE /documents/{doc_id} - Delete a document
- GET  /health    - Health check

CORS & OPTIONS:
- CORSMiddleware is added FIRST (before any routes)
- OPTIONS requests are handled automatically by the middleware
- This prevents 502 errors on preflight requests
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from .schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, Source,
    DocumentInfo, DeleteResponse,
    HealthResponse, UploadResponse
)

# Load environment variables FIRST
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# APP SETUP - CORS MUST BE CONFIGURED IMMEDIATELY
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # NOTE: Model loading is LAZY (on first actual request)
    # This allows Render free tier (512MB) to boot fast
    # OPTIONS requests will NOT trigger model loading
    print("✅ App started (models will load on first POST request)...")
    yield
    print("Shutting down.")


# Create app with minimal startup
app = FastAPI(
    title="Mini RAG API",
    description="Minimal RAG system with retrieval, reranking, and cited answers",
    version="1.0.0",
    lifespan=lifespan
)


# ═══════════════════════════════════════════════════════════════════════════
# CORS MIDDLEWARE - MUST BE ADDED FIRST, BEFORE ANY ROUTES
# ═══════════════════════════════════════════════════════════════════════════
# This handles OPTIONS preflight requests automatically
# CORSMiddleware intercepts OPTIONS before they reach route handlers

cors_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://mini-ra-g-project-jhxq.vercel.app",
]

# Also allow origins from environment variable if set
env_origins = os.getenv("CORS_ORIGINS", "")
if env_origins:
    cors_origins.extend([o.strip() for o in env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight for 10 minutes
)


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the service is running."""
    return HealthResponse(status="healthy")


@app.get("/debug/env", tags=["Debug"])
async def debug_env():
    """Check if required environment variables are set (doesn't expose values)."""
    return {
        "PINECONE_API_KEY": "✅ Set" if os.getenv("PINECONE_API_KEY") else "❌ Missing",
        "GROQ_API_KEY": "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Missing",
        "COHERE_API_KEY": "✅ Set" if os.getenv("COHERE_API_KEY") else "❌ Missing",
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "mini-rag (default)"),
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Ingest"])
async def ingest_document(request: IngestRequest):
    """
    Ingest text into the RAG system.
    
    - Chunks text with token-based splitting (1000 tokens, 120 overlap)
    - Embeds chunks using all-MiniLM-L6-v2
    - Stores in Pinecone vector database
    """
    # Check required env vars BEFORE importing heavy modules
    if not os.getenv("PINECONE_API_KEY"):
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY not configured")
    
    try:
        from .services.pipeline import ingest_text
        
        result = ingest_text(
            text=request.text,
            doc_id=request.doc_id,
            metadata=request.metadata or {}
        )
        return IngestResponse(**result)
    except Exception as e:
        import traceback
        print(f"Ingest error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse, tags=["Ingest"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and ingest a file into the RAG system.
    
    Supported formats: .txt, .pdf, .docx
    Max file size: 10MB
    
    The file is:
    1. Validated (type & size)
    2. Text extracted
    3. Chunked, embedded, and stored (reuses existing pipeline)
    """
    from .services.pipeline import ingest_text
    from .services.file_extractor import (
        extract_text, validate_file,
        FileExtractionError, UnsupportedFileTypeError, FileTooLargeError
    )
    
    filename = file.filename or "unknown"
    
    try:
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file type and size
        file_ext = validate_file(filename, file_size)
        
        # Extract text from file
        text = extract_text(filename, content)
        
        if not text.strip():
            return UploadResponse(
                doc_id="",
                filename=filename,
                file_type=file_ext,
                chunks_created=0,
                status="error",
                message="No text content found in file"
            )
        
        # Generate doc_id from filename (sanitized)
        doc_id = Path(filename).stem.lower()
        doc_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc_id)
        doc_id = doc_id[:50]  # Limit length
        
        # Build metadata for citations
        metadata = {
            "title": filename,
            "filename": filename,
            "file_type": file_ext,
            "source": "upload",
            "file_size_bytes": file_size
        }
        
        # Reuse existing ingest pipeline
        result = ingest_text(
            text=text,
            doc_id=doc_id,
            metadata=metadata
        )
        
        return UploadResponse(
            doc_id=result["doc_id"],
            filename=filename,
            file_type=file_ext,
            chunks_created=result["chunks_created"],
            status=result["status"],
            message=f"Successfully processed {filename}"
        )
        
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    
    except FileExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query the RAG system.
    
    Pipeline:
    1. Embed query
    2. Retrieve top-20 chunks from Pinecone
    3. Rerank with Cohere to top-k
    4. Generate answer with Groq LLM
    5. Return answer with inline citations
    """
    from .services.pipeline import rag_pipeline
    
    try:
        result = rag_pipeline(
            query=request.question,
            retrieve_k=20,
            rerank_k=request.top_k,
            doc_id=request.doc_id
        )
        
        # Convert to response model
        sources = [
            Source(
                index=src["index"],
                chunk_id=src["chunk_id"],
                doc_id=src["doc_id"],
                text=src["text"],
                score=src["score"],
                title=src.get("title"),
                char_start=src.get("char_start", 0),
                char_end=src.get("char_end", 0)
            )
            for src in result.sources
        ]
        
        return QueryResponse(
            answer=result.answer,
            sources=sources,
            has_answer=result.has_answer,
            tokens_used=result.tokens_used,
            retrieval_count=result.retrieval_count,
            rerank_count=result.rerank_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all ingested documents."""
    from .services.vector_store import list_documents
    
    try:
        docs = list_documents()
        return [
            DocumentInfo(
                doc_id=doc["doc_id"],
                title=doc.get("title"),
                chunk_count=doc.get("total_chunks", 0),
                created_at=doc.get("created_at")
            )
            for doc in docs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and all its chunks."""
    from .services.vector_store import delete_document
    
    try:
        result = delete_document(doc_id)
        return DeleteResponse(doc_id=result["deleted_doc_id"], status=result["status"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# ROOT
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """API root with basic info."""
    return {
        "name": "Mini RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
