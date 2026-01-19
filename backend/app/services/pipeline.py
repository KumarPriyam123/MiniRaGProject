"""
Unified RAG Pipeline
═══════════════════════════════════════════════════════════════════════════════

Flow: Query → Embed → Retrieve → Rerank → LLM → Answer + Citations

This is the main entry point for the RAG system. One function, clear flow.

NOTE: All heavy imports (embedder, vector_store, reranker, llm) are done
inside functions to enable lazy loading. This reduces startup memory,
allowing deployment on Render free tier (512MB).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

# Type hints only - no runtime import
if TYPE_CHECKING:
    from .retriever import RetrievedChunk
    from .llm import AnswerResponse


@dataclass
class RAGResult:
    """Complete RAG pipeline result."""
    answer: str                         # Generated answer with [1][2] citations
    sources: list[dict]                 # Cited source chunks
    has_answer: bool                    # False if context was insufficient
    tokens_used: int                    # LLM token consumption
    retrieval_count: int                # Chunks retrieved from vector DB
    rerank_count: int                   # Chunks after reranking
    
    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "has_answer": self.has_answer,
            "tokens_used": self.tokens_used,
            "retrieval_count": self.retrieval_count,
            "rerank_count": self.rerank_count
        }


def rag_pipeline(
    query: str,
    retrieve_k: int = 20,
    rerank_k: int = 5,
    doc_id: Optional[str] = None
) -> RAGResult:
    """
    Execute the complete RAG pipeline.
    
    Query → Embed → Retrieve → Rerank → LLM → Answer + Citations
    
    Args:
        query: User's natural language question
        retrieve_k: Number of chunks to fetch from vector DB (cast wide net)
        rerank_k: Number of chunks to keep after reranking (for LLM context)
        doc_id: Optional filter to search within a specific document
        
    Returns:
        RAGResult with answer, sources, and metadata
        
    Example:
        result = rag_pipeline("What causes climate change?")
        print(result.answer)  # "Climate change is caused by... [1][2]"
        for src in result.sources:
            print(f"[{src['index']}] {src['text'][:100]}...")
    """
    # Lazy imports to reduce startup memory
    from .vector_store import query_similar
    from .retriever import RetrievedChunk
    from .reranker import rerank
    from .llm import generate_answer
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: RETRIEVE
    # Embed query and fetch top-k similar chunks from vector DB
    # ─────────────────────────────────────────────────────────────────────────
    
    # Build optional filter
    filter_dict = {"doc_id": {"$eq": doc_id}} if doc_id else None
    
    # Query vector store (embeds query internally)
    matches = query_similar(query=query, top_k=retrieve_k, filter_doc_id=doc_id)
    
    # Convert to RetrievedChunk objects
    retrieved_chunks = []
    for match in matches:
        meta = match["metadata"]
        # Separate core fields from user metadata
        core_fields = {"text", "doc_id", "chunk_index", "total_chunks",
                       "char_start", "char_end", "token_count", "created_at"}
        user_meta = {k: v for k, v in meta.items() if k not in core_fields}
        
        chunk = RetrievedChunk(
            chunk_id=match["id"],
            doc_id=meta.get("doc_id", ""),
            text=meta.get("text", ""),
            score=match["score"],
            chunk_index=meta.get("chunk_index", 0),
            total_chunks=meta.get("total_chunks", 1),
            char_start=meta.get("char_start", 0),
            char_end=meta.get("char_end", 0),
            metadata=user_meta
        )
        retrieved_chunks.append(chunk)
    
    retrieval_count = len(retrieved_chunks)
    
    # Handle empty retrieval
    if not retrieved_chunks:
        return RAGResult(
            answer="I cannot answer this question as no relevant documents were found in the knowledge base.",
            sources=[],
            has_answer=False,
            tokens_used=0,
            retrieval_count=0,
            rerank_count=0
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: RERANK
    # Use cross-encoder to re-score chunks by relevance to query
    # ─────────────────────────────────────────────────────────────────────────
    
    reranked_chunks = rerank(
        query=query,
        chunks=retrieved_chunks,
        top_k=rerank_k
    )
    
    rerank_count = len(reranked_chunks)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: GENERATE
    # Send reranked chunks + query to LLM for cited answer
    # ─────────────────────────────────────────────────────────────────────────
    
    llm_response: AnswerResponse = generate_answer(
        question=query,
        chunks=reranked_chunks
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: FORMAT RESPONSE
    # Build clean response with sources for frontend
    # ─────────────────────────────────────────────────────────────────────────
    
    sources = []
    for i, chunk in enumerate(reranked_chunks, 1):
        sources.append({
            "index": i,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "score": round(chunk.score, 4),
            "title": chunk.metadata.get("title", "Untitled"),
            "char_start": chunk.char_start,
            "char_end": chunk.char_end
        })
    
    return RAGResult(
        answer=llm_response.answer,
        sources=sources,
        has_answer=llm_response.has_answer,
        tokens_used=llm_response.tokens_used,
        retrieval_count=retrieval_count,
        rerank_count=rerank_count
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INGEST PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_text(
    text: str,
    doc_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 120
) -> dict:
    """
    Ingest text into the RAG system.
    
    Text → Chunk → Embed → Store
    
    Args:
        text: Raw text content to ingest
        doc_id: Optional document ID (auto-generated if None)
        metadata: Optional metadata (title, source, etc.)
        chunk_size: Tokens per chunk
        chunk_overlap: Overlap tokens between chunks
        
    Returns:
        {"doc_id": str, "chunks_created": int, "status": str}
    """
    from .chunker import chunk_text
    from .vector_store import upsert_chunks
    
    # Chunk the text
    chunks = chunk_text(
        text=text,
        doc_id=doc_id,
        metadata=metadata or {},
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not chunks:
        return {"doc_id": None, "chunks_created": 0, "status": "empty_text"}
    
    # Embed and store
    result = upsert_chunks(chunks)
    
    return {
        "doc_id": result["doc_id"],
        "chunks_created": result["upserted_count"],
        "status": "success"
    }
