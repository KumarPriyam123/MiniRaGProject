"""
Reranking service using Cohere Rerank API.

Why Reranking is Essential in RAG:
─────────────────────────────────────
Vector similarity (embedding dot product) is APPROXIMATE:
- Embeddings compress semantics into fixed dimensions (384)
- Information loss is inevitable
- "Similar vectors" ≠ "Best answer for this question"

Rerankers use CROSS-ATTENTION:
- Process (query, document) pairs jointly
- Full transformer attention between query and each chunk
- Much more accurate relevance scoring
- But expensive: O(n) model calls vs O(1) for vector search

The Two-Stage Pattern:
1. Vector search: Fast, cheap, recall-focused (top-20)
2. Rerank: Slow, accurate, precision-focused (top-5)

Result: Best of both worlds — speed AND accuracy.

Cohere Rerank:
- Free tier: 1,000 calls/month
- Model: rerank-english-v3.0 (or rerank-multilingual-v3.0)
- Latency: ~200ms for 20 documents
"""

import os
from typing import Optional
import cohere

from .retriever import RetrievedChunk


# Cohere client singleton
_client: Optional[cohere.Client] = None


def get_client() -> cohere.Client:
    """Get or create Cohere client."""
    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        _client = cohere.Client(api_key)
    return _client


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
    model: str = "rerank-english-v3.0"
) -> list[RetrievedChunk]:
    """
    Rerank retrieved chunks using Cohere Rerank API.
    
    Args:
        query: The user's question
        chunks: List of chunks from vector retrieval
        top_k: Number of top chunks to return after reranking
        model: Cohere rerank model (english or multilingual)
        
    Returns:
        Top-k chunks reordered by relevance, with updated scores
        
    Example:
        # Retrieve 20, rerank to top 5
        chunks = retrieve(query, top_k=20)
        reranked = rerank(query, chunks, top_k=5)
    """
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        # No need to rerank if we have fewer chunks than requested
        return chunks
    
    client = get_client()
    
    # Extract texts for reranking
    documents = [chunk.text for chunk in chunks]
    
    # Call Cohere Rerank API
    response = client.rerank(
        query=query,
        documents=documents,
        top_n=top_k,
        model=model,
        return_documents=False  # We already have the texts
    )
    
    # Rebuild chunks with reranker scores
    reranked_chunks = []
    for result in response.results:
        original_chunk = chunks[result.index]
        
        # Create new chunk with reranker score
        reranked_chunk = RetrievedChunk(
            chunk_id=original_chunk.chunk_id,
            doc_id=original_chunk.doc_id,
            text=original_chunk.text,
            score=result.relevance_score,  # Reranker score (0-1)
            chunk_index=original_chunk.chunk_index,
            total_chunks=original_chunk.total_chunks,
            char_start=original_chunk.char_start,
            char_end=original_chunk.char_end,
            metadata=original_chunk.metadata
        )
        reranked_chunks.append(reranked_chunk)
    
    return reranked_chunks


def retrieve_and_rerank(
    query: str,
    retrieve_k: int = 20,
    rerank_k: int = 5,
    doc_id: Optional[str] = None
) -> list[RetrievedChunk]:
    """
    Full retrieval pipeline: vector search → rerank.
    
    Args:
        query: User's question
        retrieve_k: Chunks to fetch from vector DB (cast wide net)
        rerank_k: Final chunks after reranking (for LLM context)
        doc_id: Optional filter to specific document
        
    Returns:
        Top rerank_k chunks sorted by reranker relevance
    """
    from .retriever import retrieve
    
    # Stage 1: Vector retrieval (fast, approximate)
    candidates = retrieve(query, top_k=retrieve_k, doc_id=doc_id)
    
    if not candidates:
        return []
    
    # Stage 2: Rerank (slow, accurate)
    reranked = rerank(query, candidates, top_k=rerank_k)
    
    return reranked


def format_context_with_citations(chunks: list[RetrievedChunk]) -> str:
    """
    Format reranked chunks as numbered context for LLM.
    
    Returns string like:
        [1] First relevant chunk text...
        
        [2] Second relevant chunk text...
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk.text}")
    return "\n\n".join(parts)
