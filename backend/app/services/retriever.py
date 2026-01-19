"""
Vector retrieval for RAG pipeline.

Retrieval Strategy:
- Embed query using same model as documents (all-MiniLM-L6-v2)
- Retrieve top-k chunks via cosine similarity
- Return chunks with metadata for reranking and citation

k Value Guidance:
- k=20 (default): Cast wide net before reranking narrows to top 5
- k=5-10: Direct use without reranker, lower latency
- k=50+: Complex queries needing broad context, higher cost

NOTE: Lazy imports used to reduce startup memory for Render free tier.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance score and citation metadata."""
    chunk_id: str
    doc_id: str
    text: str
    score: float              # Cosine similarity (0-1, higher = better)
    chunk_index: int          # Position in source document
    total_chunks: int         # Total chunks in source document
    char_start: int           # Character offset for highlighting
    char_end: int
    metadata: dict            # Title, source, etc.

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": self.metadata
        }


def retrieve(
    query: str,
    top_k: int = 20,
    doc_id: Optional[str] = None,
    min_score: float = 0.0
) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks for a query.
    
    Args:
        query: Natural language search query
        top_k: Number of chunks to retrieve (default 20 for reranking)
        doc_id: Optional filter to search within a specific document
        min_score: Minimum similarity threshold (0-1)
        
    Returns:
        List of RetrievedChunk sorted by relevance (highest first)
        
    Example:
        chunks = retrieve("What causes climate change?", top_k=20)
        for chunk in chunks[:5]:
            print(f"[{chunk.score:.3f}] {chunk.text[:100]}...")
    """
    from .vector_store import query_similar  # Lazy import
    
    if not query or not query.strip():
        return []
    
    # Query vector store
    matches = query_similar(
        query=query,
        top_k=top_k,
        filter_doc_id=doc_id
    )
    
    # Convert to RetrievedChunk objects
    chunks = []
    for match in matches:
        meta = match["metadata"]
        score = match["score"]
        
        # Apply minimum score filter
        if score < min_score:
            continue
        
        # Extract user metadata (everything not in core fields)
        core_fields = {
            "text", "doc_id", "chunk_index", "total_chunks",
            "char_start", "char_end", "token_count", "created_at"
        }
        user_metadata = {
            k: v for k, v in meta.items() 
            if k not in core_fields
        }
        
        chunk = RetrievedChunk(
            chunk_id=match["id"],
            doc_id=meta.get("doc_id", ""),
            text=meta.get("text", ""),
            score=score,
            chunk_index=meta.get("chunk_index", 0),
            total_chunks=meta.get("total_chunks", 1),
            char_start=meta.get("char_start", 0),
            char_end=meta.get("char_end", 0),
            metadata=user_metadata
        )
        chunks.append(chunk)
    
    return chunks


def retrieve_as_context(
    query: str,
    top_k: int = 5,
    doc_id: Optional[str] = None
) -> tuple[str, list[RetrievedChunk]]:
    """
    Retrieve chunks and format as LLM context string.
    
    Returns:
        (formatted_context, chunks) tuple
        
    The context string is formatted for injection into LLM prompts:
        [1] First chunk text...
        [2] Second chunk text...
    """
    chunks = retrieve(query, top_k=top_k, doc_id=doc_id)
    
    # Format as numbered context
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] {chunk.text}")
    
    context = "\n\n".join(context_parts)
    return context, chunks
