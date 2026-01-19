"""
Pinecone vector store operations.

Index Configuration:
- Dimension: 384 (matches all-MiniLM-L6-v2)
- Metric: cosine (normalized embeddings)
- Pod type: starter (free tier)

ID Strategy:
- Vector ID = "{doc_id}_chunk_{index:04d}"
- Enables: prefix filtering by doc_id, ordered retrieval, easy deletion

Metadata Strategy:
- Store all Chunk fields as flat key-value pairs
- Pinecone metadata supports: str, int, float, bool, list[str]
- Text stored in metadata for retrieval (Pinecone doesn't return vectors)

NOTE: Lazy imports used for embedder to reduce startup memory.
"""

from __future__ import annotations
import os
from typing import Optional, TYPE_CHECKING

from pinecone import Pinecone

if TYPE_CHECKING:
    from .chunker import Chunk


# Index configuration
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mini-rag")
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
DIMENSION = EMBEDDING_DIMENSION
METRIC = "cosine"

# Singleton client
_client: Optional[Pinecone] = None
_index = None


def get_client() -> Pinecone:
    """Get or create Pinecone client."""
    global _client
    if _client is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        _client = Pinecone(api_key=api_key)
    return _client


def get_index():
    """Get or create the Pinecone index."""
    global _index
    if _index is None:
        client = get_client()
        
        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in client.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            client.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric=METRIC,
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        
        _index = client.Index(INDEX_NAME)
    return _index


def upsert_chunks(chunks: list) -> dict:
    """
    Embed and upsert chunks to Pinecone.
    
    Args:
        chunks: List of Chunk objects from chunker
        
    Returns:
        {"upserted_count": int, "doc_id": str}
    """
    from .embedder import embed_texts  # Lazy import
    
    if not chunks:
        return {"upserted_count": 0, "doc_id": None}
    
    index = get_index()
    
    # Batch embed all chunk texts
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)
    
    # Prepare vectors for upsert
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk.chunk_id,  # "{doc_id}_chunk_{index:04d}"
            "values": embedding,
            "metadata": {
                # Core fields for retrieval
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.index,
                "total_chunks": chunk.total_chunks,
                # Citation tracking
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_count": chunk.token_count,
                "created_at": chunk.created_at,
                # User metadata (flattened)
                **{k: v for k, v in chunk.metadata.items() 
                   if isinstance(v, (str, int, float, bool))}
            }
        })
    
    # Upsert in batches of 100 (Pinecone limit)
    batch_size = 100
    total_upserted = 0
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        result = index.upsert(vectors=batch)
        total_upserted += result.upserted_count
    
    return {
        "upserted_count": total_upserted,
        "doc_id": chunks[0].doc_id
    }


def query_similar(
    query: str,
    top_k: int = 20,
    filter_doc_id: Optional[str] = None
) -> list[dict]:
    """
    Query for similar chunks.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        filter_doc_id: Optional doc_id to filter results
        
    Returns:
        List of {id, score, metadata} dicts
    """
    from .embedder import embed_text  # Lazy import
    
    index = get_index()
    
    # Embed query
    query_embedding = embed_text(query)
    
    # Build filter
    filter_dict = None
    if filter_doc_id:
        filter_dict = {"doc_id": {"$eq": filter_doc_id}}
    
    # Query
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    
    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata
        }
        for match in results.matches
    ]


def delete_document(doc_id: str) -> dict:
    """
    Delete all chunks for a document.
    
    Uses prefix-based deletion: all IDs starting with "{doc_id}_chunk_"
    """
    index = get_index()
    
    # Pinecone serverless uses filter-based deletion
    index.delete(filter={"doc_id": {"$eq": doc_id}})
    
    return {"deleted_doc_id": doc_id, "status": "success"}


def list_documents() -> list[dict]:
    """
    List all unique document IDs in the index.
    
    Note: Pinecone doesn't support DISTINCT queries, so we sample
    and deduplicate. For production, maintain a separate doc registry.
    """
    index = get_index()
    
    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    
    if total_vectors == 0:
        return []
    
    # Sample vectors to find unique doc_ids
    # This is a workaround - production should use a metadata store
    results = index.query(
        vector=[0.0] * DIMENSION,  # Dummy vector
        top_k=min(1000, total_vectors),
        include_metadata=True
    )
    
    # Deduplicate by doc_id
    docs = {}
    for match in results.matches:
        doc_id = match.metadata.get("doc_id")
        if doc_id and doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "title": match.metadata.get("title", "Untitled"),
                "created_at": match.metadata.get("created_at"),
                "total_chunks": match.metadata.get("total_chunks", 0)
            }
    
    return list(docs.values())


def get_index_stats() -> dict:
    """Get index statistics."""
    index = get_index()
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "dimension": DIMENSION,
        "index_name": INDEX_NAME
    }
