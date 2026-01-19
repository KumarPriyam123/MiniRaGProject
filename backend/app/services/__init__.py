# Services module
# NOTE: Lazy imports to avoid loading heavy models at startup (Render 512MB limit)
# Import directly from submodules when needed:
#   from app.services.pipeline import rag_pipeline, ingest_text
#   from app.services.embedder import embed_text

# EMBEDDING_DIMENSION is a constant, so we can safely load it
# However, to be extra safe, let's define it here directly
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 dimension

__all__ = [
    "EMBEDDING_DIMENSION",
]


def __getattr__(name):
    """Lazy import for heavy modules to reduce startup memory."""
    # Chunker
    if name in ("chunk_text", "get_chunker", "TokenChunker", "Chunk"):
        from . import chunker
        return getattr(chunker, name)
    # Embedder
    if name in ("embed_text", "embed_texts"):
        from . import embedder
        return getattr(embedder, name)
    # Vector store
    if name in ("upsert_chunks", "query_similar", "delete_document", 
                "list_documents", "get_index_stats"):
        from . import vector_store
        return getattr(vector_store, name)
    # Retriever
    if name in ("retrieve", "retrieve_as_context", "RetrievedChunk"):
        from . import retriever
        return getattr(retriever, name)
    # Reranker
    if name in ("rerank", "retrieve_and_rerank", "format_context_with_citations"):
        from . import reranker
        return getattr(reranker, name)
    # LLM
    if name in ("generate_answer", "answer_question", "AnswerResponse", "Citation"):
        from . import llm
        return getattr(llm, name)
    # Pipeline
    if name in ("rag_pipeline", "ingest_text", "RAGResult"):
        from . import pipeline
        return getattr(pipeline, name)
    
    raise AttributeError(f"module 'services' has no attribute '{name}'")
