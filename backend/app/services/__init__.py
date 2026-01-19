# Services module
from .chunker import chunk_text, get_chunker, TokenChunker, Chunk
from .embedder import embed_text, embed_texts, EMBEDDING_DIMENSION
from .vector_store import (
    upsert_chunks,
    query_similar,
    delete_document,
    list_documents,
    get_index_stats
)
from .retriever import retrieve, retrieve_as_context, RetrievedChunk
from .reranker import rerank, retrieve_and_rerank, format_context_with_citations
from .llm import generate_answer, answer_question, AnswerResponse, Citation
from .pipeline import rag_pipeline, ingest_text, RAGResult

__all__ = [
    # Chunker
    "chunk_text", "get_chunker", "TokenChunker", "Chunk",
    # Embedder
    "embed_text", "embed_texts", "EMBEDDING_DIMENSION",
    # Vector store
    "upsert_chunks", "query_similar", "delete_document", 
    "list_documents", "get_index_stats",
    # Retriever
    "retrieve", "retrieve_as_context", "RetrievedChunk",
    # Reranker
    "rerank", "retrieve_and_rerank", "format_context_with_citations",
    # LLM
    "generate_answer", "answer_question", "AnswerResponse", "Citation",
    # Pipeline (main entry points)
    "rag_pipeline", "ingest_text", "RAGResult"
]
