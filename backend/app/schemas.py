"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
# INGEST
# ═══════════════════════════════════════════════════════════════════════════

class IngestRequest(BaseModel):
    """Request to ingest text into the RAG system."""
    text: str = Field(..., min_length=1, description="Raw text content to ingest")
    doc_id: Optional[str] = Field(None, description="Custom document ID (auto-generated if omitted)")
    metadata: Optional[dict] = Field(default_factory=dict, description="Optional metadata (title, source, etc.)")

    model_config = {"json_schema_extra": {
        "example": {
            "text": "Climate change refers to long-term shifts in temperatures...",
            "doc_id": "climate_report_2024",
            "metadata": {"title": "IPCC Climate Report", "source": "ipcc.ch"}
        }
    }}


class IngestResponse(BaseModel):
    """Response after ingesting a document."""
    doc_id: str
    chunks_created: int
    status: str  # "success" | "empty_text" | "error"


# ═══════════════════════════════════════════════════════════════════════════
# QUERY
# ═══════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(5, ge=1, le=20, description="Number of sources to return")
    doc_id: Optional[str] = Field(None, description="Filter to specific document")

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What causes climate change?",
            "top_k": 5
        }
    }}


class Source(BaseModel):
    """A source chunk used in the answer."""
    index: int
    chunk_id: str
    doc_id: str
    text: str
    score: float
    title: Optional[str] = None
    char_start: int = 0
    char_end: int = 0


class QueryResponse(BaseModel):
    """Response with answer and citations."""
    answer: str
    sources: list[Source]
    has_answer: bool
    tokens_used: int
    retrieval_count: int
    rerank_count: int


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════

class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    doc_id: str
    title: Optional[str] = None
    chunk_count: int = 0
    created_at: Optional[str] = None


class DeleteResponse(BaseModel):
    """Response after deleting a document."""
    doc_id: str
    status: str


# ═══════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    """Response after uploading a file."""
    doc_id: str
    filename: str
    file_type: str
    chunks_created: int
    status: str  # "success" | "error"
    message: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"
