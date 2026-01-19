"""
Token-based text chunking with overlap for RAG retrieval.

Design choices:
- 1000 tokens/chunk: Fits comfortably in most LLM context windows while 
  providing enough semantic content for meaningful retrieval. Balances
  specificity (smaller = more precise) vs context (larger = more coherent).
- 120 token overlap (~12%): Prevents information loss at chunk boundaries.
  Key sentences often span boundaries; overlap ensures they appear in at
  least one chunk fully. 10-15% overlap is the sweet spot—less loses context,
  more wastes storage/compute.
- Token-based (not char): Aligns with LLM tokenization, giving predictable
  behavior when chunks are passed to embeddings and generation models.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


@dataclass
class Chunk:
    """A single chunk with metadata for citation tracking."""
    chunk_id: str
    doc_id: str
    text: str
    index: int              # Position in original document (0-based)
    total_chunks: int       # Total chunks from this document
    char_start: int         # Character offset in source
    char_end: int           # Character end offset
    token_count: int        # Actual token count
    created_at: str         # ISO timestamp
    metadata: dict          # User-provided metadata (title, source, etc.)

    def to_dict(self) -> dict:
        """Convert to dict for Qdrant payload storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "index": self.index,
            "total_chunks": self.total_chunks,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "created_at": self.created_at,
            **self.metadata
        }


class TokenChunker:
    """
    Token-aware text splitter using tiktoken for accurate counting
    and LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 120,
        model_name: str = "cl100k_base"  # GPT-4/3.5 tokenizer, works well generally
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(model_name)
        
        # LangChain splitter with token-based length function
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            separators=[
                "\n\n",      # Paragraph breaks (highest priority)
                "\n",        # Line breaks
                ". ",        # Sentence ends
                "? ",
                "! ",
                "; ",        # Clause breaks
                ", ",        # Phrase breaks
                " ",         # Words (last resort)
                ""           # Characters (fallback)
            ],
            keep_separator=True
        )

    def _token_length(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))

    def chunk_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> list[Chunk]:
        """
        Split text into overlapping chunks with full citation metadata.

        Args:
            text: Raw text to chunk
            doc_id: Document identifier (auto-generated if None)
            metadata: Additional metadata to attach to each chunk

        Returns:
            List of Chunk objects ready for embedding and storage
        """
        if not text or not text.strip():
            return []

        doc_id = doc_id or f"doc_{uuid4().hex[:12]}"
        metadata = metadata or {}
        now = datetime.now(timezone.utc).isoformat()

        # Split with LangChain
        splits = self.splitter.split_text(text)
        total_chunks = len(splits)

        # Track character positions for citation linking
        chunks = []
        search_start = 0

        for idx, split_text in enumerate(splits):
            # Find actual position in source document
            char_start = text.find(split_text, search_start)
            if char_start == -1:
                # Fallback: overlapping text may not match exactly
                char_start = search_start
            char_end = char_start + len(split_text)
            
            # Move search window (account for overlap)
            search_start = max(search_start, char_start + 1)

            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{idx:04d}",
                doc_id=doc_id,
                text=split_text,
                index=idx,
                total_chunks=total_chunks,
                char_start=char_start,
                char_end=char_end,
                token_count=self._token_length(split_text),
                created_at=now,
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks


# Module-level instance for simple imports
_default_chunker: Optional[TokenChunker] = None


def get_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 120
) -> TokenChunker:
    """Get or create a chunker instance."""
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = TokenChunker(chunk_size, chunk_overlap)
    return _default_chunker


def chunk_text(
    text: str,
    doc_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 120
) -> list[Chunk]:
    """
    Convenience function to chunk text with default settings.
    
    Example:
        chunks = chunk_text(
            "Your long document text here...",
            doc_id="my_doc",
            metadata={"title": "My Document", "source": "upload"}
        )
        for chunk in chunks:
            print(f"[{chunk.index+1}/{chunk.total_chunks}] {chunk.text[:50]}...")
    """
    chunker = get_chunker(chunk_size, chunk_overlap)
    return chunker.chunk_text(text, doc_id, metadata)
