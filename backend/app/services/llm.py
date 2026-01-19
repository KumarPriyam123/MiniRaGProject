"""
LLM answering service with inline citations.

Uses Groq API (free tier):
- Model: llama-3.1-8b-instant
- Speed: ~200 tokens/sec
- Free: 30 RPM, 14,400 requests/day

Citation Strategy:
- Context chunks are numbered [1], [2], [3]...
- LLM is instructed to cite inline: "The answer is X [1][3]."
- If no relevant context, return explicit "I don't know"
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
from groq import Groq

from .retriever import RetrievedChunk


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise research assistant. Answer questions using ONLY the provided context.

STRICT RULES:
1. Use ONLY information from the numbered context chunks below
2. Cite sources with inline brackets: [1], [2], etc.
3. If multiple chunks support a point, cite all: [1][3]
4. If the context doesn't contain the answer, say: "I cannot answer this based on the provided documents."
5. Never make up information or use external knowledge
6. Keep answers concise but complete

CITATION FORMAT:
- Place citations immediately after the relevant statement
- Example: "Machine learning uses algorithms to learn from data [1]. Deep learning is a subset using neural networks [2][3]."
"""

CONTEXT_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

ANSWER (with inline citations):"""


# ═══════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Citation:
    """A single citation reference."""
    index: int              # Citation number (1-based)
    chunk_id: str
    doc_id: str
    text: str               # Source chunk text
    title: Optional[str]    # Document title if available
    
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title
        }


@dataclass
class AnswerResponse:
    """LLM response with answer and citations."""
    answer: str                     # Generated answer with [1][2] references
    citations: list[Citation]       # Source chunks for each citation
    tokens_used: int                # Total tokens (prompt + completion)
    model: str                      # Model used
    has_answer: bool                # False if "cannot answer" response
    
    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "tokens_used": self.tokens_used,
            "model": self.model,
            "has_answer": self.has_answer
        }


# ═══════════════════════════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════

_client: Optional[Groq] = None


def get_client() -> Groq:
    """Get or create Groq client."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 1024,
    temperature: float = 0.1  # Low for factual accuracy
) -> AnswerResponse:
    """
    Generate an answer with inline citations from retrieved chunks.
    
    Args:
        question: User's question
        chunks: Reranked chunks to use as context
        model: Groq model name
        max_tokens: Maximum response length
        temperature: Sampling temperature (lower = more deterministic)
        
    Returns:
        AnswerResponse with answer, citations, and metadata
    """
    if not chunks:
        return AnswerResponse(
            answer="I cannot answer this question as no relevant documents were found.",
            citations=[],
            tokens_used=0,
            model=model,
            has_answer=False
        )
    
    # Format context with numbered chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.metadata.get("title", "")
        title_str = f" (from: {title})" if title else ""
        context_parts.append(f"[{i}]{title_str}: {chunk.text}")
    
    context = "\n\n".join(context_parts)
    
    # Build prompt
    user_prompt = CONTEXT_TEMPLATE.format(
        context=context,
        question=question
    )
    
    # Call Groq API
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    answer = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens
    
    # Build citation objects
    citations = []
    for i, chunk in enumerate(chunks, 1):
        citations.append(Citation(
            index=i,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            title=chunk.metadata.get("title")
        ))
    
    # Detect "no answer" responses
    no_answer_phrases = [
        "cannot answer",
        "don't have enough information",
        "not mentioned in",
        "no information about",
        "doesn't contain",
        "does not contain"
    ]
    has_answer = not any(phrase in answer.lower() for phrase in no_answer_phrases)
    
    return AnswerResponse(
        answer=answer,
        citations=citations,
        tokens_used=tokens_used,
        model=model,
        has_answer=has_answer
    )


def answer_question(
    question: str,
    retrieve_k: int = 20,
    rerank_k: int = 5,
    doc_id: Optional[str] = None
) -> AnswerResponse:
    """
    Full RAG pipeline: retrieve → rerank → generate answer.
    
    This is the main entry point for answering questions.
    
    Args:
        question: User's natural language question
        retrieve_k: Chunks to fetch from vector DB
        rerank_k: Chunks to keep after reranking
        doc_id: Optional filter to specific document
        
    Returns:
        AnswerResponse with cited answer
    """
    from .reranker import retrieve_and_rerank
    
    # Retrieve and rerank
    chunks = retrieve_and_rerank(
        query=question,
        retrieve_k=retrieve_k,
        rerank_k=rerank_k,
        doc_id=doc_id
    )
    
    # Generate answer
    return generate_answer(question, chunks)
