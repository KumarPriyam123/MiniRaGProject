"""
Text embedding using Cohere API (cloud-based, no local model needed).

Model: embed-english-light-v3.0
- Dimension: 384
- Speed: Fast API calls
- Quality: Good for semantic similarity
- Memory: ~0MB (API-based, no local model)

This avoids OOM issues on Render free tier (512MB limit).
"""

import os
from typing import Optional
import cohere

# Model config - using light model for 384 dimensions
MODEL_NAME = "embed-english-light-v3.0"
EMBEDDING_DIMENSION = 384

# Lazy-loaded client singleton
_client: Optional[cohere.Client] = None


def get_client() -> cohere.Client:
    """Get or create Cohere client."""
    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        _client = cohere.Client(api_key)
        print("✅ Cohere embedding client initialized")
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Cohere API.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        List of 384-dimensional float vectors
    """
    if not texts:
        return []
    
    client = get_client()
    
    # Cohere API call - much faster than local model
    response = client.embed(
        texts=texts,
        model=MODEL_NAME,
        input_type="search_document",  # For documents being stored
        truncate="END"
    )
    
    return response.embeddings


def embed_query(text: str) -> list[float]:
    """Embed a query string (uses different input_type for better search)."""
    client = get_client()
    response = client.embed(
        texts=[text],
        model=MODEL_NAME,
        input_type="search_query",  # For search queries
        truncate="END"
    )
    return response.embeddings[0]


def embed_text(text: str) -> list[float]:
    """Embed a single text string (document)."""
    return embed_texts([text])[0]
