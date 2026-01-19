"""
Text embedding using sentence-transformers.

Model: all-MiniLM-L6-v2
- Dimension: 384
- Speed: ~14K sentences/sec on CPU
- Quality: Strong for semantic similarity tasks
- Size: 80MB (fast cold start)

NOTE: Model is loaded LAZILY on first call to avoid OOM on Render free tier.
"""

from typing import TYPE_CHECKING

# Avoid importing heavy libraries at module load time
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# Model config
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Fixed for this model

# Lazy-loaded model singleton
_model = None


def get_model() -> "SentenceTransformer":
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        List of 384-dimensional float vectors
    """
    if not texts:
        return []
    
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
        show_progress_bar=False
    )
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    return embed_texts([text])[0]
