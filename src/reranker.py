"""
Reranker Module - Cross-Encoder for Statement Relevance

This module implements the reranking step of the fact-checking pipeline.
Given a claim and a set of Wikidata statements, it scores each statement
for relevance to the claim using a Cross-Encoder model.

The Cross-Encoder takes (query, document) pairs and produces a relevance score,
which is more accurate than bi-encoder similarity for reranking tasks.

Recommended Models:
- cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
- BAAI/bge-reranker-base (better quality)
- jinaai/jina-reranker-v1-base-en (best quality, commercial use allowed)

Author: LLM Wikidata Grounding Contributors
License: MIT
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from sentence_transformers import CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Reranking will be disabled.")


# =============================================================================
# Configuration
# =============================================================================

# Default Cross-Encoder model
# Good balance of speed and quality
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Threshold for relevance score (statements below this are filtered out)
DEFAULT_RELEVANCE_THRESHOLD = 0.3


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class RankedStatement:
    """
    A statement with its relevance score.
    
    Attributes:
        text: The statement text (e.g., "Albert Einstein | occupation | physicist")
        score: Relevance score from 0 to 1
        original: Original statement data (if available)
    """
    text: str
    score: float
    original: Optional[Dict] = None
    
    def __repr__(self):
        return f"RankedStatement(score={self.score:.3f}, text='{self.text[:50]}...')"


# =============================================================================
# Reranker Class
# =============================================================================

class Reranker:
    """
    Cross-Encoder based reranker for filtering relevant statements.
    
    This class loads a Cross-Encoder model and uses it to score
    (claim, statement) pairs for relevance. Statements with low
    relevance scores can be filtered out before NLI classification.
    
    Example:
        >>> reranker = Reranker()
        >>> claim = "Einstein discovered relativity"
        >>> statements = [
        ...     "Albert Einstein | occupation | physicist",
        ...     "Albert Einstein | notable work | theory of relativity",
        ...     "Albert Einstein | birth place | Ulm",
        ...     "Albert Einstein | spouse | Elsa Einstein"
        ... ]
        >>> ranked = reranker.rerank(claim, statements, top_k=2)
        >>> print(ranked[0].text)
        "Albert Einstein | notable work | theory of relativity"
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for Cross-Encoder
            threshold: Minimum relevance score to keep (0-1)
        """
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Warning: sentence-transformers not installed. Install with:")
            print("  pip install sentence-transformers")
    
    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None and HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading reranker model: {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
            print("Reranker model loaded.")
        return self._model
    
    def rerank(
        self,
        claim: str,
        statements: List[str],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[RankedStatement]:
        """
        Rerank statements by relevance to the claim.
        
        Args:
            claim: The claim to verify
            statements: List of statement texts to rank
            top_k: Keep only top K statements (None = keep all above threshold)
            threshold: Override default threshold
        
        Returns:
            List of RankedStatement sorted by score descending
        """
        if not statements:
            return []
        
        threshold = threshold if threshold is not None else self.threshold
        
        if not HAS_SENTENCE_TRANSFORMERS or self.model is None:
            # Fallback: return all statements with score 1.0
            print("Warning: Reranker unavailable, returning all statements")
            return [RankedStatement(text=s, score=1.0) for s in statements]
        
        # Create (claim, statement) pairs for Cross-Encoder
        pairs = [(claim, statement) for statement in statements]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Normalize scores to 0-1 range (Cross-Encoder outputs can vary)
        # Using sigmoid-like normalization
        import numpy as np
        scores = 1 / (1 + np.exp(-scores))
        
        # Create ranked statements
        ranked = [
            RankedStatement(text=statement, score=float(score))
            for statement, score in zip(statements, scores)
        ]
        
        # Sort by score descending
        ranked.sort(key=lambda x: x.score, reverse=True)
        
        # Apply threshold filter
        ranked = [r for r in ranked if r.score >= threshold]
        
        # Apply top_k limit
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    def rerank_with_metadata(
        self,
        claim: str,
        statements: List[Dict],
        text_key: str = "text",
        top_k: Optional[int] = None
    ) -> List[RankedStatement]:
        """
        Rerank statement dictionaries, preserving metadata.
        
        Args:
            claim: The claim to verify
            statements: List of statement dictionaries
            text_key: Key for the text field in each dictionary
            top_k: Keep only top K statements
        
        Returns:
            List of RankedStatement with original dict preserved
        """
        if not statements:
            return []
        
        texts = [s.get(text_key, str(s)) for s in statements]
        
        if not HAS_SENTENCE_TRANSFORMERS or self.model is None:
            return [
                RankedStatement(text=t, score=1.0, original=s)
                for t, s in zip(texts, statements)
            ]
        
        pairs = [(claim, t) for t in texts]
        scores = self.model.predict(pairs)
        
        import numpy as np
        scores = 1 / (1 + np.exp(-scores))
        
        ranked = [
            RankedStatement(text=t, score=float(sc), original=st)
            for t, sc, st in zip(texts, scores, statements)
        ]
        
        ranked.sort(key=lambda x: x.score, reverse=True)
        ranked = [r for r in ranked if r.score >= self.threshold]
        
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked


# =============================================================================
# Convenience Functions
# =============================================================================

def filter_relevant_statements(
    claim: str,
    statements: List[str],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Filter statements to keep only those relevant to the claim.
    
    This is a convenience function for quick reranking without
    managing a Reranker instance.
    
    Args:
        claim: The claim to verify
        statements: List of statement texts
        threshold: Minimum relevance score
        top_k: Maximum statements to return
    
    Returns:
        List of (statement, score) tuples
    """
    reranker = Reranker(threshold=threshold)
    ranked = reranker.rerank(claim, statements, top_k=top_k)
    return [(r.text, r.score) for r in ranked]


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reranker Module - Self Test")
    print("=" * 60)
    
    if not HAS_SENTENCE_TRANSFORMERS:
        print("\n✗ sentence-transformers not installed")
        print("  Install with: pip install sentence-transformers")
        exit(1)
    
    # Test reranking
    claim = "Albert Einstein discovered the theory of relativity"
    
    statements = [
        "Albert Einstein | occupation | physicist",
        "Albert Einstein | notable work | theory of relativity",
        "Albert Einstein | birth place | Ulm",
        "Albert Einstein | spouse | Elsa Einstein",
        "Albert Einstein | citizenship | Switzerland",
        "Albert Einstein | award | Nobel Prize in Physics",
        "Albert Einstein | field of work | theoretical physics",
    ]
    
    print(f"\nClaim: {claim}")
    print(f"\nReranking {len(statements)} statements...\n")
    
    reranker = Reranker()
    ranked = reranker.rerank(claim, statements, top_k=5)
    
    print("Top 5 most relevant statements:")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. [{r.score:.3f}] {r.text}")
    
    print("\n" + "=" * 60)
    print("Self-test complete!")
    print("=" * 60)
