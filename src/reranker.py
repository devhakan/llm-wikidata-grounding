"""
Reranker Module - Cross-Encoder for Statement Relevance

Scores (claim, statement) pairs for relevance using a Cross-Encoder model.
Statements with low relevance are filtered out before LLM classification.

Recommended Models:
- cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
- BAAI/bge-reranker-base (better quality)
- jinaai/jina-reranker-v1-base-en (best quality, commercial use allowed)

Author: LLM Wikidata Grounding Contributors
License: MIT
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Reranking will be disabled.")

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RELEVANCE_THRESHOLD = 0.3


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class RankedStatement:
    """A statement with its relevance score (0-1)."""
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

    Example:
        >>> reranker = Reranker()
        >>> ranked = reranker.rerank("Einstein discovered relativity", statements, top_k=2)
        >>> print(ranked[0].text)
        "Albert Einstein | notable work | theory of relativity"
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self._model = None

        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None and HAS_SENTENCE_TRANSFORMERS:
            logger.info("Loading reranker model: %s...", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded.")
        return self._model

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Score (claim, statement) pairs and normalize to 0-1 via sigmoid."""
        raw_scores = self.model.predict(pairs)
        return 1 / (1 + np.exp(-raw_scores))

    def _filter_and_sort(
        self,
        ranked: List[RankedStatement],
        threshold: Optional[float],
        top_k: Optional[int],
    ) -> List[RankedStatement]:
        """Sort by score descending, apply threshold and top_k limits."""
        effective_threshold = threshold if threshold is not None else self.threshold
        ranked.sort(key=lambda x: x.score, reverse=True)
        ranked = [r for r in ranked if r.score >= effective_threshold]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def rerank(
        self,
        claim: str,
        statements: List[str],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
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

        if not HAS_SENTENCE_TRANSFORMERS or self.model is None:
            logger.warning("Reranker unavailable, returning all statements unranked.")
            return [RankedStatement(text=s, score=1.0) for s in statements]

        pairs = [(claim, s) for s in statements]
        scores = self._score_pairs(pairs)

        ranked = [
            RankedStatement(text=s, score=float(sc))
            for s, sc in zip(statements, scores)
        ]

        return self._filter_and_sort(ranked, threshold, top_k)

    def rerank_with_metadata(
        self,
        claim: str,
        statements: List[Dict],
        text_key: str = "text",
        top_k: Optional[int] = None,
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
        scores = self._score_pairs(pairs)

        ranked = [
            RankedStatement(text=t, score=float(sc), original=st)
            for t, sc, st in zip(texts, scores, statements)
        ]

        return self._filter_and_sort(ranked, None, top_k)


# =============================================================================
# Convenience Functions
# =============================================================================

def filter_relevant_statements(
    claim: str,
    statements: List[str],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Filter statements to keep only those relevant to the claim.

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
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Reranker Module - Self Test")
    print("=" * 60)

    if not HAS_SENTENCE_TRANSFORMERS:
        print("\n✗ sentence-transformers not installed")
        print("  Install with: pip install sentence-transformers")
        exit(1)

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
