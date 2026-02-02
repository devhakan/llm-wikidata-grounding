"""
LLM Wikidata Grounding - Source Package

A fact-checking system that grounds claims against Wikidata's
structured knowledge base using a hybrid pipeline:

1. Vector Search - Find relevant entities semantically
2. Statement Retrieval - Get Wikidata claims
3. Reranking - Filter by relevance (Cross-Encoder)
4. Ollama LLM - Natural language reasoning for verdict

Usage:
    from src import HybridFactChecker, verify
    
    # Quick verification
    result = verify("Einstein discovered relativity")
    print(result.verdict)  # SUPPORTED
    
    # Full control
    checker = HybridFactChecker(model="qwen2.5:7b", verbose=True)
    result = checker.check("Marie Curie won two Nobel Prizes")
"""

from .wikidata_api import (
    search_entities,
    vector_search,
    get_entity_claims,
    execute_sparql,
)

from .reranker import (
    Reranker,
    RankedStatement,
    filter_relevant_statements,
)

from .ollama_classifier import (
    OllamaClassifier,
    Verdict,
    ClassificationResult,
)

from .hybrid_pipeline import (
    HybridFactChecker,
    FactCheckResult,
    verify,
)

__version__ = "0.3.0"
__all__ = [
    # Wikidata API
    "search_entities",
    "vector_search",
    "get_entity_claims",
    "execute_sparql",
    # Reranker
    "Reranker",
    "RankedStatement",
    "filter_relevant_statements",
    # Ollama Classifier
    "OllamaClassifier",
    "Verdict",
    "ClassificationResult",
    # Hybrid Pipeline
    "HybridFactChecker",
    "FactCheckResult",
    "verify",
]
