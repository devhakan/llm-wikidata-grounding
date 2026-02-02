"""
LLM Wikidata Grounding - Source Package

A fact-checking system that grounds claims against Wikidata's
structured knowledge base using a pipeline approach:

1. Vector/Keyword Search - Find relevant entities
2. Statement Retrieval - Get Wikidata claims
3. Reranking - Filter by relevance (Cross-Encoder)
4. NLI Classification - Determine support/contradiction

Usage:
    from src import FactChecker, verify
    
    # Quick verification
    result = verify("Einstein discovered relativity")
    print(result.verdict)  # SUPPORTED
    
    # Full control
    checker = FactChecker(verbose=True)
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

from .nli_classifier import (
    NLIClassifier,
    Verdict,
    ClassificationResult,
    verify_claim,
)

from .pipeline import (
    FactChecker,
    FactCheckResult,
    VerificationResult,
    verify,
)

__version__ = "0.2.0"
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
    # NLI
    "NLIClassifier",
    "Verdict",
    "ClassificationResult",
    "verify_claim",
    # Pipeline
    "FactChecker",
    "FactCheckResult",
    "VerificationResult",
    "verify",
]
