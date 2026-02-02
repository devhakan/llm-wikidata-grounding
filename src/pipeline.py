"""
Fact Checking Pipeline

This module implements the complete fact-checking pipeline as described
in the Wikidata fact-checking workshop:

1. SEARCH: Find relevant Wikidata entities using vector/keyword search
2. RETRIEVE: Get claims (statements) for those entities
3. RERANK: Filter statements by relevance to the claim (Cross-Encoder)
4. CLASSIFY: Determine if evidence supports/contradicts claim (NLI)

The pipeline can verify claims like:
- "Einstein discovered relativity" → SUPPORTED
- "Leonardo da Vinci invented the light bulb" → REFUTED
- "Marie Curie was born in 1900" → REFUTED (actual: 1867)

Author: LLM Wikidata Grounding Contributors
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

# Handle both package and standalone imports
try:
    from .wikidata_api import (
        search_entities,
        vector_search,
        get_entity_claims,
        get_entity_statements_text,
    )
    from .reranker import Reranker, RankedStatement
    from .nli_classifier import NLIClassifier, Verdict, ClassificationResult
except ImportError:
    from wikidata_api import (
        search_entities,
        vector_search,
        get_entity_claims,
        get_entity_statements_text,
    )
    from reranker import Reranker, RankedStatement
    from nli_classifier import NLIClassifier, Verdict, ClassificationResult


# =============================================================================
# Configuration
# =============================================================================

# Maximum entities to retrieve per search
DEFAULT_ENTITY_LIMIT = 5

# Maximum statements to retrieve per entity
DEFAULT_STATEMENT_LIMIT = 100

# Maximum statements after reranking
DEFAULT_RERANK_TOP_K = 10

# Minimum relevance score for reranking
DEFAULT_RELEVANCE_THRESHOLD = 0.3


# =============================================================================
# Data Types
# =============================================================================

class VerificationResult(Enum):
    """Final verification verdict."""
    SUPPORTED = "SUPPORTED"       # Evidence confirms the claim
    REFUTED = "REFUTED"           # Evidence contradicts the claim
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"  # Insufficient evidence


@dataclass
class FactCheckResult:
    """
    Complete result of fact-checking a claim.
    
    Attributes:
        claim: The original claim that was verified
        verdict: Final verdict (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)
        confidence: Confidence score (0-1)
        evidence: List of relevant statements from Wikidata
        entities_found: Wikidata entities that were consulted
        explanation: Human-readable explanation of the verdict
        details: Detailed intermediate results for debugging
    """
    claim: str
    verdict: VerificationResult
    confidence: float
    evidence: List[str] = field(default_factory=list)
    entities_found: List[Dict[str, str]] = field(default_factory=list)
    explanation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        symbol = {
            VerificationResult.SUPPORTED: "✓",
            VerificationResult.REFUTED: "✗",
            VerificationResult.NOT_ENOUGH_INFO: "?"
        }[self.verdict]
        return f"{symbol} {self.verdict.value} ({self.confidence:.1%}): {self.claim[:50]}..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "claim": self.claim,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "entities_found": self.entities_found,
            "explanation": self.explanation,
        }


# =============================================================================
# Fact Checker Pipeline
# =============================================================================

class FactChecker:
    """
    Complete fact-checking pipeline using Wikidata.
    
    This class orchestrates the full fact-checking process:
    1. Search Wikidata for relevant entities
    2. Retrieve statements (claims) for those entities
    3. Rerank statements by relevance (using Cross-Encoder)
    4. Classify the relationship between evidence and claim (using NLI)
    
    Example:
        >>> checker = FactChecker()
        >>> result = checker.check("Einstein discovered relativity")
        >>> print(result.verdict)
        VerificationResult.SUPPORTED
        >>> print(result.evidence)
        ["Albert Einstein | notable work | theory of relativity"]
    """
    
    def __init__(
        self,
        use_vector_search: bool = True,
        reranker_model: Optional[str] = None,
        nli_model: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the fact checker.
        
        Args:
            use_vector_search: Use vector search if available (falls back to keyword)
            reranker_model: Custom Cross-Encoder model for reranking
            nli_model: Custom NLI model for classification
            verbose: Print progress information
        """
        self.use_vector_search = use_vector_search
        self.verbose = verbose
        
        # Initialize components (lazy loading)
        self._reranker = None
        self._classifier = None
        self._reranker_model = reranker_model
        self._nli_model = nli_model
    
    @property
    def reranker(self) -> Reranker:
        """Lazy load reranker."""
        if self._reranker is None:
            self._reranker = Reranker(
                model_name=self._reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        return self._reranker
    
    @property
    def classifier(self) -> NLIClassifier:
        """Lazy load NLI classifier."""
        if self._classifier is None:
            self._classifier = NLIClassifier(
                model_name=self._nli_model or "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
            )
        return self._classifier
    
    def _log(self, message: str):
        """Print if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def check(
        self,
        claim: str,
        entity_limit: int = DEFAULT_ENTITY_LIMIT,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K
    ) -> FactCheckResult:
        """
        Verify a claim against Wikidata.
        
        Args:
            claim: The claim to verify
            entity_limit: Maximum entities to search for
            rerank_top_k: Maximum relevant statements to keep
        
        Returns:
            FactCheckResult with verdict and evidence
        """
        details = {}
        
        # =====================================================================
        # Step 1: Search for relevant entities
        # =====================================================================
        self._log(f"\n[1/4] Searching for entities related to: '{claim}'")
        
        if self.use_vector_search:
            entities = vector_search(claim, limit=entity_limit)
        else:
            # Extract key terms and search
            entities = search_entities(claim, limit=entity_limit)
        
        if not entities:
            return FactCheckResult(
                claim=claim,
                verdict=VerificationResult.NOT_ENOUGH_INFO,
                confidence=0.0,
                explanation="No relevant entities found in Wikidata.",
                details={"step_failed": "entity_search"}
            )
        
        self._log(f"    Found {len(entities)} entities:")
        for e in entities[:3]:
            self._log(f"      - {e['id']}: {e.get('label', 'N/A')}")
        
        details["entities"] = entities
        
        # =====================================================================
        # Step 2: Retrieve statements for each entity
        # =====================================================================
        self._log(f"\n[2/4] Retrieving statements from Wikidata...")
        
        all_statements = []
        for entity in entities:
            entity_id = entity["id"]
            statements_text = get_entity_statements_text(entity_id)
            
            if statements_text:
                # Split into individual statements
                lines = statements_text.strip().split("\n") if isinstance(statements_text, str) else []
                all_statements.extend(lines[:DEFAULT_STATEMENT_LIMIT])
        
        if not all_statements:
            return FactCheckResult(
                claim=claim,
                verdict=VerificationResult.NOT_ENOUGH_INFO,
                confidence=0.0,
                entities_found=entities,
                explanation="Found entities but couldn't retrieve their statements.",
                details={"step_failed": "statement_retrieval", **details}
            )
        
        self._log(f"    Retrieved {len(all_statements)} total statements")
        details["total_statements"] = len(all_statements)
        
        # =====================================================================
        # Step 3: Rerank statements by relevance
        # =====================================================================
        self._log(f"\n[3/4] Reranking statements by relevance...")
        
        ranked = self.reranker.rerank(
            claim=claim,
            statements=all_statements,
            top_k=rerank_top_k
        )
        
        if not ranked:
            return FactCheckResult(
                claim=claim,
                verdict=VerificationResult.NOT_ENOUGH_INFO,
                confidence=0.0,
                entities_found=entities,
                explanation="No statements were relevant enough to the claim.",
                details={"step_failed": "reranking", **details}
            )
        
        relevant_statements = [r.text for r in ranked]
        self._log(f"    Kept {len(ranked)} relevant statements:")
        for r in ranked[:3]:
            self._log(f"      [{r.score:.2f}] {r.text[:60]}...")
        
        details["ranked_statements"] = [(r.text, r.score) for r in ranked]
        
        # =====================================================================
        # Step 4: Classify using NLI
        # =====================================================================
        self._log(f"\n[4/4] Classifying with NLI model...")
        
        # Combine top relevant statements as premise
        combined_evidence = " ".join(relevant_statements[:5])
        
        classification = self.classifier.classify(
            premise=combined_evidence,
            hypothesis=claim
        )
        
        self._log(f"    NLI Result: {classification.verdict.value}")
        self._log(f"    Confidence: {classification.confidence:.2%}")
        
        details["nli_result"] = {
            "verdict": classification.verdict.value,
            "confidence": classification.confidence,
            "scores": classification.scores
        }
        
        # =====================================================================
        # Convert NLI result to verification verdict
        # =====================================================================
        if classification.verdict == Verdict.ENTAILMENT:
            verdict = VerificationResult.SUPPORTED
            explanation = f"The claim is SUPPORTED by Wikidata evidence: {relevant_statements[0]}"
        elif classification.verdict == Verdict.CONTRADICTION:
            verdict = VerificationResult.REFUTED
            explanation = f"The claim is REFUTED by Wikidata evidence: {relevant_statements[0]}"
        else:
            verdict = VerificationResult.NOT_ENOUGH_INFO
            explanation = "The available evidence neither clearly supports nor refutes the claim."
        
        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            confidence=classification.confidence,
            evidence=relevant_statements,
            entities_found=entities,
            explanation=explanation,
            details=details
        )
    
    def check_batch(
        self,
        claims: List[str],
        **kwargs
    ) -> List[FactCheckResult]:
        """
        Verify multiple claims.
        
        Args:
            claims: List of claims to verify
            **kwargs: Arguments passed to check()
        
        Returns:
            List of FactCheckResult objects
        """
        results = []
        for i, claim in enumerate(claims, 1):
            self._log(f"\n{'='*60}")
            self._log(f"Checking claim {i}/{len(claims)}: {claim}")
            self._log("="*60)
            
            result = self.check(claim, **kwargs)
            results.append(result)
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def verify(claim: str, verbose: bool = False) -> FactCheckResult:
    """
    Quick verification of a single claim.
    
    Args:
        claim: The claim to verify
        verbose: Print progress
    
    Returns:
        FactCheckResult
    """
    checker = FactChecker(verbose=verbose)
    return checker.check(claim)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for fact checking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify claims against Wikidata"
    )
    parser.add_argument(
        "claim",
        nargs="?",
        help="Claim to verify (interactive mode if not provided)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Use keyword search instead of vector search"
    )
    
    args = parser.parse_args()
    
    checker = FactChecker(
        use_vector_search=not args.no_vector,
        verbose=args.verbose
    )
    
    if args.claim:
        # Single claim mode
        result = checker.check(args.claim)
        print(f"\n{result}")
        print(f"\nExplanation: {result.explanation}")
        if result.evidence:
            print(f"\nEvidence:")
            for e in result.evidence[:5]:
                print(f"  - {e}")
    else:
        # Interactive mode
        print("=" * 60)
        print("Wikidata Fact Checker - Interactive Mode")
        print("=" * 60)
        print("\nEnter claims to verify. Type 'quit' to exit.\n")
        
        while True:
            try:
                claim = input("Claim: ").strip()
                if claim.lower() in ("quit", "exit", "q"):
                    break
                if not claim:
                    continue
                
                result = checker.check(claim)
                print(f"\n{result}")
                print(f"Explanation: {result.explanation}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
