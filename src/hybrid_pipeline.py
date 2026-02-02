#!/usr/bin/env python3
"""
Hybrid Fact-Checking Pipeline

Combines:
1. Wikidata Vector Search - Find relevant entities
2. Cross-Encoder Reranking - Filter relevant statements  
3. Ollama LLM - Natural language reasoning for verdict

Based on Philippe Saade's workshop approach, adapted for local LLM.
"""

import sys
import os

# Handle imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wikidata_api import vector_search, get_entity_claims
from reranker import Reranker
from ollama_classifier import OllamaClassifier, Verdict, ClassificationResult

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FactCheckResult:
    """Complete result of fact-checking."""
    claim: str
    verdict: str
    confidence: float
    reasoning: str
    evidence: List[str]
    entities_found: List[str]


class HybridFactChecker:
    """
    Hybrid fact-checker using Vector Search + Reranker + Ollama.
    
    Pipeline:
        Claim → Vector Search → Get Claims → Rerank → Ollama → Verdict
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        verbose: bool = False
    ):
        self.model = model
        self.verbose = verbose
        self._reranker = None
        self._classifier = None
        self.reranker_model = reranker_model
    
    @property
    def reranker(self):
        if self._reranker is None:
            if self.verbose:
                print(f"Loading reranker: {self.reranker_model}")
            self._reranker = Reranker(model_name=self.reranker_model)
        return self._reranker
    
    @property
    def classifier(self):
        if self._classifier is None:
            if self.verbose:
                print(f"Using Ollama model: {self.model}")
            self._classifier = OllamaClassifier(model=self.model)
        return self._classifier
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def check(self, claim: str, top_k: int = 10) -> FactCheckResult:
        """
        Verify a claim against Wikidata.
        
        Args:
            claim: The claim to verify
            top_k: Number of relevant statements to keep
            
        Returns:
            FactCheckResult with verdict and evidence
        """
        # Step 1: Vector Search
        self._log(f"\n[1/4] Vector Search: '{claim[:50]}...'")
        
        entities = vector_search(claim, limit=20)
        
        if not entities:
            return FactCheckResult(
                claim=claim,
                verdict="NOT_ENOUGH_INFO",
                confidence=0.0,
                reasoning="No relevant entities found.",
                evidence=[],
                entities_found=[]
            )
        
        entity_ids = [e["id"] for e in entities if e.get("id")][:10]
        self._log(f"    Found {len(entity_ids)} entities: {entity_ids[:3]}")
        
        # Step 2: Get Claims from Wikidata
        self._log(f"\n[2/4] Fetching Wikidata claims...")
        
        all_statements = []
        for eid in entity_ids[:5]:  # Limit to top 5 entities
            claims_data = get_entity_claims(eid)
            for stmt in claims_data:
                text = stmt.get("text", str(stmt))
                if text:
                    all_statements.append(text)
        
        self._log(f"    Retrieved {len(all_statements)} statements")
        
        if not all_statements:
            return FactCheckResult(
                claim=claim,
                verdict="NOT_ENOUGH_INFO",
                confidence=0.0,
                reasoning="No statements found for entities.",
                evidence=[],
                entities_found=entity_ids
            )
        
        # Step 3: Rerank
        self._log(f"\n[3/4] Reranking statements...")
        
        ranked = self.reranker.rerank(
            claim=claim,
            statements=all_statements,
            top_k=top_k
        )
        
        relevant_evidence = [r.text for r in ranked]
        self._log(f"    Kept {len(relevant_evidence)} relevant statements")
        
        if not relevant_evidence:
            return FactCheckResult(
                claim=claim,
                verdict="NOT_ENOUGH_INFO",
                confidence=0.0,
                reasoning="No relevant evidence after filtering.",
                evidence=[],
                entities_found=entity_ids
            )
        
        # Step 4: Classify with Ollama
        self._log(f"\n[4/4] Classifying with {self.model}...")
        
        result = self.classifier.classify(claim, relevant_evidence)
        
        self._log(f"    Verdict: {result.verdict.value}")
        self._log(f"    Confidence: {result.confidence:.0%}")
        
        return FactCheckResult(
            claim=claim,
            verdict=result.verdict.value,
            confidence=result.confidence,
            reasoning=result.reasoning,
            evidence=relevant_evidence[:5],
            entities_found=entity_ids
        )


def verify(claim: str, verbose: bool = False) -> FactCheckResult:
    """Quick helper to verify a single claim."""
    checker = HybridFactChecker(verbose=verbose)
    return checker.check(claim)


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fact-check a claim against Wikidata")
    parser.add_argument("claim", nargs="?", help="Claim to verify")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-m", "--model", default="qwen2.5:7b", help="Ollama model")
    
    args = parser.parse_args()
    
    if not args.claim:
        # Interactive mode
        print("Hybrid Fact-Checker (Vector Search + Reranker + Ollama)")
        print("=" * 60)
        print("Enter claims to verify (Ctrl+C to exit)\n")
        
        checker = HybridFactChecker(model=args.model, verbose=True)
        
        while True:
            try:
                claim = input("\nClaim: ").strip()
                if not claim:
                    continue
                
                result = checker.check(claim)
                
                symbol = {"SUPPORTED": "✓", "REFUTED": "✗"}.get(result.verdict, "?")
                print(f"\n{symbol} {result.verdict} ({result.confidence:.0%})")
                print(f"Reasoning: {result.reasoning}")
                
            except KeyboardInterrupt:
                print("\nBye!")
                break
    else:
        # Single claim mode
        checker = HybridFactChecker(model=args.model, verbose=args.verbose)
        result = checker.check(args.claim)
        
        symbol = {"SUPPORTED": "✓", "REFUTED": "✗"}.get(result.verdict, "?")
        print(f"{symbol} {result.verdict} ({result.confidence:.0%})")
        print(f"Reasoning: {result.reasoning}")
        
        if args.verbose and result.evidence:
            print("\nEvidence:")
            for e in result.evidence[:3]:
                print(f"  • {e[:80]}")
