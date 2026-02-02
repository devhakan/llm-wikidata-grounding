"""
Basic Fact Checking Example

This example demonstrates the complete fact-checking pipeline:
1. Search Wikidata for relevant entities
2. Retrieve statements (claims) for those entities
3. Rerank statements by relevance (Cross-Encoder)
4. Classify using NLI to determine if evidence supports/refutes claim

Usage:
    python examples/basic_example.py
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import FactChecker, VerificationResult


def main():
    print("=" * 70)
    print("LLM Wikidata Grounding - Basic Example")
    print("=" * 70)
    
    # Initialize the fact checker with verbose output
    print("\nInitializing fact checker (this may take a moment to load models)...\n")
    checker = FactChecker(verbose=True)
    
    # Test claims
    test_claims = [
        # Should be SUPPORTED
        ("Albert Einstein discovered the theory of relativity", "SUPPORTED"),
        
        # Should be REFUTED
        ("Leonardo da Vinci invented the light bulb", "REFUTED"),
        
        # Should be SUPPORTED (multi-fact)
        ("Marie Curie won two Nobel Prizes", "SUPPORTED"),
    ]
    
    print("\n" + "=" * 70)
    print("Running fact checks...")
    print("=" * 70)
    
    results = []
    for claim, expected in test_claims:
        print(f"\n{'─' * 70}")
        print(f"CLAIM: {claim}")
        print(f"EXPECTED: {expected}")
        print("─" * 70)
        
        result = checker.check(claim)
        results.append((result, expected))
        
        # Display result
        symbol = "✓" if result.verdict == VerificationResult.SUPPORTED else (
            "✗" if result.verdict == VerificationResult.REFUTED else "?"
        )
        print(f"\nRESULT: {symbol} {result.verdict.value}")
        print(f"CONFIDENCE: {result.confidence:.1%}")
        print(f"EXPLANATION: {result.explanation}")
        
        if result.evidence:
            print("\nEVIDENCE (top 3):")
            for e in result.evidence[:3]:
                print(f"  • {e[:80]}...")
        
        # Check if matched expected
        match = "✓ PASS" if result.verdict.value == expected else "✗ FAIL"
        print(f"\nTEST: {match}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r, e in results if r.verdict.value == e)
    print(f"\nPassed: {passed}/{len(results)}")
    
    for (result, expected) in results:
        match = "✓" if result.verdict.value == expected else "✗"
        print(f"  {match} {result.claim[:50]}... → {result.verdict.value}")


if __name__ == "__main__":
    main()
