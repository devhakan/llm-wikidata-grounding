"""
NLI Classifier Module - Natural Language Inference for Fact Verification

This module implements the classification step of the fact-checking pipeline.
Given a premise (evidence from Wikidata) and a hypothesis (the claim),
it determines the logical relationship between them:

- ENTAILMENT: The premise supports/proves the hypothesis
- CONTRADICTION: The premise contradicts the hypothesis  
- NEUTRAL: The premise neither supports nor contradicts the hypothesis

This is based on the Natural Language Inference (NLI) task from NLP research.

Recommended Models:
- MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual, good quality)
- facebook/bart-large-mnli (English, high quality)
- microsoft/deberta-v3-large-mnli (English, best quality but large)

Author: LLM Wikidata Grounding Contributors
License: MIT
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. NLI classification will be disabled.")


# =============================================================================
# Configuration
# =============================================================================

# Default NLI model - good balance of quality and speed
# Supports multiple languages
DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Alternative models (English only but higher quality)
ALTERNATIVE_MODELS = {
    "fast": "typeform/distilbert-base-uncased-mnli",
    "balanced": "facebook/bart-large-mnli",
    "best": "microsoft/deberta-v3-large-mnli"
}


# =============================================================================
# Data Types
# =============================================================================

class Verdict(Enum):
    """Classification result from NLI."""
    ENTAILMENT = "entailment"      # Evidence supports the claim
    CONTRADICTION = "contradiction" # Evidence contradicts the claim
    NEUTRAL = "neutral"             # Evidence is unrelated to the claim


@dataclass
class ClassificationResult:
    """
    Result of NLI classification.
    
    Attributes:
        verdict: The classification (ENTAILMENT, CONTRADICTION, NEUTRAL)
        confidence: Confidence score for the verdict (0-1)
        scores: All class scores
        premise: The evidence used
        hypothesis: The claim being verified
    """
    verdict: Verdict
    confidence: float
    scores: Dict[str, float]
    premise: str
    hypothesis: str
    
    def __repr__(self):
        return f"ClassificationResult({self.verdict.value}, confidence={self.confidence:.3f})"
    
    @property
    def is_supported(self) -> bool:
        """Returns True if evidence supports the claim."""
        return self.verdict == Verdict.ENTAILMENT
    
    @property
    def is_refuted(self) -> bool:
        """Returns True if evidence contradicts the claim."""
        return self.verdict == Verdict.CONTRADICTION


# =============================================================================
# NLI Classifier Class
# =============================================================================

class NLIClassifier:
    """
    Natural Language Inference classifier for fact verification.
    
    This class uses a pre-trained NLI model to determine whether
    evidence (premise) supports, contradicts, or is neutral to
    a claim (hypothesis).
    
    Example:
        >>> classifier = NLIClassifier()
        >>> result = classifier.classify(
        ...     premise="Albert Einstein developed the theory of relativity.",
        ...     hypothesis="Einstein discovered relativity."
        ... )
        >>> print(result.verdict)
        Verdict.ENTAILMENT
        >>> print(result.confidence)
        0.95
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: Optional[str] = None
    ):
        """
        Initialize the NLI classifier.
        
        Args:
            model_name: HuggingFace model name for NLI
            device: Device to run on ("cpu", "cuda", "mps" for Apple Silicon)
                   Auto-detected if None.
        """
        self.model_name = model_name
        self._pipeline = None
        self._device = device
        
        if not HAS_TRANSFORMERS:
            print("Warning: transformers not installed. Install with:")
            print("  pip install transformers torch")
    
    @property
    def device(self) -> str:
        """Get the device to use for inference."""
        if self._device:
            return self._device
        
        if HAS_TRANSFORMERS and torch.cuda.is_available():
            return "cuda"
        elif HAS_TRANSFORMERS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    @property
    def pipeline(self):
        """Lazy load the model on first use."""
        if self._pipeline is None and HAS_TRANSFORMERS:
            print(f"Loading NLI model: {self.model_name}...")
            print(f"Device: {self.device}")
            
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device if self.device != "cpu" else -1
            )
            print("NLI model loaded.")
        return self._pipeline
    
    def classify(
        self,
        premise: str,
        hypothesis: str
    ) -> ClassificationResult:
        """
        Classify the relationship between premise and hypothesis.
        
        Args:
            premise: The evidence/fact (from Wikidata)
            hypothesis: The claim to verify
        
        Returns:
            ClassificationResult with verdict and confidence
        """
        if not HAS_TRANSFORMERS or self.pipeline is None:
            # Fallback: return neutral with low confidence
            return ClassificationResult(
                verdict=Verdict.NEUTRAL,
                confidence=0.0,
                scores={"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34},
                premise=premise,
                hypothesis=hypothesis
            )
        
        # Use zero-shot classification with NLI labels
        result = self.pipeline(
            hypothesis,  # The claim
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template=f"Based on: {premise}. This claim is {{}}"
        )
        
        # Parse results
        labels = result["labels"]
        scores = result["scores"]
        
        score_dict = dict(zip(labels, scores))
        top_label = labels[0]
        top_score = scores[0]
        
        verdict = Verdict(top_label)
        
        return ClassificationResult(
            verdict=verdict,
            confidence=top_score,
            scores=score_dict,
            premise=premise,
            hypothesis=hypothesis
        )
    
    def classify_with_multiple_premises(
        self,
        premises: List[str],
        hypothesis: str,
        aggregation: str = "max"
    ) -> ClassificationResult:
        """
        Classify using multiple pieces of evidence.
        
        When multiple Wikidata statements are relevant, we need to
        aggregate their evidence. This method supports different
        aggregation strategies.
        
        Args:
            premises: List of evidence statements
            hypothesis: The claim to verify
            aggregation: How to combine results ("max", "mean", "vote")
        
        Returns:
            Aggregated ClassificationResult
        """
        if not premises:
            return ClassificationResult(
                verdict=Verdict.NEUTRAL,
                confidence=0.0,
                scores={"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0},
                premise="(no evidence)",
                hypothesis=hypothesis
            )
        
        # Classify each premise
        results = [self.classify(p, hypothesis) for p in premises]
        
        if aggregation == "max":
            # Return result with highest confidence
            best = max(results, key=lambda r: r.confidence)
            return best
        
        elif aggregation == "mean":
            # Average scores across all premises
            avg_scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            for r in results:
                for k, v in r.scores.items():
                    avg_scores[k] += v / len(results)
            
            top_label = max(avg_scores, key=avg_scores.get)
            
            return ClassificationResult(
                verdict=Verdict(top_label),
                confidence=avg_scores[top_label],
                scores=avg_scores,
                premise=" | ".join(premises[:3]) + ("..." if len(premises) > 3 else ""),
                hypothesis=hypothesis
            )
        
        elif aggregation == "vote":
            # Majority voting
            votes = {"entailment": 0, "contradiction": 0, "neutral": 0}
            for r in results:
                votes[r.verdict.value] += 1
            
            winner = max(votes, key=votes.get)
            confidence = votes[winner] / len(results)
            
            return ClassificationResult(
                verdict=Verdict(winner),
                confidence=confidence,
                scores={k: v / len(results) for k, v in votes.items()},
                premise=f"({len(premises)} statements)",
                hypothesis=hypothesis
            )
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")


# =============================================================================
# Convenience Functions
# =============================================================================

def verify_claim(
    evidence: str,
    claim: str,
    model: str = DEFAULT_NLI_MODEL
) -> Tuple[str, float]:
    """
    Quick verification of a claim against evidence.
    
    Args:
        evidence: The supporting/refuting evidence (premise)
        claim: The claim to verify (hypothesis)
        model: NLI model to use
    
    Returns:
        Tuple of (verdict_string, confidence)
    """
    classifier = NLIClassifier(model_name=model)
    result = classifier.classify(evidence, claim)
    return result.verdict.value, result.confidence


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NLI Classifier Module - Self Test")
    print("=" * 60)
    
    if not HAS_TRANSFORMERS:
        print("\n✗ transformers not installed")
        print("  Install with: pip install transformers torch")
        exit(1)
    
    classifier = NLIClassifier()
    
    # Test cases
    test_cases = [
        {
            "premise": "Albert Einstein developed the theory of relativity.",
            "hypothesis": "Einstein discovered relativity.",
            "expected": "entailment"
        },
        {
            "premise": "The light bulb was invented by Thomas Edison.",
            "hypothesis": "Leonardo da Vinci invented the light bulb.",
            "expected": "contradiction"
        },
        {
            "premise": "Marie Curie was born in Warsaw, Poland.",
            "hypothesis": "Marie Curie won two Nobel Prizes.",
            "expected": "neutral"
        }
    ]
    
    print("\nRunning test cases:\n")
    
    for i, tc in enumerate(test_cases, 1):
        result = classifier.classify(tc["premise"], tc["hypothesis"])
        
        status = "✓" if result.verdict.value == tc["expected"] else "✗"
        print(f"{status} Test {i}: {result.verdict.value} (expected: {tc['expected']})")
        print(f"   Premise: {tc['premise'][:50]}...")
        print(f"   Hypothesis: {tc['hypothesis']}")
        print(f"   Confidence: {result.confidence:.3f}")
        print()
    
    print("=" * 60)
    print("Self-test complete!")
    print("=" * 60)
