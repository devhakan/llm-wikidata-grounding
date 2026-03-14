"""
Ollama-based Fact Classifier

Uses a local Ollama LLM to classify claims against Wikidata evidence.
Provides natural language reasoning for fact-checking verdicts.

Author: LLM Wikidata Grounding Contributors
License: MIT
"""

import logging
import os
import requests
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Verdict(Enum):
    """Possible verdicts for fact-checking."""
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


@dataclass
class ClassificationResult:
    """Result of classifying a claim."""
    verdict: Verdict
    confidence: float
    reasoning: str
    raw_response: str = ""


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = "qwen2.5:7b"

# Map confidence words to numeric values
_CONFIDENCE_MAP = {
    "HIGH": 0.9,
    "MEDIUM": 0.7,
    "LOW": 0.5,
}

FACT_CHECK_PROMPT = """You are a fact-checker. Your job is to determine if the given claim is supported by the evidence from Wikidata.

CLAIM: {claim}

EVIDENCE FROM WIKIDATA:
{evidence}

Based ONLY on the evidence above, classify the claim as:
- SUPPORTED: The evidence confirms the claim is true
- REFUTED: The evidence shows the claim is false  
- NOT_ENOUGH_INFO: The evidence doesn't clearly confirm or deny the claim

Respond in this exact format:
VERDICT: [SUPPORTED/REFUTED/NOT_ENOUGH_INFO]
CONFIDENCE: [high/medium/low]
REASONING: [one sentence explanation]"""


class OllamaClassifier:
    """
    Fact classifier using Ollama LLM.

    Advantages over NLI model:
    - Natural language understanding
    - Better reasoning with context
    - No model download required (uses existing Ollama)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ):
        self.model = model
        self.ollama_url = ollama_url

    def classify(
        self,
        claim: str,
        evidence: list[str],
    ) -> ClassificationResult:
        """
        Classify a claim against evidence.

        Args:
            claim: The claim to verify
            evidence: List of evidence statements from Wikidata

        Returns:
            ClassificationResult with verdict and reasoning
        """
        evidence_text = "\n".join(f"• {e}" for e in evidence[:10])

        if not evidence_text.strip():
            return ClassificationResult(
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.0,
                reasoning="No evidence found in Wikidata.",
            )

        prompt = FACT_CHECK_PROMPT.format(claim=claim, evidence=evidence_text)

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 200,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            result_text = response.json().get("response", "")
            return self._parse_response(result_text)

        except requests.exceptions.ConnectionError:
            return ClassificationResult(
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.0,
                reasoning="Could not connect to Ollama. Is it running?",
            )
        except Exception as e:
            logger.error("Ollama classification error: %s", e)
            return ClassificationResult(
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
            )

    def _parse_response(self, text: str) -> ClassificationResult:
        """Parse LLM response into structured result.
        
        Handles multiple response formats from various LLMs:
        - Exact format: VERDICT: SUPPORTED
        - Standalone keyword: SUPPORTED (anywhere in first line)
        - Reasoning-based: "...supports the claim..." → SUPPORTED
        """
        text_clean = text.strip().strip("[]\"'")
        text_upper = text_clean.upper()

        # Strategy 1: Look for explicit "VERDICT: X" format
        verdict = Verdict.NOT_ENOUGH_INFO
        if "VERDICT: SUPPORTED" in text_upper or "VERDICT:SUPPORTED" in text_upper:
            verdict = Verdict.SUPPORTED
        elif "VERDICT: REFUTED" in text_upper or "VERDICT:REFUTED" in text_upper:
            verdict = Verdict.REFUTED
        elif "VERDICT: NOT_ENOUGH_INFO" in text_upper or "VERDICT:NOT_ENOUGH_INFO" in text_upper:
            verdict = Verdict.NOT_ENOUGH_INFO

        # Strategy 2: Check for standalone keywords (first 200 chars)
        if verdict == Verdict.NOT_ENOUGH_INFO:
            head = text_upper[:200]
            if "SUPPORTED" in head and "NOT" not in head.split("SUPPORTED")[0][-5:]:
                verdict = Verdict.SUPPORTED
            elif "REFUTED" in head:
                verdict = Verdict.REFUTED

        # Strategy 3: Infer from reasoning text
        if verdict == Verdict.NOT_ENOUGH_INFO:
            lower = text_clean.lower()
            support_phrases = ["supports the claim", "directly supports", "confirms the claim",
                               "is supported", "evidence supports", "consistent with"]
            refute_phrases = ["contradicts the claim", "refutes the claim", "is refuted",
                              "evidence contradicts", "inconsistent with", "does not support"]
            if any(p in lower for p in support_phrases):
                verdict = Verdict.SUPPORTED
            elif any(p in lower for p in refute_phrases):
                verdict = Verdict.REFUTED

        # Extract confidence
        confidence: float = _CONFIDENCE_MAP.get("LOW", 0.5)
        if verdict != Verdict.NOT_ENOUGH_INFO:
            # If we found a clear verdict, default to HIGH confidence
            confidence = _CONFIDENCE_MAP.get("HIGH", 0.9)
        for level, score in _CONFIDENCE_MAP.items():
            if f"CONFIDENCE: {level}" in text_upper:
                confidence = score
                break

        # Extract reasoning
        reasoning = self._extract_reasoning(text_clean)

        return ClassificationResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning or "Based on Wikidata evidence.",
            raw_response=text,
        )

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        """Extract the reasoning sentence from LLM response."""
        marker = "REASONING:"
        idx = text.upper().find(marker)
        if idx != -1:
            reasoning = text[idx + len(marker):].strip()
            # Take first sentence only
            dot_idx = reasoning.find(".")
            if dot_idx != -1:
                reasoning = reasoning[:dot_idx + 1]
            return reasoning

        # Fallback: use the full text, stripping any verdict/confidence lines
        lines = text.strip().splitlines()
        reasoning_lines = []
        for line in lines:
            upper = line.strip().upper()
            if upper.startswith("VERDICT:") or upper.startswith("CONFIDENCE:"):
                continue
            if line.strip():
                reasoning_lines.append(line.strip())
        return " ".join(reasoning_lines)[:500] if reasoning_lines else ""

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.ok:
                models = [m["name"] for m in response.json().get("models", [])]
                return any(self.model in m for m in models)
            return False
        except Exception:
            return False


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Ollama Classifier...")
    print("=" * 60)

    classifier = OllamaClassifier()

    if not classifier.is_available():
        print(f"⚠ Ollama not available or model '{classifier.model}' not found")
        print(f"  Run: ollama pull {classifier.model}")
        exit(1)

    claim = "Aziz Sancar won the Nobel Prize in Chemistry in 2015"
    evidence = [
        "Aziz Sancar | award received | Nobel Prize in Chemistry",
        "Nobel Prize | point in time | 2015",
        "Aziz Sancar | together with | Tomas Lindahl, Paul L. Modrich",
    ]

    result = classifier.classify(claim, evidence)

    print(f"Claim: {claim}")
    print(f"Verdict: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Reasoning: {result.reasoning}")
