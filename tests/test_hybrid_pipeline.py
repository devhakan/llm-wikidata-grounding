"""
Tests for hybrid_pipeline module.

Tests cover FactCheckResult dataclass, pipeline edge cases,
and the verify() convenience function.
"""

import pytest
from unittest.mock import patch, MagicMock
from hybrid_pipeline import HybridFactChecker, FactCheckResult, verify
from ollama_classifier import Verdict, ClassificationResult
from reranker import RankedStatement


class TestFactCheckResult:
    """Tests for FactCheckResult dataclass."""
    
    def test_all_fields(self):
        result = FactCheckResult(
            claim="Einstein was born in Germany",
            verdict="SUPPORTED",
            confidence=0.9,
            reasoning="Evidence confirms birthplace",
            evidence=["Albert Einstein | place of birth | Ulm"],
            entities_found=["Q937"],
        )
        
        assert result.claim == "Einstein was born in Germany"
        assert result.verdict == "SUPPORTED"
        assert result.confidence == 0.9
        assert len(result.evidence) == 1
        assert len(result.entities_found) == 1


class TestHybridFactChecker:
    """Tests for HybridFactChecker pipeline."""
    
    def test_no_entities_returns_nei(self):
        """Returns NOT_ENOUGH_INFO when vector search finds nothing."""
        checker = HybridFactChecker()
        
        with patch("hybrid_pipeline.vector_search", return_value=[]):
            result = checker.check("completely unknown claim xyz")
        
        assert result.verdict == "NOT_ENOUGH_INFO"
        assert result.confidence == 0.0
        assert result.entities_found == []
    
    def test_no_statements_returns_nei(self):
        """Returns NOT_ENOUGH_INFO when entities found but no claims."""
        checker = HybridFactChecker()
        
        with patch("hybrid_pipeline.vector_search", return_value=[{"id": "Q1234"}]):
            with patch("hybrid_pipeline.get_entity_claims", return_value=[]):
                result = checker.check("some claim")
        
        assert result.verdict == "NOT_ENOUGH_INFO"
        assert "Q1234" in result.entities_found
    
    def test_full_pipeline_supported(self):
        """Full pipeline returns SUPPORTED for confirmed claim."""
        checker = HybridFactChecker(verbose=False)
        
        mock_entities = [{"id": "Q937"}, {"id": "Q3012"}]
        mock_claims = [
            {"text": "Albert Einstein | place of birth | Ulm"},
            {"text": "Albert Einstein | occupation | physicist"},
        ]
        mock_ranked = [
            RankedStatement(text="Albert Einstein | place of birth | Ulm", score=0.95),
        ]
        mock_classification = ClassificationResult(
            verdict=Verdict.SUPPORTED,
            confidence=0.9,
            reasoning="Evidence confirms birthplace."
        )
        
        with patch("hybrid_pipeline.vector_search", return_value=mock_entities):
            with patch("hybrid_pipeline.get_entity_claims", return_value=mock_claims):
                with patch.object(checker, "_reranker") as mock_reranker:
                    mock_reranker.rerank.return_value = mock_ranked
                    checker._reranker = mock_reranker
                    
                    with patch.object(checker, "_classifier") as mock_classifier:
                        mock_classifier.classify.return_value = mock_classification
                        checker._classifier = mock_classifier
                        
                        result = checker.check("Einstein was born in Ulm")
        
        assert result.verdict == "SUPPORTED"
        assert result.confidence == 0.9
    
    def test_lazy_model_loading(self):
        """Reranker and classifier are not loaded until first use."""
        checker = HybridFactChecker()
        
        assert checker._reranker is None
        assert checker._classifier is None


class TestVerifyFunction:
    """Tests for the verify() convenience function."""
    
    def test_returns_fact_check_result(self):
        """verify() returns a FactCheckResult."""
        with patch("hybrid_pipeline.vector_search", return_value=[]):
            result = verify("some claim")
        
        assert isinstance(result, FactCheckResult)
