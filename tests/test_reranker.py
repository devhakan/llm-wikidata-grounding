"""
Tests for reranker module.

Tests cover reranking behavior: sorting, top_k, threshold filtering,
empty input handling, and fallback when sentence-transformers is unavailable.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
from reranker import Reranker, RankedStatement, filter_relevant_statements


class TestReranker:
    """Tests for Reranker class."""
    
    def test_rerank_empty_input(self):
        """Returns empty list for empty input."""
        reranker = Reranker()
        result = reranker.rerank("some claim", [])
        assert result == []
    
    def test_rerank_returns_ranked_statements(self, sample_statements_for_reranking):
        """Returns RankedStatement objects sorted by score."""
        reranker = Reranker()
        
        # Mock the model to return predictable scores
        mock_model = MagicMock()
        # Higher score for "theory of relativity", lower for "spouse"
        scores = np.array([3.0, 5.0, 1.0, -2.0, 0.5, 2.0, 4.0])
        mock_model.predict.return_value = scores
        reranker._model = mock_model
        
        claim = "Einstein discovered the theory of relativity"
        ranked = reranker.rerank(claim, sample_statements_for_reranking)
        
        assert len(ranked) > 0
        assert all(isinstance(r, RankedStatement) for r in ranked)
        
        # Verify sorted in descending order
        scores_out = [r.score for r in ranked]
        assert scores_out == sorted(scores_out, reverse=True)
    
    def test_rerank_top_k(self, sample_statements_for_reranking):
        """top_k parameter limits number of results."""
        reranker = Reranker(threshold=0.0)
        
        mock_model = MagicMock()
        scores = np.array([3.0, 5.0, 1.0, -2.0, 0.5, 2.0, 4.0])
        mock_model.predict.return_value = scores
        reranker._model = mock_model
        
        ranked = reranker.rerank(
            "Einstein discovered relativity",
            sample_statements_for_reranking,
            top_k=3
        )
        
        assert len(ranked) == 3
    
    def test_rerank_threshold_filtering(self, sample_statements_for_reranking):
        """Statements below threshold are filtered out."""
        reranker = Reranker(threshold=0.8)
        
        mock_model = MagicMock()
        # Most scores will be below 0.8 after sigmoid
        scores = np.array([0.5, 5.0, -1.0, -3.0, -0.5, 0.0, 3.0])
        mock_model.predict.return_value = scores
        reranker._model = mock_model
        
        ranked = reranker.rerank(
            "Einstein discovered relativity",
            sample_statements_for_reranking,
            threshold=0.8
        )
        
        assert all(r.score >= 0.8 for r in ranked)
    
    def test_fallback_without_model(self, sample_statements_for_reranking):
        """Returns all statements with score 1.0 when model is unavailable."""
        reranker = Reranker()
        reranker._model = None  # Ensure model is None
        
        with patch("reranker.HAS_SENTENCE_TRANSFORMERS", False):
            ranked = reranker.rerank(
                "Einstein",
                sample_statements_for_reranking
            )
        
        assert len(ranked) == len(sample_statements_for_reranking)
        assert all(r.score == 1.0 for r in ranked)


class TestRankedStatement:
    """Tests for RankedStatement dataclass."""
    
    def test_repr(self):
        stmt = RankedStatement(text="Albert Einstein | occupation | physicist", score=0.95)
        repr_str = repr(stmt)
        assert "0.950" in repr_str
        assert "Albert Einstein" in repr_str
    
    def test_default_original_is_none(self):
        stmt = RankedStatement(text="test", score=0.5)
        assert stmt.original is None


class TestFilterRelevantStatements:
    """Tests for the convenience function."""
    
    def test_returns_tuples(self, sample_statements_for_reranking):
        """Returns list of (text, score) tuples."""
        mock_model = MagicMock()
        scores = np.array([3.0, 5.0, 1.0, -2.0, 0.5, 2.0, 4.0])
        mock_model.predict.return_value = scores
        
        with patch("reranker.CrossEncoder", return_value=mock_model):
            with patch("reranker.HAS_SENTENCE_TRANSFORMERS", True):
                reranker_instance = Reranker(threshold=0.0)
                reranker_instance._model = mock_model
                
                with patch("reranker.Reranker", return_value=reranker_instance):
                    result = filter_relevant_statements(
                        "Einstein",
                        sample_statements_for_reranking,
                        threshold=0.0,
                        top_k=3
                    )
        
        assert len(result) <= 3
        if result:
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2
