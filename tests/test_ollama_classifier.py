"""
Tests for ollama_classifier module.

Tests cover response parsing logic (verdict, confidence, reasoning)
without requiring a running Ollama instance.
"""

import pytest
from unittest.mock import patch, MagicMock
from ollama_classifier import OllamaClassifier, Verdict, ClassificationResult


class TestParseResponse:
    """Tests for _parse_response — the core parsing logic."""
    
    def setup_method(self):
        self.classifier = OllamaClassifier(model="test-model")
    
    def test_parse_supported(self):
        text = "VERDICT: SUPPORTED\nCONFIDENCE: HIGH\nREASONING: Evidence confirms the claim."
        result = self.classifier._parse_response(text)
        
        assert result.verdict == Verdict.SUPPORTED
        assert result.confidence == 0.9
        assert "confirms" in result.reasoning.lower()
    
    def test_parse_refuted(self):
        text = "VERDICT: REFUTED\nCONFIDENCE: HIGH\nREASONING: Evidence contradicts the claim."
        result = self.classifier._parse_response(text)
        
        assert result.verdict == Verdict.REFUTED
        assert result.confidence == 0.9
    
    def test_parse_not_enough_info(self):
        text = "VERDICT: NOT_ENOUGH_INFO\nCONFIDENCE: LOW\nREASONING: Insufficient evidence."
        result = self.classifier._parse_response(text)
        
        assert result.verdict == Verdict.NOT_ENOUGH_INFO
        assert result.confidence == 0.5
    
    def test_parse_medium_confidence(self):
        text = "VERDICT: SUPPORTED\nCONFIDENCE: MEDIUM\nREASONING: Partial match."
        result = self.classifier._parse_response(text)
        
        assert result.confidence == 0.7
    
    def test_parse_no_colon_space(self):
        """Handles 'VERDICT:SUPPORTED' without space after colon."""
        text = "VERDICT:SUPPORTED\nCONFIDENCE: HIGH\nREASONING: Yes."
        result = self.classifier._parse_response(text)
        
        assert result.verdict == Verdict.SUPPORTED
    
    def test_parse_unknown_format_defaults_to_nei(self):
        """Unknown format defaults to NOT_ENOUGH_INFO."""
        text = "I'm not sure what to think about this claim."
        result = self.classifier._parse_response(text)
        
        assert result.verdict == Verdict.NOT_ENOUGH_INFO
    
    def test_raw_response_preserved(self):
        text = "VERDICT: SUPPORTED\nCONFIDENCE: HIGH\nREASONING: Yes."
        result = self.classifier._parse_response(text)
        
        assert result.raw_response == text


class TestClassify:
    """Tests for classify method with mocked Ollama."""
    
    def setup_method(self):
        self.classifier = OllamaClassifier(model="test-model")
    
    def test_empty_evidence_returns_nei(self):
        """Empty evidence returns NOT_ENOUGH_INFO without calling Ollama."""
        result = self.classifier.classify("Some claim", [])
        
        assert result.verdict == Verdict.NOT_ENOUGH_INFO
        assert result.confidence == 0.0
    
    def test_connection_error_returns_nei(self):
        """Connection error to Ollama returns NOT_ENOUGH_INFO gracefully."""
        import requests
        with patch("ollama_classifier.requests.post", side_effect=requests.exceptions.ConnectionError):
            result = self.classifier.classify("Some claim", ["evidence"])
        
        assert result.verdict == Verdict.NOT_ENOUGH_INFO
        assert "connect" in result.reasoning.lower() or "Ollama" in result.reasoning
    
    def test_successful_classification(self, sample_evidence):
        """Returns parsed verdict from Ollama response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "VERDICT: SUPPORTED\nCONFIDENCE: HIGH\nREASONING: Evidence confirms Einstein was born in Germany."
        }
        mock_resp.raise_for_status = MagicMock()
        
        with patch("ollama_classifier.requests.post", return_value=mock_resp):
            result = self.classifier.classify("Einstein was born in Germany", sample_evidence)
        
        assert result.verdict == Verdict.SUPPORTED
        assert result.confidence == 0.9


class TestIsAvailable:
    """Tests for is_available method."""
    
    def test_available_with_matching_model(self):
        classifier = OllamaClassifier(model="qwen2.5:7b")
        
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5:7b"}]
        }
        
        with patch("ollama_classifier.requests.get", return_value=mock_resp):
            assert classifier.is_available() is True
    
    def test_not_available_when_offline(self):
        classifier = OllamaClassifier()
        
        with patch("ollama_classifier.requests.get", side_effect=ConnectionError):
            assert classifier.is_available() is False
