"""
Tests for wikidata_api module.

Tests cover: search_entities parsing, vector_search fallback,
resolve_labels caching, extract_qids, and get_entity_claims label resolution.
"""

import pytest
from unittest.mock import patch, MagicMock
import wikidata_api


class TestSearchEntities:
    """Tests for search_entities function."""
    
    def test_parses_search_results(self, sample_entity_search_response):
        """Correctly parses entity search API response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_entity_search_response
        mock_resp.raise_for_status = MagicMock()
        
        with patch.object(wikidata_api._session, "get", return_value=mock_resp):
            results = wikidata_api.search_entities("Albert Einstein", limit=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "Q937"
        assert results[0]["label"] == "Albert Einstein"
        assert results[1]["id"] == "Q7186"
    
    def test_empty_results(self):
        """Returns empty list when no results found."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"search": []}
        mock_resp.raise_for_status = MagicMock()
        
        with patch.object(wikidata_api._session, "get", return_value=mock_resp):
            results = wikidata_api.search_entities("nonexistent_entity_xyz")
        
        assert results == []


class TestVectorSearch:
    """Tests for vector_search function."""
    
    def test_returns_results_with_correct_keys(self):
        """Vector search results have id, label, score keys."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"QID": "Q937", "label": "Albert Einstein", "similarity_score": 0.89}
        ]
        mock_resp.raise_for_status = MagicMock()
        
        with patch.object(wikidata_api._session, "get", return_value=mock_resp):
            results = wikidata_api.vector_search("physicist relativity")
        
        assert len(results) == 1
        assert results[0]["id"] == "Q937"
        assert results[0]["score"] == 0.89
    
    def test_falls_back_to_keyword_search(self, sample_entity_search_response):
        """Falls back to keyword search when vector DB fails."""
        # First call (vector search) raises, second call (keyword) succeeds
        mock_keyword_resp = MagicMock()
        mock_keyword_resp.json.return_value = sample_entity_search_response
        mock_keyword_resp.raise_for_status = MagicMock()
        
        def side_effect(url, **kwargs):
            if "vectordb" in str(url):
                raise ConnectionError("Vector DB down")
            return mock_keyword_resp
        
        with patch.object(wikidata_api._session, "get", side_effect=side_effect):
            results = wikidata_api.vector_search("Albert Einstein")
        
        # Should return keyword search results as fallback
        assert len(results) > 0
        assert results[0]["id"] == "Q937"


class TestResolveLabels:
    """Tests for resolve_labels function."""
    
    def setup_method(self):
        """Clear the label cache before each test."""
        wikidata_api._label_cache.clear()
    
    def test_resolves_multiple_ids(self, sample_labels_response):
        """Resolves multiple P/Q IDs in a single batch."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_labels_response
        mock_resp.raise_for_status = MagicMock()
        
        with patch.object(wikidata_api._session, "get", return_value=mock_resp):
            labels = wikidata_api.resolve_labels(["P106", "Q169470"])
        
        assert labels["P106"] == "occupation"
        assert labels["Q169470"] == "physicist"
    
    def test_uses_cache_on_second_call(self, sample_labels_response):
        """Uses cached results instead of making another API call."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_labels_response
        mock_resp.raise_for_status = MagicMock()
        
        with patch.object(wikidata_api._session, "get", return_value=mock_resp) as mock_get:
            wikidata_api.resolve_labels(["P106"])
            wikidata_api.resolve_labels(["P106"])  # Should use cache
            
            # API should only be called once
            assert mock_get.call_count == 1
    
    def test_graceful_degradation_on_api_failure(self):
        """Returns IDs as labels when API fails."""
        with patch.object(wikidata_api._session, "get", side_effect=ConnectionError("down")):
            labels = wikidata_api.resolve_labels(["P999"])
        
        # Should return the ID itself as fallback
        assert labels["P999"] == "P999"
    
    def test_skips_invalid_ids(self):
        """Skips non-Q/P IDs without making API calls."""
        with patch.object(wikidata_api._session, "get") as mock_get:
            labels = wikidata_api.resolve_labels(["not-an-id", ""])
            mock_get.assert_not_called()


class TestGetEntityClaims:
    """Tests for get_entity_claims function with label resolution."""
    
    def setup_method(self):
        """Clear the label cache before each test."""
        wikidata_api._label_cache.clear()
    
    def test_returns_human_readable_statements(self, sample_entity_data, sample_labels_response):
        """Claims contain resolved labels instead of raw IDs."""
        def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            
            params = kwargs.get("params", {})
            ids_param = params.get("ids", "")
            
            # First call: get entity data, subsequent: resolve labels
            if ids_param == "Q937":
                resp.json.return_value = sample_entity_data
            else:
                resp.json.return_value = sample_labels_response
            return resp
        
        with patch.object(wikidata_api._session, "get", side_effect=mock_get):
            claims = wikidata_api.get_entity_claims("Q937")
        
        assert len(claims) > 0
        
        # Check that at least one statement has resolved labels
        texts = [c["text"] for c in claims]
        # Should contain "occupation" instead of just "P106"
        has_resolved = any("occupation" in t for t in texts)
        assert has_resolved, f"No resolved labels found in: {texts}"


class TestExtractQids:
    """Tests for extract_qids_from_text utility."""
    
    def test_extracts_single_qid(self):
        assert wikidata_api.extract_qids_from_text("Entity Q937 is Einstein") == ["Q937"]
    
    def test_extracts_multiple_qids(self):
        result = wikidata_api.extract_qids_from_text("Q937 and Q7186")
        assert result == ["Q937", "Q7186"]
    
    def test_no_qids(self):
        assert wikidata_api.extract_qids_from_text("No entities here") == []


class TestFormatStatementForNli:
    """Tests for format_statement_for_nli utility."""
    
    def test_converts_pipe_format(self):
        result = wikidata_api.format_statement_for_nli("Albert Einstein | occupation | physicist")
        assert result == "Albert Einstein's occupation is physicist"
    
    def test_passthrough_non_pipe(self):
        text = "Just a regular sentence"
        assert wikidata_api.format_statement_for_nli(text) == text


class TestFormatDatavalue:
    """Tests for _format_datavalue helper."""

    def test_entity_reference(self):
        dv = {"type": "wikibase-entityid", "value": {"id": "Q937"}}
        resolved = {"Q937": "Albert Einstein"}
        result = wikidata_api._format_datavalue(dv, resolved)
        assert result == "Albert Einstein (Q937)"

    def test_time_value(self):
        dv = {"type": "time", "value": {"time": "+1879-03-14T00:00:00Z"}}
        result = wikidata_api._format_datavalue(dv, {})
        assert result == "1879-03-14"

    def test_string_value(self):
        dv = {"type": "string", "value": "hello world"}
        result = wikidata_api._format_datavalue(dv, {})
        assert result == "hello world"

    def test_quantity_value(self):
        dv = {"type": "quantity", "value": {"amount": "+42"}}
        result = wikidata_api._format_datavalue(dv, {})
        assert result == "+42"

    def test_unknown_type(self):
        dv = {"type": "unknown", "value": "something"}
        result = wikidata_api._format_datavalue(dv, {})
        assert result == "something"

    def test_monolingual_text(self):
        dv = {"type": "monolingualtext", "value": {"text": "Bonjour", "language": "fr"}}
        result = wikidata_api._format_datavalue(dv, {})
        assert result == "Bonjour"


class TestExecuteSparql:
    """Tests for execute_sparql function."""

    def test_successful_query(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "person": {"value": "http://www.wikidata.org/entity/Q937"},
                        "personLabel": {"value": "Albert Einstein"},
                    }
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(wikidata_api._session, "get", return_value=mock_resp):
            results = wikidata_api.execute_sparql("SELECT ?person WHERE { ?person wdt:P31 wd:Q5 }")

        assert len(results) == 1
        assert results[0]["person"] == "Q937"
        assert results[0]["personLabel"] == "Albert Einstein"

    def test_adds_limit_if_missing(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": {"bindings": []}}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(wikidata_api._session, "get", return_value=mock_resp) as mock_get:
            wikidata_api.execute_sparql("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }")
            call_url = mock_get.call_args[0][0]
            assert "LIMIT" in call_url

    def test_bad_request_raises(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Syntax error"

        with patch.object(wikidata_api._session, "get", return_value=mock_resp):
            with pytest.raises(ValueError, match="SPARQL error"):
                wikidata_api.execute_sparql("INVALID SPARQL")


class TestGetEntityClaimsTextify:
    """Tests for _get_entity_claims_textify fallback."""

    def test_parses_triplet_format(self):
        mock_resp = MagicMock()
        mock_resp.text = "Einstein: occupation: physicist\nEinstein: birth place: Ulm"
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(wikidata_api._session, "get", return_value=mock_resp):
            results = wikidata_api._get_entity_claims_textify("Q937")

        assert len(results) == 2
        assert results[0]["subject"] == "Einstein"
        assert results[0]["property"] == "occupation"
        assert results[0]["value"] == "physicist"

    def test_returns_empty_on_failure(self):
        import requests
        with patch.object(
            wikidata_api._session, "get",
            side_effect=requests.RequestException("down")
        ):
            results = wikidata_api._get_entity_claims_textify("Q937")
        assert results == []

