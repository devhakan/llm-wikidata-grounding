"""
Wikidata API Utilities

This module provides functions to interact with Wikidata's APIs for
fact-checking purposes. It includes:

1. Keyword Search - Find entities by name
2. Vector Search - Semantic similarity search (no API key required)
3. Label Resolution - Batch resolve IDs to human-readable labels
4. Entity Claims - Get structured facts about entities
5. SPARQL - Complex graph queries

The vector search uses Wikidata's experimental vector database which
provides embedding-based similarity search over entity labels and descriptions.

Author: LLM Wikidata Grounding Contributors
License: MIT
"""

import re
import logging
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

WD_API_URI = "https://www.wikidata.org/w/api.php"
WD_VECTORDB_URI = "https://wd-vectordb.wmcloud.org"
WD_QUERY_URI = "https://query.wikidata.org/sparql"
WD_TEXTIFY_URI = "https://wd-textify.toolforge.org"

USER_AGENT = "LLM-Wikidata-Grounding/1.0 (https://github.com/devhakan/llm-wikidata-grounding)"

# Maximum IDs per wbgetentities batch (Wikidata API limit)
_BATCH_SIZE = 50

# Module-level cache for resolved entity/property labels
_label_cache: Dict[str, str] = {}


# =============================================================================
# Internal Helpers
# =============================================================================

def _wikidata_get(params: Dict[str, Any], timeout: int = 30) -> requests.Response:
    """Make a GET request to the Wikidata API with standard headers."""
    response = requests.get(
        WD_API_URI,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def _format_id_with_label(entity_id: str, resolved: Dict[str, str]) -> str:
    """Format an entity ID with its resolved label: 'label (ID)' or just 'ID'."""
    label = resolved.get(entity_id, entity_id)
    if label != entity_id:
        return f"{label} ({entity_id})"
    return entity_id


def _format_datavalue(datavalue: Dict, resolved: Dict[str, str]) -> str:
    """
    Convert a Wikidata datavalue to a human-readable string.

    Handles entity references, time values, strings, quantities,
    and falls back to str() for unknown types.
    """
    value_type = datavalue.get("type", "")
    value = datavalue.get("value", "")

    if value_type == "wikibase-entityid":
        return _format_id_with_label(value.get("id", ""), resolved)

    if value_type == "time":
        return value.get("time", "")[:10].replace("+", "")

    if value_type in ("string", "monolingualtext"):
        return value if isinstance(value, str) else value.get("text", str(value))

    if value_type == "quantity":
        return value.get("amount", str(value))

    return str(value)


# =============================================================================
# Keyword Search
# =============================================================================

def search_entities(
    query: str,
    entity_type: str = "item",
    limit: int = 10,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """
    Search Wikidata for entities by name using keyword matching.

    Args:
        query: Search text (e.g., "Albert Einstein")
        entity_type: "item" for Q-entities, "property" for P-entities
        limit: Maximum results to return (1-50)
        language: Language code for search (e.g., "en", "de")

    Returns:
        List of entity dictionaries with id, label, description

    Example:
        >>> results = search_entities("Marie Curie")
        >>> print(results[0])
        {"id": "Q7186", "label": "Marie Curie", "description": "Polish-French physicist"}
    """
    params = {
        "action": "wbsearchentities",
        "type": entity_type,
        "search": query,
        "limit": limit,
        "language": language,
        "format": "json",
        "origin": "*",
    }

    response = _wikidata_get(params)
    results = response.json().get("search", [])

    return [
        {
            "id": entity["id"],
            "label": entity.get("display", {}).get("label", {}).get("value", entity.get("label", "")),
            "description": entity.get("display", {}).get("description", {}).get("value", entity.get("description", "")),
        }
        for entity in results
    ]


# =============================================================================
# Vector Search (Semantic Similarity)
# =============================================================================

def vector_search(
    query: str,
    limit: int = 20,
    language: str = "en",
    entity_type: str = "item",
) -> List[Dict[str, Any]]:
    """
    Search Wikidata using semantic similarity (vector embeddings).

    Uses Wikidata's experimental vector database (alpha).
    No API key required — just a descriptive User-Agent.

    Args:
        query: Natural language query or claim
        limit: Maximum results to return
        language: Language code (e.g., "en", "de")
        entity_type: "item" for Q-entities, "property" for P-entities

    Returns:
        List of entity dictionaries with id, label, description, and score

    Example:
        >>> results = vector_search("physicist who discovered relativity")
        >>> print(results[0])
        {"id": "Q937", "label": "Albert Einstein", "score": 0.89}
    """
    endpoint = "item" if entity_type == "item" else "property"

    try:
        response = requests.get(
            f"{WD_VECTORDB_URI}/{endpoint}/query",
            params={"query": query, "lang": language, "limit": limit},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        results = data if isinstance(data, list) else data.get("results", data.get("items", []))

        return [
            {
                "id": r.get("QID", r.get("qid", r.get("id", ""))),
                "label": r.get("label", ""),
                "description": r.get("description", ""),
                "score": r.get("similarity_score", r.get("rrf_score", r.get("score", 0.0))),
                "source": r.get("source", ""),
            }
            for r in results
        ]

    except Exception as e:
        logger.warning("Vector search failed: %s. Falling back to keyword search.", e)
        return search_entities(query, limit=limit, language=language)


# =============================================================================
# Label Resolution (Batch)
# =============================================================================

def resolve_labels(
    ids: List[str],
    language: str = "en",
) -> Dict[str, str]:
    """
    Resolve multiple Wikidata IDs to their labels in a single API call.

    Uses batch wbgetentities to minimize API requests.
    Results are cached in the module-level _label_cache.

    Args:
        ids: List of Wikidata IDs (e.g., ["P106", "Q169470", "P27"])
        language: Language code for labels

    Returns:
        Dictionary mapping ID to label

    Example:
        >>> labels = resolve_labels(["P106", "Q169470"])
        >>> print(labels)
        {"P106": "occupation", "Q169470": "physicist"}
    """
    global _label_cache

    uncached = [
        i for i in ids
        if i and i not in _label_cache and re.match(r'^[PQ]\d+$', i)
    ]

    if not uncached:
        return {i: _label_cache.get(i, i) for i in ids}

    for batch_start in range(0, len(uncached), _BATCH_SIZE):
        batch = uncached[batch_start:batch_start + _BATCH_SIZE]

        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": language,
            "format": "json",
        }

        try:
            response = _wikidata_get(params, timeout=15)
            entities = response.json().get("entities", {})
            for eid, edata in entities.items():
                label = edata.get("labels", {}).get(language, {}).get("value", "")
                _label_cache[eid] = label if label else eid
        except Exception:
            for eid in batch:
                _label_cache[eid] = eid

    return {i: _label_cache.get(i, i) for i in ids}


# =============================================================================
# Entity Claims
# =============================================================================

def _collect_ids_from_claims(claims: Dict) -> set:
    """Collect all entity/property IDs from claims for batch label resolution."""
    ids = set(claims.keys())

    for claim_list in claims.values():
        for claim in claim_list:
            mainsnak = claim.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                continue

            datavalue = mainsnak.get("datavalue", {})
            if datavalue.get("type") == "wikibase-entityid":
                val_id = datavalue.get("value", {}).get("id", "")
                if val_id:
                    ids.add(val_id)

            for qual_prop, qual_list in claim.get("qualifiers", {}).items():
                ids.add(qual_prop)
                for qual in qual_list:
                    qv = qual.get("datavalue", {}).get("value", "")
                    if isinstance(qv, dict) and "id" in qv:
                        ids.add(qv["id"])

    return ids


def _build_statement(subject: str, prop_display: str, value_str: str) -> Dict[str, str]:
    """Create a statement dictionary with consistent format."""
    return {
        "subject": subject,
        "property": prop_display,
        "value": value_str,
        "text": f"{subject} | {prop_display} | {value_str}",
    }


def get_entity_claims(
    entity_id: str,
    language: str = "en",
) -> List[Dict[str, str]]:
    """
    Get all claims (facts) about a Wikidata entity.

    Uses native Wikidata API (wbgetentities) with batch label resolution
    for human-readable output. Falls back to Textify if needed.

    Args:
        entity_id: Wikidata entity ID (e.g., "Q937")
        language: Language for labels

    Returns:
        List of statement dictionaries with subject, property, value, and text

    Example:
        >>> claims = get_entity_claims("Q937")
        >>> for c in claims[:3]:
        ...     print(c["text"])
        "Albert Einstein (Q937) | occupation (P106) | physicist (Q169470)"
    """
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels|descriptions|claims",
        "languages": language,
        "format": "json",
    }

    try:
        response = _wikidata_get(params, timeout=15)
        data = response.json()

        entity = data.get("entities", {}).get(entity_id, {})
        label = entity.get("labels", {}).get(language, {}).get("value", entity_id)
        description = entity.get("descriptions", {}).get(language, {}).get("value", "")
        subject = f"{label} ({entity_id})"

        statements = []

        if description:
            statements.append(_build_statement(subject, "description", description))

        claims = entity.get("claims", {})
        resolved = resolve_labels(list(_collect_ids_from_claims(claims)), language=language)

        for prop_id, claim_list in claims.items():
            prop_display = _format_id_with_label(prop_id, resolved)

            for claim in claim_list:
                mainsnak = claim.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue

                value_str = _format_datavalue(mainsnak.get("datavalue", {}), resolved)
                statements.append(_build_statement(subject, prop_display, value_str))

                for qual_prop, qual_list in claim.get("qualifiers", {}).items():
                    qual_prop_display = _format_id_with_label(qual_prop, resolved)
                    for qual in qual_list:
                        qual_dv = qual.get("datavalue", {})
                        qual_str = _format_datavalue(qual_dv, resolved) if qual_dv else ""
                        if qual_str:
                            statements.append(_build_statement(subject, qual_prop_display, qual_str))

        return statements

    except Exception as e:
        logger.warning("Native API failed for %s: %s. Falling back to Textify.", entity_id, e)
        return _get_entity_claims_textify(entity_id, language)


def _get_entity_claims_textify(
    entity_id: str,
    language: str = "en",
) -> List[Dict[str, str]]:
    """Fallback: Get claims via Textify service."""
    params = {
        "id": entity_id,
        "external_ids": "false",
        "all_ranks": "false",
        "lang": language,
        "format": "triplet",
    }

    try:
        response = requests.get(
            WD_TEXTIFY_URI,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
    except Exception as e:
        logger.warning("Textify API failed for %s: %s", entity_id, e)
        return []

    if not response.text:
        return []

    content_type = response.headers.get("content-type", "")
    raw_text = response.json() if content_type.startswith("application/json") else response.text

    if not isinstance(raw_text, str):
        return []

    statements = []
    for line in raw_text.strip().split("\n"):
        parts = line.split(": ", 2)
        if len(parts) >= 3:
            statements.append(_build_statement(
                parts[0].strip(), parts[1].strip(), parts[2].strip()
            ))
    return statements


# =============================================================================
# SPARQL Queries
# =============================================================================

def execute_sparql(query: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Execute a SPARQL query against Wikidata.

    Args:
        query: SPARQL query string
        limit: Maximum results (added if not in query)

    Returns:
        List of result dictionaries

    Example:
        >>> query = '''
        ... SELECT ?person ?personLabel WHERE {
        ...   ?person wdt:P31 wd:Q5 .
        ...   ?person wdt:P166 wd:Q38104 .
        ...   SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        ... } LIMIT 5
        ... '''
        >>> results = execute_sparql(query)
    """
    if "LIMIT" not in query.upper():
        query = f"{query}\nLIMIT {limit}"

    params = urlencode({"query": query, "format": "json"})
    url = f"{WD_QUERY_URI}?{params}"

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )

    if response.status_code == 400:
        raise ValueError(f"SPARQL error: {response.text[:200]}")

    response.raise_for_status()
    bindings = response.json()["results"]["bindings"]

    wikidata_prefix = "http://www.wikidata.org/entity/"
    return [
        {
            key: val.get("value", "").replace(wikidata_prefix, "")
            for key, val in binding.items()
        }
        for binding in bindings
    ]


# =============================================================================
# Utility Functions
# =============================================================================

def extract_qids_from_text(text: str) -> List[str]:
    """Extract all Wikidata QIDs from text."""
    return re.findall(r"Q\d+", text)


def format_statement_for_nli(statement: str) -> str:
    """
    Format a Wikidata statement for use with NLI models.

    Converts pipe-separated format to natural language.
    "Albert Einstein | occupation | physicist" → "Albert Einstein's occupation is physicist"
    """
    parts = statement.split(" | ")
    if len(parts) == 3:
        subject, prop, value = parts
        return f"{subject}'s {prop} is {value}"
    return statement


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Wikidata API Utilities - Self Test")
    print("=" * 60)

    # Test 1: Keyword Search
    print("\n[1] Testing keyword search...")
    results = search_entities("Albert Einstein", limit=3)
    if results and results[0]["id"] == "Q937":
        print(f"    ✓ Found Einstein: {results[0]}")
    else:
        print("    ✗ Einstein not found")

    # Test 2: Entity Claims
    print("\n[2] Testing entity claims...")
    claims = get_entity_claims("Q937")
    if claims:
        print(f"    ✓ Retrieved {len(claims)} claims")
        print(f"    Sample: {claims[0]['text']}")
    else:
        print("    ✗ No claims retrieved")

    # Test 3: Vector Search
    print("\n[3] Testing vector search...")
    try:
        results = vector_search("physicist who developed relativity theory")
        if results:
            print(f"    ✓ Vector search returned {len(results)} results")
            print(f"    Top result: {results[0]}")
        else:
            print("    ⚠ Vector search returned no results")
    except Exception as e:
        print(f"    ✗ Vector search error: {e}")

    print("\n" + "=" * 60)
    print("Self-test complete!")
    print("=" * 60)
