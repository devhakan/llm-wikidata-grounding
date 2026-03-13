"""
Wikidata API Utilities

This module provides functions to interact with Wikidata's APIs for
fact-checking purposes. It includes:

1. Keyword Search - Find entities by name
2. Vector Search - Semantic similarity search (no API key required)
3. Entity Claims - Get structured facts about entities
4. SPARQL - Complex graph queries

The vector search uses Wikidata's experimental vector database which
provides embedding-based similarity search over entity labels and descriptions.

Author: LLM Wikidata Grounding Contributors
License: MIT
"""

import os
import re
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode


# =============================================================================
# Label Cache
# =============================================================================

# Module-level cache for resolved entity/property labels
# Maps ID (e.g., "P106", "Q169470") to label (e.g., "occupation", "physicist")
_label_cache: Dict[str, str] = {}

# =============================================================================
# API Endpoints
# =============================================================================

# Main Wikidata API for entity operations
WD_API_URI = "https://www.wikidata.org/w/api.php"

# Vector search API (experimental - requires API key)
WD_VECTORDB_URI = "https://wd-vectordb.wmcloud.org"

# SPARQL endpoint for complex queries
WD_QUERY_URI = "https://query.wikidata.org/sparql"

# Textify service - converts claims to human-readable format
WD_TEXTIFY_URI = "https://wd-textify.toolforge.org"

# User agent for API requests (required by Wikimedia policy)
USER_AGENT = "LLM-Wikidata-Grounding/1.0 (https://github.com/devhakan/llm-wikidata-grounding)"


# =============================================================================
# Keyword Search
# =============================================================================

def search_entities(
    query: str,
    entity_type: str = "item",
    limit: int = 10,
    language: str = "en"
) -> List[Dict[str, Any]]:
    """
    Search Wikidata for entities by name using keyword matching.
    
    This is a simple text-based search that matches entity labels,
    aliases, and descriptions. For semantic similarity search,
    use vector_search() instead.
    
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
    
    response = requests.get(
        WD_API_URI,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30
    )
    response.raise_for_status()
    
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
    entity_type: str = "item"
) -> List[Dict[str, Any]]:
    """
    Search Wikidata using semantic similarity (vector embeddings).
    
    This function uses Wikidata's experimental vector database to find
    entities that are semantically similar to the query, even if they
    don't contain the exact search terms.
    
    The vector database contains embeddings of entity labels and descriptions
    computed using a sentence transformer model.
    
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
        
    Note:
        Uses Wikidata's experimental vector database (alpha).
        No API key required - just a descriptive User-Agent.
    """
    try:
        # Use the correct endpoint format: /item/query or /property/query
        endpoint = "item" if entity_type == "item" else "property"
        
        params = {
            "query": query,
            "lang": language,
            "limit": limit
        }
        
        headers = {
            "User-Agent": USER_AGENT,
        }
        
        response = requests.get(
            f"{WD_VECTORDB_URI}/{endpoint}/query",
            params=params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, list):
            results = data
        else:
            results = data.get("results", data.get("items", []))
        
        return [
            {
                "id": r.get("QID", r.get("qid", r.get("id", ""))),
                "label": r.get("label", ""),
                "description": r.get("description", ""),
                "score": r.get("similarity_score", r.get("rrf_score", r.get("score", 0.0))),
                "source": r.get("source", "")
            }
            for r in results
        ]
        
    except Exception as e:
        print(f"Vector search failed: {e}")
        print("Falling back to keyword search...")
        return search_entities(query, limit=limit, language=language)


# =============================================================================
# Label Resolution (Batch)
# =============================================================================

def resolve_labels(
    ids: List[str],
    language: str = "en"
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
    
    # Filter out already-cached and invalid IDs
    uncached = [i for i in ids if i and i not in _label_cache and re.match(r'^[PQ]\d+$', i)]
    
    if not uncached:
        return {i: _label_cache.get(i, i) for i in ids}
    
    # Wikidata API supports up to 50 IDs per request
    for batch_start in range(0, len(uncached), 50):
        batch = uncached[batch_start:batch_start + 50]
        
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": language,
            "format": "json",
        }
        
        try:
            response = requests.get(
                WD_API_URI,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            entities = data.get("entities", {})
            for eid, edata in entities.items():
                label = edata.get("labels", {}).get(language, {}).get("value", "")
                _label_cache[eid] = label if label else eid
        except Exception:
            # On failure, cache IDs as their own labels (graceful degradation)
            for eid in batch:
                _label_cache[eid] = eid
    
    return {i: _label_cache.get(i, i) for i in ids}


# =============================================================================
# Entity Claims
# =============================================================================

def get_entity_claims(
    entity_id: str,
    language: str = "en"
) -> List[Dict[str, str]]:
    """
    Get all claims (facts) about a Wikidata entity.
    
    Uses native Wikidata API (wbgetentities) for faster, more reliable access.
    Falls back to Textify if needed.
    
    Args:
        entity_id: Wikidata entity ID (e.g., "Q937")
        language: Language for labels
    
    Returns:
        List of statement dictionaries with subject, property, value, and text
        
    Example:
        >>> claims = get_entity_claims("Q937")
        >>> for c in claims[:3]:
        ...     print(c["text"])
        "Albert Einstein (Q937) | occupation (P106) | physicist"
    """
    # First get entity label
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels|descriptions|claims",
        "languages": language,
        "format": "json",
    }
    
    try:
        response = requests.get(
            WD_API_URI,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        entity = data.get("entities", {}).get(entity_id, {})
        label = entity.get("labels", {}).get(language, {}).get("value", entity_id)
        description = entity.get("descriptions", {}).get(language, {}).get("value", "")
        
        statements = []
        
        # Add description as first statement
        if description:
            statements.append({
                "subject": f"{label} ({entity_id})",
                "property": "description",
                "value": description,
                "text": f"{label} ({entity_id}) | description | {description}"
            })
        
        # Process claims — collect all IDs for batch label resolution
        claims = entity.get("claims", {})
        ids_to_resolve = set(claims.keys())  # property IDs
        
        # First pass: collect all entity-value and qualifier IDs
        for prop_id, claim_list in claims.items():
            for claim in claim_list:
                mainsnak = claim.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue
                datavalue = mainsnak.get("datavalue", {})
                if datavalue.get("type") == "wikibase-entityid":
                    val_id = datavalue.get("value", {}).get("id", "")
                    if val_id:
                        ids_to_resolve.add(val_id)
                # Qualifier IDs
                for qual_prop, qual_list in claim.get("qualifiers", {}).items():
                    ids_to_resolve.add(qual_prop)
                    for qual in qual_list:
                        qv = qual.get("datavalue", {}).get("value", "")
                        if isinstance(qv, dict) and "id" in qv:
                            ids_to_resolve.add(qv["id"])
        
        # Batch resolve all IDs to human-readable labels
        resolved = resolve_labels(list(ids_to_resolve), language=language)
        
        # Second pass: build statements with resolved labels
        for prop_id, claim_list in claims.items():
            prop_label = resolved.get(prop_id, prop_id)
            
            for claim in claim_list:
                mainsnak = claim.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue
                
                datavalue = mainsnak.get("datavalue", {})
                value_type = datavalue.get("type", "")
                value = datavalue.get("value", "")
                
                # Format value based on type
                if value_type == "wikibase-entityid":
                    value_id = value.get("id", "")
                    value_label = resolved.get(value_id, value_id)
                    value_str = f"{value_label} ({value_id})" if value_label != value_id else value_id
                elif value_type == "time":
                    value_str = value.get("time", "")[:10].replace("+", "")  # Just date
                elif value_type == "string" or value_type == "monolingualtext":
                    value_str = value if isinstance(value, str) else value.get("text", str(value))
                elif value_type == "quantity":
                    value_str = value.get("amount", str(value))
                else:
                    value_str = str(value)
                
                prop_display = f"{prop_label} ({prop_id})" if prop_label != prop_id else prop_id
                
                statements.append({
                    "subject": f"{label} ({entity_id})",
                    "property": prop_display,
                    "value": value_str,
                    "text": f"{label} ({entity_id}) | {prop_display} | {value_str}"
                })
                
                # Add qualifiers (like point in time for awards)
                qualifiers = claim.get("qualifiers", {})
                for qual_prop, qual_list in qualifiers.items():
                    qual_prop_label = resolved.get(qual_prop, qual_prop)
                    qual_prop_display = f"{qual_prop_label} ({qual_prop})" if qual_prop_label != qual_prop else qual_prop
                    
                    for qual in qual_list:
                        qual_value = qual.get("datavalue", {}).get("value", "")
                        if isinstance(qual_value, dict):
                            if "time" in qual_value:
                                qual_str = qual_value["time"][:10].replace("+", "")
                            elif "id" in qual_value:
                                qual_id = qual_value["id"]
                                qual_label = resolved.get(qual_id, qual_id)
                                qual_str = f"{qual_label} ({qual_id})" if qual_label != qual_id else qual_id
                            else:
                                qual_str = str(qual_value)
                        else:
                            qual_str = str(qual_value)
                        
                        statements.append({
                            "subject": f"{label} ({entity_id})",
                            "property": qual_prop_display,
                            "value": qual_str,
                            "text": f"{label} ({entity_id}) | {qual_prop_display} | {qual_str}"
                        })
        
        return statements
        
    except Exception as e:
        # Fallback to Textify if native API fails
        return _get_entity_claims_textify(entity_id, language)


def _get_entity_claims_textify(
    entity_id: str,
    language: str = "en"
) -> List[Dict[str, str]]:
    """Fallback: Get claims via Textify service."""
    params = {
        "id": entity_id,
        "external_ids": "false",
        "all_ranks": "false",
        "lang": language,
        "format": "triplet",
    }
    
    response = requests.get(
        WD_TEXTIFY_URI,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30
    )
    response.raise_for_status()
    
    if not response.text:
        return []
    
    raw_text = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
    
    if isinstance(raw_text, str):
        statements = []
        for line in raw_text.strip().split("\n"):
            if ": " in line:
                parts = line.split(": ", 2)
                if len(parts) >= 3:
                    statements.append({
                        "subject": parts[0].strip(),
                        "property": parts[1].strip(),
                        "value": parts[2].strip(),
                        "text": f"{parts[0].strip()} | {parts[1].strip()} | {parts[2].strip()}"
                    })
        return statements
    
    return []


def get_entity_statements_text(
    entity_id: str,
    language: str = "en"
) -> str:
    """
    Get all claims for an entity as a single text string.
    
    This format is suitable for embedding or NLI model input.
    
    Args:
        entity_id: Wikidata entity ID
        language: Language code
    
    Returns:
        Formatted string with all statements, one per line
    """
    params = {
        "id": entity_id,
        "external_ids": "false",
        "all_ranks": "false",
        "lang": language,
        "format": "triplet",
    }
    
    response = requests.get(
        WD_TEXTIFY_URI,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30
    )
    response.raise_for_status()
    
    return response.json() if response.text else ""


# =============================================================================
# SPARQL Queries
# =============================================================================

def execute_sparql(query: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Execute a SPARQL query against Wikidata.
    
    SPARQL enables complex queries across the knowledge graph,
    such as finding all entities matching certain criteria.
    
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
        timeout=60
    )
    
    if response.status_code == 400:
        raise ValueError(f"SPARQL error: {response.text[:200]}")
    
    response.raise_for_status()
    
    bindings = response.json()["results"]["bindings"]
    
    # Simplify results
    results = []
    for binding in bindings:
        row = {}
        for key, value in binding.items():
            val = value.get("value", "")
            # Shorten Wikidata URIs
            if val.startswith("http://www.wikidata.org/entity/"):
                val = val.replace("http://www.wikidata.org/entity/", "")
            row[key] = val
        results.append(row)
    
    return results


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
    """
    # "Albert Einstein | occupation | physicist" -> "Albert Einstein's occupation is physicist"
    parts = statement.split(" | ")
    if len(parts) == 3:
        subject, prop, value = parts
        return f"{subject}'s {prop} is {value}"
    return statement


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
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
