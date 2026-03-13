"""
Shared pytest fixtures for llm-wikidata-grounding tests.
"""

import sys
import os
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_entity_search_response():
    """Mock response from wbsearchentities API."""
    return {
        "search": [
            {
                "id": "Q937",
                "label": "Albert Einstein",
                "description": "German-born theoretical physicist",
                "display": {
                    "label": {"value": "Albert Einstein"},
                    "description": {"value": "German-born theoretical physicist"},
                },
            },
            {
                "id": "Q7186",
                "label": "Marie Curie",
                "description": "Polish-French physicist and chemist",
                "display": {
                    "label": {"value": "Marie Curie"},
                    "description": {"value": "Polish-French physicist and chemist"},
                },
            },
        ]
    }


@pytest.fixture
def sample_entity_data():
    """Mock response from wbgetentities API for Q937 (Einstein)."""
    return {
        "entities": {
            "Q937": {
                "labels": {"en": {"value": "Albert Einstein"}},
                "descriptions": {"en": {"value": "German-born theoretical physicist"}},
                "claims": {
                    "P106": [
                        {
                            "mainsnak": {
                                "snaktype": "value",
                                "datavalue": {
                                    "type": "wikibase-entityid",
                                    "value": {"id": "Q169470"},
                                },
                            }
                        }
                    ],
                    "P569": [
                        {
                            "mainsnak": {
                                "snaktype": "value",
                                "datavalue": {
                                    "type": "time",
                                    "value": {"time": "+1879-03-14T00:00:00Z"},
                                },
                            }
                        }
                    ],
                    "P19": [
                        {
                            "mainsnak": {
                                "snaktype": "value",
                                "datavalue": {
                                    "type": "wikibase-entityid",
                                    "value": {"id": "Q3012"},
                                },
                            }
                        }
                    ],
                },
            }
        }
    }


@pytest.fixture
def sample_labels_response():
    """Mock response from wbgetentities for label resolution."""
    return {
        "entities": {
            "P106": {"labels": {"en": {"value": "occupation"}}},
            "Q169470": {"labels": {"en": {"value": "physicist"}}},
            "P569": {"labels": {"en": {"value": "date of birth"}}},
            "P19": {"labels": {"en": {"value": "place of birth"}}},
            "Q3012": {"labels": {"en": {"value": "Ulm"}}},
        }
    }


@pytest.fixture
def sample_evidence():
    """Sample evidence statements for classifier tests."""
    return [
        "Albert Einstein (Q937) | occupation (P106) | physicist (Q169470)",
        "Albert Einstein (Q937) | date of birth (P569) | 1879-03-14",
        "Albert Einstein (Q937) | place of birth (P19) | Ulm (Q3012)",
        "Albert Einstein (Q937) | award received (P166) | Nobel Prize in Physics (Q38104)",
        "Albert Einstein (Q937) | field of work (P101) | theoretical physics (Q18362)",
    ]


@pytest.fixture
def sample_statements_for_reranking():
    """Sample statements for reranker tests."""
    return [
        "Albert Einstein | occupation | physicist",
        "Albert Einstein | notable work | theory of relativity",
        "Albert Einstein | birth place | Ulm",
        "Albert Einstein | spouse | Elsa Einstein",
        "Albert Einstein | citizenship | Switzerland",
        "Albert Einstein | award | Nobel Prize in Physics",
        "Albert Einstein | field of work | theoretical physics",
    ]
