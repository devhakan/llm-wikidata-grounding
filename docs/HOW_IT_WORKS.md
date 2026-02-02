# How It Works

This document provides a detailed technical explanation of how the LLM Wikidata Grounding system works.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                                │
│                 "Einstein discovered relativity"                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Local LLM (Ollama)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Model: qwen3:4b / qwen2.5:7b / llama3:8b               │  │
│  │  Tool-calling capability enabled                          │  │
│  │  Decides which Wikidata APIs to call                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ search_entity │ │  get_claims   │ │    SPARQL     │
        │    (API)      │ │  (Textify)    │ │   (Query)     │
        └───────────────┘ └───────────────┘ └───────────────┘
                    │             │             │
                    └─────────────┼─────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Wikidata Knowledge Base                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │  Q937   │──│  P800   │──│ Q43514  │  │  ...    │           │
│  │Einstein │  │ notable │  │relativity│  │         │           │
│  └─────────┘  │  work   │  └─────────┘  └─────────┘           │
│               └─────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Local LLM (Ollama)

The system uses [Ollama](https://ollama.ai) to run open-source LLMs locally on your machine.

**Why Local LLM?**
- **Privacy**: Your data never leaves your machine
- **No API costs**: No per-token charges
- **Offline capable**: Works without internet (after model download)
- **Customizable**: Use any compatible model

**Recommended Models:**

| Model | Parameters | RAM Required | Speed* | Best For |
|-------|------------|--------------|--------|----------|
| qwen3:4b | 4B | 6 GB | ~42 tok/s | Quick checks |
| qwen2.5:7b | 7B | 8 GB | ~25 tok/s | Balanced |
| llama3:8b | 8B | 8 GB | ~22 tok/s | Alternative |
| qwen2.5:14b | 14B | 16 GB | ~12 tok/s | Best quality |

*Speed measured on Apple M3 Pro with GPU acceleration

**Hardware Requirements:**

- **Minimum**: 8 GB RAM, any CPU (slow, ~4 tok/s)
- **Recommended**: 16 GB RAM, Apple Silicon or NVIDIA GPU with 8+ GB VRAM (~30-50 tok/s)
- **Optimal**: 32+ GB RAM, Apple M3 Pro/Max or NVIDIA RTX 4080+ (~50+ tok/s)

### 2. Tool Calling System

Modern LLMs support "tool calling" (also called function calling). The LLM can:

1. Analyze the user's query
2. Decide which tools/functions to call
3. Generate the function arguments
4. Process the results
5. Continue until it has enough information

**Available Tools:**

```python
# 1. Entity Search
search_entities(query="Albert Einstein")
# Returns: {"Q937": {"label": "Albert Einstein", "description": "..."}}

# 2. Get Claims
get_entity_claims(entity_id="Q937")
# Returns: "Albert Einstein (Q937): notable work (P800): relativity..."

# 3. SPARQL Query
execute_sparql(query="SELECT ?x WHERE { ?x wdt:P166 wd:Q38104 }")
# Returns: Table of Nobel Prize winners
```

### 3. Wikidata APIs

We use three different Wikidata services:

#### a) MediaWiki Action API (`wbsearchentities`)

**Purpose**: Find entities by name or description

**Endpoint**: `https://www.wikidata.org/w/api.php`

**Example**:
```
GET /w/api.php?action=wbsearchentities&search=Marie+Curie&language=en&format=json
```

**Returns**:
```json
{
  "search": [
    {
      "id": "Q7186",
      "label": "Marie Curie",
      "description": "Polish-French physicist and chemist"
    }
  ]
}
```

#### b) Textify API

**Purpose**: Convert Wikidata claims to human-readable format

**Endpoint**: `https://wd-textify.toolforge.org`

**Why Textify?**
- Raw Wikidata claims are hard to read (nested JSON with QIDs everywhere)
- Textify resolves QIDs to labels
- Includes qualifiers (dates, references, etc.)

**Example**:
```
GET /textify?id=Q7186&lang=en&format=triplet
```

**Returns**:
```
Marie Curie (Q7186): occupation (P106): physicist (Q169470)
Marie Curie (Q7186): occupation (P106): chemist (Q593644)
Marie Curie (Q7186): award received (P166): Nobel Prize in Physics (Q38104) | date: 1903
Marie Curie (Q7186): award received (P166): Nobel Prize in Chemistry (Q44585) | date: 1911
Marie Curie (Q7186): child (P40): Irène Joliot-Curie (Q7504)
...
```

#### c) SPARQL Endpoint

**Purpose**: Complex queries across the knowledge graph

**Endpoint**: `https://query.wikidata.org/sparql`

**When to use SPARQL**:
- Finding multiple entities matching criteria
- Aggregation (counting, grouping)
- Path queries (relationships between entities)
- Comparing entities

**Example** - Find landlocked countries with population > 10 million:
```sparql
SELECT ?country ?countryLabel ?population WHERE {
  ?country wdt:P31 wd:Q6256 .       # is a country
  ?country wdt:P1566 ?geonames .     # has Geonames ID (excludes historical)
  ?country wdt:P1082 ?population .   # has population
  FILTER NOT EXISTS {
    ?country wdt:P206 ?water .       # has no bodies of water (coastline)
  }
  FILTER (?population > 10000000)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY DESC(?population)
```

### 4. Vector Search (Optional)

For semantic search, Wikimedia provides an experimental vector database:

**Endpoint**: `https://wd-vectordb.wmcloud.org`

**How it works**:
1. Wikidata labels/descriptions are embedded using a sentence transformer
2. Search query is embedded the same way
3. Cosine similarity finds closest matches

**Advantages over keyword search**:
- Finds "car" when searching "automobile"
- Works with misspellings
- Handles synonyms

**Note**: Requires API key from Wikimedia. Contact the Wikidata team for access.

## Verification Process

### Step-by-Step Example

**Claim**: "Marie Curie's daughter won a Nobel Prize"

**Step 1**: LLM analyzes claim, identifies entities
```
Entities: "Marie Curie", "daughter", "Nobel Prize"
```

**Step 2**: LLM calls `search_entities("Marie Curie")`
```
Result: Q7186 - Marie Curie
```

**Step 3**: LLM calls `get_entity_claims("Q7186")`
```
Result: 
...
Marie Curie (Q7186): child (P40): Irène Joliot-Curie (Q7504)
Marie Curie (Q7186): child (P40): Ève Curie (Q230068)
...
```

**Step 4**: LLM calls `get_entity_claims("Q7504")` (Irène)
```
Result:
...
Irène Joliot-Curie (Q7504): award received (P166): Nobel Prize in Chemistry (Q44585) | date: 1935
...
```

**Step 5**: LLM synthesizes answer
```
✓ VERIFIED: Marie Curie's daughter Irène Joliot-Curie (Q7504) received 
the Nobel Prize in Chemistry (Q44585) in 1935, as recorded in Wikidata.
```

## Limitations

1. **Knowledge cutoff**: Wikidata is community-edited; very recent events may not be included
2. **Coverage gaps**: Not everything is in Wikidata
3. **Language**: Best results with English; other languages have varying coverage
4. **Interpretation**: LLM may occasionally misread search results
5. **Complex reasoning**: Very complex logical chains may exceed model capability
