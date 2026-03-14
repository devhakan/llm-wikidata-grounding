# How It Works

This document provides a detailed technical explanation of how the LLM Wikidata Grounding system works.

## Architecture

The hybrid pipeline combines four stages to verify claims against Wikidata:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                                │
│              "Aziz Sancar won Nobel Prize in Chemistry"          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Vector Search (wikidata_api.py)                       │
│  Wikidata Vector DB → Find semantically similar entities        │
│  Fallback: keyword search via MediaWiki Action API              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ Entity IDs (e.g. Q929627)
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Statement Retrieval (wikidata_api.py)                 │
│  wbgetentities API → Fetch structured claims + labels           │
│  Batch label resolution: P166 → "award received"                │
│  Fallback: Textify API                                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ ~200+ statements per entity
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Cross-Encoder Reranking (reranker.py)                 │
│  ms-marco-MiniLM-L-6-v2 → Score (claim, statement) pairs       │
│  Sigmoid normalization → Filter by threshold → Top K            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ Top 10 relevant statements
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: LLM Classification (ollama_classifier.py)             │
│  Ollama (qwen2.5:7b) → Reason over evidence → Verdict          │
│  Output: SUPPORTED / REFUTED / NOT_ENOUGH_INFO + confidence     │
└─────────────────────────────────────────────────────────────────┘
```

## Stage Details

### Stage 1: Vector Search

Finds relevant Wikidata entities using semantic similarity.

**Primary**: [Wikidata Vector Database](https://wd-vectordb.wmcloud.org) (experimental, no API key required)
- Embeds the claim using sentence transformers
- Finds entities with similar labels/descriptions via cosine similarity
- Returns up to 20 results with similarity scores

**Fallback**: MediaWiki `wbsearchentities` API (keyword matching)
- Used when Vector DB is unreachable
- Exact and fuzzy keyword matching on entity labels

### Stage 2: Statement Retrieval

Fetches structured facts (claims) about found entities.

**Primary**: `wbgetentities` API with batch label resolution
- Retrieves all claims for an entity in one API call
- Resolves property/entity IDs to human-readable labels (e.g., `P106` → "occupation")
- Uses a thread-safe cache to avoid redundant label lookups
- Handles qualifiers (dates, references)
- Output format: `"Albert Einstein (Q937) | occupation (P106) | physicist (Q169470)"`

**Fallback**: [Textify API](https://wd-textify.toolforge.org) (Toolforge service)

### Stage 3: Cross-Encoder Reranking

Filters the ~200+ statements down to the most relevant ones.

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB)

**Process**:
1. Form (claim, statement) pairs
2. Cross-encoder scores each pair for relevance
3. Raw logits → sigmoid normalization → scores in [0, 1]
4. Filter by threshold (default: 0.3)
5. Keep top K statements (default: 10)

**Why Cross-Encoder?** Unlike bi-encoders, cross-encoders attend to both inputs jointly, giving much better relevance judgments at the cost of speed. Since we only score ~200 pairs, this tradeoff is worthwhile.

### Stage 4: LLM Classification

A local LLM reasons over the filtered evidence to produce a verdict.

**Model**: Ollama with `qwen2.5:7b` (configurable)

**Prompt structure**:
```
You are a fact-checker. Determine if the claim is supported by evidence.

CLAIM: {claim}
EVIDENCE FROM WIKIDATA:
• statement 1
• statement 2
...

Respond in this exact format:
VERDICT: [SUPPORTED/REFUTED/NOT_ENOUGH_INFO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [one sentence explanation]
```

**Verdict types**:

| Verdict | Meaning |
|---------|---------|
| SUPPORTED | Evidence confirms the claim |
| REFUTED | Evidence contradicts the claim |
| NOT_ENOUGH_INFO | Insufficient evidence |

## Example Walkthrough

**Claim**: "Aziz Sancar won the Nobel Prize in Chemistry in 2015"

1. **Vector Search** → Finds Q929627 (Aziz Sancar), Q44585 (Nobel Prize in Chemistry)
2. **Statement Retrieval** → 200+ statements including:
   - `Aziz Sancar | award received | Nobel Prize in Chemistry`
   - `Nobel Prize in Chemistry | point in time | 2015`
   - `Aziz Sancar | together with | Tomas Lindahl, Paul L. Modrich`
3. **Reranking** → Top statements about Nobel Prize and Chemistry scored highest
4. **LLM Classification** → `SUPPORTED (90%)` — Evidence confirms the claim

## Resilience Features

- **Retry**: HTTP requests use exponential backoff (3 retries, 0.5s factor)
- **Fallback chains**: Vector Search → Keyword Search, Native API → Textify
- **Graceful degradation**: Missing components return `NOT_ENOUGH_INFO` instead of errors
- **Lazy loading**: Models loaded on first use, not at import time
- **Thread-safe cache**: Label resolution cache uses locking for concurrent access

## Limitations

1. **Wikidata coverage**: Not all facts are in Wikidata; very recent events may be missing
2. **Language**: Best results with English; other languages have varying coverage
3. **LLM reasoning**: Complex multi-hop reasoning may exceed model capability
4. **Vector DB**: Experimental service, may have downtime
5. **Ollama requirement**: Requires local Ollama installation with sufficient RAM (~8GB for 7B model)
