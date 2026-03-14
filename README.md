# LLM Wikidata Grounding

A fact-checking system that grounds claims against Wikidata's structured knowledge base using **vector search**, **cross-encoder reranking**, and **local LLM reasoning**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Wikimedia Hackathon NWE 2026](https://img.shields.io/badge/Wikimedia_Hackathon-NWE_2026-006699.svg)](https://www.mediawiki.org/wiki/Wikimedia_Hackathon_Northwestern_Europe_2026)

## What It Does

LLMs are powerful but prone to hallucinations. This project verifies claims against Wikidata:

```
Claim → Vector Search → Statement Retrieval → Reranking → Ollama LLM → Verdict
```

| Claim | Verdict | Confidence |
|-------|---------|------------|
| "Aziz Sancar won the Nobel Prize in Chemistry in 2015" | ✓ SUPPORTED | 90% |
| "Albert Einstein was born in Ulm" | ✓ SUPPORTED | 90% |
| "Rembrandt was born in Amsterdam" | ✗ REFUTED | 90% |
| "Leonardo da Vinci invented the telephone" | ✗ REFUTED | 90% |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) with `qwen2.5:7b` model (~4.7GB)
- ~1GB disk for the reranker model (auto-downloaded on first run)

### Installation

```bash
git clone https://github.com/devhakan/llm-wikidata-grounding.git
cd llm-wikidata-grounding

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Pull the LLM model
ollama pull qwen2.5:7b
ollama serve  # if not already running

# Verify setup
python verify_setup.py
```

## Web Demo

A visual interface built with the [Wikimedia Codex](https://doc.wikimedia.org/codex/) design system:

```bash
source venv/bin/activate
python app.py
# → http://localhost:5001
```

**Features:**
- Interactive claim input with sample queries
- Color-coded verdict badges (✓ green / ✗ red / ? yellow)
- Confidence bar and timing display
- Wikidata evidence viewer with source statements
- Pipeline visualization (Vector Search → Rerank → LLM → Verdict)
- JSON API endpoint at `/api/check`

### JSON API

```bash
# GET request
curl "http://localhost:5001/api/check?claim=Marie+Curie+won+two+Nobel+Prizes"

# POST request
curl -X POST http://localhost:5001/api/check \
  -H "Content-Type: application/json" \
  -d '{"claim": "Marie Curie won two Nobel Prizes"}'
```

Response:
```json
{
  "claim": "Marie Curie won two Nobel Prizes",
  "verdict": "SUPPORTED",
  "confidence": 0.9,
  "reasoning": "Evidence confirms Marie Curie received Nobel Prizes in Physics (1903) and Chemistry (1911).",
  "evidence": ["Marie Curie (Q7186) | award received (P166) | Nobel Prize in Physics (Q38104)", "..."],
  "entities_found": ["Q7186"],
  "elapsed_seconds": 8.42
}
```

## Command-Line Usage

```bash
# Single claim
python src/hybrid_pipeline.py "Aziz Sancar won Nobel Prize in Chemistry in 2015"

# Verbose mode (shows pipeline stages)
python src/hybrid_pipeline.py -v "Aziz Sancar won Nobel Prize in Chemistry in 2015"

# Interactive mode
python src/hybrid_pipeline.py
```

### Python API

```python
from src.hybrid_pipeline import HybridFactChecker, verify

# Quick function
result = verify("Aziz Sancar won Nobel Prize in Chemistry", verbose=True)
print(result.verdict)      # "SUPPORTED"
print(result.confidence)   # 0.9
print(result.reasoning)    # "Evidence shows..."

# Reusable checker instance
checker = HybridFactChecker(model="qwen2.5:7b", verbose=True)
result = checker.check("Rembrandt was born in Amsterdam")
```

## Architecture

1. **Vector Search** — Wikidata's experimental vector database finds semantically relevant entities
2. **Statement Retrieval** — `wbgetentities` API fetches structured claims (~200+ per entity)
3. **Reranking** — Cross-encoder (`ms-marco-MiniLM-L-6-v2`) filters to top 10 relevant statements
4. **LLM Classification** — Ollama (`qwen2.5:7b`) reasons over evidence and returns a verdict

For detailed technical documentation, see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

### Verdict Types

| Verdict | Meaning |
|---------|---------|
| **SUPPORTED** | Evidence confirms the claim |
| **REFUTED** | Evidence contradicts the claim |
| **NOT_ENOUGH_INFO** | Insufficient evidence to determine |

### Resilience Features

- **Retry with backoff** — HTTP requests auto-retry (3 attempts, exponential backoff)
- **Fallback chains** — Vector Search → Keyword Search, Native API → Textify API
- **Graceful degradation** — Missing components return `NOT_ENOUGH_INFO` instead of crashing
- **Lazy loading** — Models loaded on first use, not at import time
- **Thread-safe cache** — Label resolution cache with locking for concurrent access
- **Robust verdict parsing** — Multi-strategy LLM response parsing (exact format, keywords, reasoning inference)

## Running Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

51 tests covering all pipeline stages, API interactions, and edge cases.

## Project Structure

```
llm-wikidata-grounding/
├── app.py                  # Flask web demo (Codex design)
├── templates/
│   └── index.html          # Jinja2 template (Codex components)
├── static/
│   └── style.css           # Wikimedia Codex design tokens
├── src/
│   ├── wikidata_api.py     # Vector search + Wikidata APIs + label cache
│   ├── reranker.py         # Cross-encoder reranking with sigmoid normalization
│   ├── ollama_classifier.py# Ollama LLM classification + robust parsing
│   ├── hybrid_pipeline.py  # Main 4-stage pipeline orchestrator
│   └── legacy/             # NLI-based pipeline (alternative approach)
├── tests/                  # 51 tests (pytest)
├── examples/               # Usage examples
├── docs/
│   └── HOW_IT_WORKS.md     # Detailed technical documentation
├── pyproject.toml          # Python packaging & tool config
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Dev/test dependencies
└── verify_setup.py         # Setup verification script
```

## Acknowledgments

- **Philippe Saade** (Wikimedia Deutschland) — Wikidata Vector Database & workshop
- **Jonathan Fraine** (Wikimedia) — Wikidata in the AI Web
- **Wikidata community** — Structured knowledge that makes this possible

## Author

Created by **[User:HakanIST](https://www.wikidata.org/wiki/User:HakanIST)** — Wikimedia volunteer & Wikidata contributor.

Developed for the [Wikimedia Hackathon Northwestern Europe 2026](https://www.mediawiki.org/wiki/Wikimedia_Hackathon_Northwestern_Europe_2026) (Arnhem, 13–14 March 2026).

- [Wikidata User Page](https://www.wikidata.org/wiki/User:HakanIST)
- [GitHub](https://github.com/devhakan)

## License

MIT License — see [LICENSE](LICENSE)
