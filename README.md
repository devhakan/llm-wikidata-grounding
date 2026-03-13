# LLM Wikidata Grounding

A fact-checking system that grounds claims against Wikidata's structured knowledge base using **vector search**, **cross-encoder reranking**, and **local LLM reasoning**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What It Does

LLMs are powerful but prone to hallucinations. This project verifies claims against Wikidata:

```
Claim → Vector Search → Statement Retrieval → Reranking → Ollama LLM → Verdict
```

| Claim | Verdict | Confidence |
|-------|---------|------------|
| "Aziz Sancar won the Nobel Prize in Chemistry in 2015" | SUPPORTED | 90% |
| "Albert Einstein was born in Germany" | SUPPORTED | 90% |
| "Aziz Sancar won Nobel Prize in Physics in 2020" | REFUTED | 90% |
| "Einstein was born in France" | REFUTED | 90% |

## Installation

**Requirements:** Python 3.10+, [Ollama](https://ollama.ai), ~1GB disk (reranker model)

```bash
git clone https://github.com/devhakan/llm-wikidata-grounding.git
cd llm-wikidata-grounding

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

ollama pull qwen2.5:7b
ollama serve  # if not already running

python verify_setup.py
```

## Usage

```bash
# Single claim
python src/hybrid_pipeline.py "Aziz Sancar won Nobel Prize in Chemistry in 2015"

# Verbose mode
python src/hybrid_pipeline.py -v "Aziz Sancar won Nobel Prize in Chemistry in 2015"

# Interactive mode
python src/hybrid_pipeline.py
```

### Python API

```python
from src.hybrid_pipeline import HybridFactChecker, verify

result = verify("Aziz Sancar won Nobel Prize in Chemistry", verbose=True)
print(result.verdict)      # "SUPPORTED"
print(result.confidence)   # 0.9
print(result.reasoning)    # "Evidence shows..."

checker = HybridFactChecker(model="qwen2.5:7b", verbose=True)
result = checker.check("Al-Khwarizmi lived in the 9th century")
```

## Architecture

1. **Vector Search** — Wikidata's experimental vector database finds semantically relevant entities
2. **Statement Retrieval** — `wbgetentities` API fetches structured claims (~229 per query)
3. **Reranking** — Cross-encoder (`ms-marco-MiniLM-L-6-v2`) filters to top 10 relevant statements
4. **LLM Classification** — Ollama (`qwen2.5:7b`) reasons over evidence and returns a verdict

### Verdict Types

| Verdict | Meaning |
|---------|---------|
| **SUPPORTED** | Evidence confirms the claim |
| **REFUTED** | Evidence contradicts the claim |
| **NOT_ENOUGH_INFO** | Insufficient evidence |

## Running Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## Project Structure

```
llm-wikidata-grounding/
├── src/
│   ├── wikidata_api.py      # Vector search + Wikidata APIs + Label resolution
│   ├── reranker.py          # Cross-Encoder reranking
│   ├── ollama_classifier.py # Ollama LLM classification
│   ├── hybrid_pipeline.py   # Main pipeline
│   └── legacy/              # NLI-based pipeline (alternative)
├── tests/
├── examples/
├── requirements.txt
├── requirements-dev.txt
├── verify_setup.py
└── README.md
```

## Acknowledgments

- **Philippe Saade** (Wikimedia Deutschland) — Wikidata Vector Database & workshop
- **Jonathan Fraine** (Wikimedia) — Wikidata in the AI Web
- **Wikidata community**

## Author

Created by **[User:HakanIST](https://www.wikidata.org/wiki/User:HakanIST)** — Wikimedia volunteer & Wikidata contributor.

- [Wikidata User Page](https://www.wikidata.org/wiki/User:HakanIST)
- [GitHub](https://github.com/devhakan)

## License

MIT License — see [LICENSE](LICENSE)
