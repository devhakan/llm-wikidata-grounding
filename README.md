# LLM Wikidata Grounding

A fact-checking system that grounds claims against Wikidata's structured knowledge base using **vector search**, **cross-encoder reranking**, and **local LLM reasoning**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What This Project Does

LLMs are powerful but prone to **hallucinations**. This project verifies claims against Wikidata using a hybrid pipeline:

```
Claim → Vector Search → Statement Retrieval → Reranking → Ollama LLM → Verdict
```

### Example Results (Hybrid Pipeline)

| Claim | Verdict | Confidence |
|-------|---------|------------|
| "Aziz Sancar won the Nobel Prize in Chemistry in 2015" | ✓ SUPPORTED | **90%** |
| "Albert Einstein was born in Germany" | ✓ SUPPORTED | 90% |
| "Özlem Türeci is the co-founder of BioNTech" | ✓ SUPPORTED | 90% |
| "Al-Khwarizmi lived in the 9th century" | ✓ SUPPORTED | 90% |
| "Aziz Sancar won Nobel Prize in Physics in 2020" | ✗ REFUTED | 90% |
| "Einstein was born in France" | ✗ REFUTED | 90% |

> **Note**: Using local Ollama LLM (qwen2.5:7b) provides much higher confidence than NLI models (~58-76%).


---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLAIM: "Aziz Sancar won Nobel Prize in Chemistry in 2015"         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. VECTOR SEARCH                                                    │
│     API: wd-vectordb.wmcloud.org                                     │
│     Result: Q15118973 (Aziz Sancar), Q44585 (Nobel Chemistry)...     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. STATEMENT RETRIEVAL                                              │
│     API: wikidata.org/w/api.php (wbgetentities)                      │
│     Result: ~229 statements about matched entities                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. CROSS-ENCODER RERANKING                                          │
│     Model: cross-encoder/ms-marco-MiniLM-L-6-v2                      │
│     Result: Top 10 relevant statements                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. OLLAMA LLM REASONING                                             │
│     Model: qwen2.5:7b (local)                                        │
│     Natural language understanding of evidence vs claim              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VERDICT: ✓ SUPPORTED (90%)                                          │
│  "Evidence clearly states Aziz Sancar won Nobel Prize in 2015"       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) with a model installed (e.g., `qwen2.5:7b`)
- ~1GB disk space (for reranker model)

### Quick Start

```bash
# Clone
git clone https://github.com/devhakan/llm-wikidata-grounding.git
cd llm-wikidata-grounding

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Ensure Ollama is running with a model
ollama pull qwen2.5:7b
ollama serve  # if not already running

# Verify
python verify_setup.py
```

---

## 🚀 Usage

### Command Line (Hybrid Pipeline)

```bash
# Single claim
python src/hybrid_pipeline.py "Aziz Sancar won Nobel Prize in Chemistry"

# Verbose mode
python src/hybrid_pipeline.py -v "Aziz Sancar won Nobel Prize in Chemistry in 2015"

# Interactive mode
python src/hybrid_pipeline.py
```

### Python API

```python
from src.hybrid_pipeline import HybridFactChecker, verify

# Quick check
result = verify("Özlem Türeci is the co-founder of BioNTech", verbose=True)
print(result.verdict)      # "SUPPORTED"
print(result.confidence)   # 0.9
print(result.reasoning)    # "Evidence shows..."

# With custom model
checker = HybridFactChecker(model="qwen2.5:7b", verbose=True)
result = checker.check("Al-Khwarizmi lived in the 9th century")
```

---

## 🔬 Components

### 1. Vector Search
Semantic search using Wikidata's experimental vector database:
```python
from src.wikidata_api import vector_search
results = vector_search("Turkish scientist Nobel Prize")
# → [{"id": "Q15118973", "score": 0.86}, ...]
```

### 2. Label Resolution
Batch resolution of Wikidata IDs to human-readable labels:
```python
from src.wikidata_api import resolve_labels
labels = resolve_labels(["P106", "Q169470"])
# → {"P106": "occupation", "Q169470": "physicist"}
```

### 3. Cross-Encoder Reranking
Filter hundreds of statements to the most relevant:
```python
from src.reranker import Reranker
reranker = Reranker()
ranked = reranker.rerank(claim, statements, top_k=10)
```

### 4. Ollama LLM Classification
Natural language reasoning with local LLM:
```python
from src.ollama_classifier import OllamaClassifier
classifier = OllamaClassifier(model="qwen2.5:7b")
result = classifier.classify(claim, evidence)
```

---

## 📊 Verdict Types

| Verdict | Meaning |
|---------|---------|
| ✓ **SUPPORTED** | Evidence confirms the claim |
| ✗ **REFUTED** | Evidence contradicts the claim |
| ? **NOT_ENOUGH_INFO** | Insufficient evidence |

---

## ⚙️ Configuration

### Ollama Models

Any Ollama model works. Tested with:
- `qwen2.5:7b` (recommended)
- `llama3.2`
- `mistral`

```bash
# Pull a model
ollama pull qwen2.5:7b

# Use in code
checker = HybridFactChecker(model="qwen2.5:7b")
```

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src
```

---

## 📁 Project Structure

```
llm-wikidata-grounding/
├── src/
│   ├── wikidata_api.py      # Vector search + Wikidata APIs + Label resolution
│   ├── reranker.py          # Cross-Encoder reranking
│   ├── ollama_classifier.py # Ollama LLM classification
│   ├── hybrid_pipeline.py   # Main hybrid pipeline ⭐
│   └── legacy/              # NLI-based pipeline (alternative)
│       ├── pipeline.py
│       └── nli_classifier.py
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_wikidata_api.py
│   ├── test_reranker.py
│   ├── test_ollama_classifier.py
│   └── test_hybrid_pipeline.py
├── examples/
│   └── basic_example.py
├── requirements.txt
├── requirements-dev.txt
├── verify_setup.py
└── README.md
```

---

## 🙏 Acknowledgments

- **Philippe Saade** (Wikimedia Deutschland) - Wikidata Vector Database & workshop
- **Jonathan Fraine** (Wikimedia) - Wikidata in the AI Web
- **Wikidata community** - Maintaining the knowledge base

---

## 👤 Author

Created by **[User:HakanIST](https://www.wikidata.org/wiki/User:HakanIST)** - Wikimedia volunteer & Wikidata contributor.

- 🌐 [Wikidata User Page](https://www.wikidata.org/wiki/User:HakanIST)
- 💻 [GitHub](https://github.com/devhakan)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)
