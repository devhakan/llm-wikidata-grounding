# LLM Wikidata Grounding

A fact-checking system that grounds claims against Wikidata's structured knowledge base using vector search, cross-encoder reranking, and NLI classification.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What This Project Does

LLMs are powerful but prone to **hallucinations** — generating plausible-sounding but incorrect information. This project verifies claims against Wikidata using a multi-stage pipeline:

```
Claim → Vector Search → Statement Retrieval → Reranking → NLI → Verdict
```

### Example Results

| Claim | Verdict | Confidence |
|-------|---------|------------|
| "Ibn al-Haytham was born in Basra" | ✓ SUPPORTED | 76% |
| "Aziz Sancar won the Nobel Prize in Chemistry in 2015" | ✓ SUPPORTED | 58% |
| "Özlem Türeci is the co-founder of BioNTech" | ✓ SUPPORTED | 73% |
| "Al-Khwarizmi lived in the 9th century" | ✓ SUPPORTED | 73% |

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
│     Query: wd-vectordb.wmcloud.org                                   │
│     Result: Q15118973 (Aziz Sancar), Q44585 (Nobel Chemistry)...     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. STATEMENT RETRIEVAL                                              │
│     API: wd-textify.toolforge.org                                    │
│     Result: 1225 statements about matched entities                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. CROSS-ENCODER RERANKING                                          │
│     Model: cross-encoder/ms-marco-MiniLM-L-6-v2                      │
│     Result: Top 10 relevant statements (score: 1.00)                 │
│       • "Aziz Sancar | award received | Nobel Prize in Chemistry"    │
│       • "Aziz Sancar | description | Nobel Prize Chemistry 2015"     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. NLI CLASSIFICATION                                               │
│     Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli                   │
│     Premise: "Aziz Sancar received Nobel Prize Chemistry 2015"       │
│     Hypothesis: "Aziz Sancar won Nobel Prize Chemistry in 2015"      │
│     Result: ENTAILMENT (58% confidence)                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VERDICT: ✓ SUPPORTED                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- ~4GB disk space (for ML models)
- ~8GB RAM recommended

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

# Verify
python verify_setup.py
```

> **Note**: First run downloads ~650MB of ML models. Subsequent runs use cached models.

---

## 🚀 Usage

### Command Line

```bash
# Single claim
python -m src.pipeline "Aziz Sancar won Nobel Prize in Chemistry"

# Verbose mode
python -m src.pipeline -v "Ibn al-Haytham was born in Basra"

# Interactive mode
python -m src.pipeline
```

### Python API

```python
from src import FactChecker, verify

# Quick check
result = verify("Özlem Türeci is the co-founder of BioNTech")
print(result.verdict)      # VerificationResult.SUPPORTED
print(result.confidence)   # 0.73
print(result.evidence)     # ["Özlem Türeci | affiliation | BioNTech", ...]

# With options
checker = FactChecker(verbose=True)
result = checker.check("Al-Khwarizmi lived in the 9th century")
```

### Example Script

```bash
python examples/basic_example.py
```

---

## 🔬 Components

### 1. Vector Search

Semantic search using Wikidata's experimental vector database:

```python
from src.wikidata_api import vector_search

# Find entities semantically similar to the query
results = vector_search("Turkish scientist who won Nobel Prize")
# → [{"id": "Q15118973", "score": 0.86}, ...]  # Aziz Sancar
```

**API**: `wd-vectordb.wmcloud.org` (no API key required)

### 2. Cross-Encoder Reranking

Filter thousands of statements to the most relevant ones:

```python
from src.reranker import Reranker

reranker = Reranker()
ranked = reranker.rerank(
    claim="Sancar won Nobel Prize",
    statements=all_statements,  # 1000+ statements
    top_k=10                    # Keep top 10
)
# → [RankedStatement(text="Sancar | award | Nobel Prize", score=1.0), ...]
```

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB)

### 3. NLI Classification

Determine if evidence supports or contradicts the claim:

```python
from src.nli_classifier import NLIClassifier

classifier = NLIClassifier()
result = classifier.classify(
    premise="Ibn al-Haytham's place of birth is Basra",
    hypothesis="Ibn al-Haytham was born in Basra"
)
# → ClassificationResult(verdict=ENTAILMENT, confidence=0.76)
```

**Model**: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (~558MB, multilingual)

---

## 📊 Verdict Types

| Verdict | Meaning | NLI Result |
|---------|---------|------------|
| ✓ **SUPPORTED** | Evidence confirms the claim | ENTAILMENT |
| ✗ **REFUTED** | Evidence contradicts the claim | CONTRADICTION |
| ? **NOT_ENOUGH_INFO** | Can't verify or refute | NEUTRAL |

---

## 🌍 Multilingual Support

The NLI model (`mDeBERTa`) supports 100+ languages:

```python
# Turkish
verify("Aziz Sancar 2015'te Nobel Kimya Ödülü kazandı")

# German
verify("Einstein entdeckte die Relativitätstheorie")

# Arabic
verify("ابن الهيثم ولد في البصرة")
```

---

## ⚙️ Configuration

### Custom Models

```python
checker = FactChecker(
    reranker_model="BAAI/bge-reranker-base",           # Better quality
    nli_model="microsoft/deberta-v3-large-mnli",       # Higher accuracy
    verbose=True
)
```

### Environment Variables

```bash
# Force CPU (if GPU issues)
CUDA_VISIBLE_DEVICES=-1

# HuggingFace token (optional, for faster downloads)
HF_TOKEN=your_token
```

---

## 📁 Project Structure

```
llm-wikidata-grounding/
├── src/
│   ├── wikidata_api.py    # Vector search + Wikidata APIs
│   ├── reranker.py        # Cross-Encoder reranking
│   ├── nli_classifier.py  # NLI classification
│   ├── pipeline.py        # Main fact-checking pipeline
│   └── __init__.py
├── examples/
│   └── basic_example.py
├── docs/
│   └── HOW_IT_WORKS.md
├── requirements.txt
├── verify_setup.py
└── README.md
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/devhakan/llm-wikidata-grounding.git
cd llm-wikidata-grounding
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python verify_setup.py
```

---

## 📚 Resources

- [Wikidata Vector Database](https://www.wikidata.org/wiki/Wikidata:Vector_Database)
- [Wikidata Query Service](https://query.wikidata.org/)
- [Cross-Encoders (SBERT)](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [NLI with Transformers](https://huggingface.co/tasks/text-classification)

---

## 🙏 Acknowledgments

- **Philippe Saade** (Wikimedia Deutschland) - Wikidata Vector Database & workshop
- **Jonathan Fraine** (Wikimedia) - [Wikidata in the AI Web](https://commons.wikimedia.org/wiki/File:Wikidata_in_the_AI_Web_-_Lightning_Talks_Futures_Lab.pdf)
- **Wikidata community** - Maintaining the knowledge base
- **HuggingFace** - Pre-trained models

---

## � Author

Created by **[User:HakanIST](https://www.wikidata.org/wiki/User:HakanIST)** - Wikimedia volunteer & Wikidata contributor.

- 🌐 [Wikidata User Page](https://www.wikidata.org/wiki/User:HakanIST)
- 💻 [GitHub](https://github.com/devhakan)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## ⚠️ Limitations

- **First run**: Downloads ~650MB of models
- **Speed**: 5-15 seconds per claim (includes API calls)
- **Coverage**: Not all facts are in Wikidata
- **Confidence**: Lower confidence ≠ wrong, may just need more evidence
