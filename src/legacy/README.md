# Legacy NLI Pipeline

This folder contains the original NLI-based pipeline that has been replaced 
by the hybrid approach (Ollama LLM).

## Files

- `pipeline.py` - Original NLI-based fact-checking pipeline
- `nli_classifier.py` - Transformer-based NLI classifier

## Why Deprecated?

The NLI approach had limitations:
- Low confidence scores (58-76%)
- Required ~650MB model download
- Slower inference

The hybrid approach (`hybrid_pipeline.py`) uses:
- Vector Search + Reranker + Ollama LLM
- Higher confidence (90%)
- Uses existing local Ollama setup
