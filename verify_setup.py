#!/usr/bin/env python3
"""
Setup Verification Script

Verifies that all components are correctly installed:
1. Python version
2. Required packages
3. Wikidata API access
4. ML models
5. Ollama LLM
6. Source modules

Usage:
    python verify_setup.py
"""

import sys
import os


def check_python_version():
    """Verify Python version is 3.10+."""
    print("[1/6] Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"      ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"      ✗ Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_dependencies():
    """Verify required packages are installed."""
    print("[2/6] Checking dependencies...")
    
    packages = {
        "requests": "HTTP client",
        "sentence_transformers": "Cross-Encoder reranking",
        "torch": "PyTorch",
    }
    
    all_ok = True
    for package, desc in packages.items():
        try:
            __import__(package)
            print(f"      ✓ {package} ({desc})")
        except ImportError:
            print(f"      ✗ {package} not installed ({desc})")
            all_ok = False
    
    return all_ok


def check_wikidata():
    """Verify Wikidata API is accessible."""
    print("[3/6] Checking Wikidata API...")
    
    try:
        import requests
        
        params = {
            "action": "wbsearchentities",
            "search": "Albert Einstein",
            "language": "en",
            "limit": 1,
            "format": "json",
        }
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params=params,
            headers={"User-Agent": "LLM-Wikidata-Grounding/1.0 Verify"},
            timeout=10
        )
        response.raise_for_status()
        
        results = response.json().get("search", [])
        if results and results[0]["id"] == "Q937":
            print("      ✓ Wikidata API accessible")
            return True
        else:
            print("      ⚠ Wikidata API responded but unexpected data")
            return True  # Still accessible
            
    except Exception as e:
        print(f"      ✗ Cannot reach Wikidata: {e}")
        return False


def check_vector_search():
    """Check if Wikidata Vector Database is accessible."""
    print("[4/6] Checking vector search...")
    
    try:
        import requests
        
        response = requests.get(
            "https://wd-vectordb.wmcloud.org/item/query",
            params={"query": "test", "lang": "en", "limit": 1},
            headers={"User-Agent": "LLM-Wikidata-Grounding/1.0 Verify"},
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            print("      ✓ Wikidata Vector DB accessible")
            return True
        else:
            print("      ⚠ Vector DB responded but no results")
            return True  # Still accessible
            
    except Exception as e:
        print(f"      ⚠ Vector DB not reachable: {e}")
        print("        (Will fall back to keyword search)")
        return True  # Not critical - fallback available


def check_ollama():
    """Verify Ollama is running and has models available."""
    print("[5/6] Checking Ollama LLM...")
    
    try:
        import requests
        
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        response.raise_for_status()
        
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        if models:
            print(f"      ✓ Ollama is running ({len(models)} model(s) available)")
            for name in model_names[:3]:
                print(f"        • {name}")
            return True
        else:
            print("      ⚠ Ollama is running but no models installed")
            print("        Run: ollama pull qwen2.5:7b")
            return False
            
    except Exception as e:
        print(f"      ✗ Ollama not reachable: {e}")
        print("        Run: ollama serve")
        return False


def check_src_import():
    """Verify src package can be imported (hybrid pipeline modules)."""
    print("[6/6] Checking src package...")
    
    try:
        src_path = os.path.join(os.path.dirname(__file__), "src")
        sys.path.insert(0, src_path)
        
        # Test imports — hybrid pipeline modules
        import wikidata_api
        print("      ✓ wikidata_api module")
        
        import reranker
        print("      ✓ reranker module")
        
        import ollama_classifier
        print("      ✓ ollama_classifier module")
        
        import hybrid_pipeline
        print("      ✓ hybrid_pipeline module")
        
        return True
        
    except ImportError as e:
        print(f"      ✗ Import error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("LLM Wikidata Grounding - Setup Verification")
    print("=" * 60)
    print()
    
    results = [
        check_python_version(),
        check_dependencies(),
        check_wikidata(),
        check_vector_search(),
        check_ollama(),
        check_src_import(),
    ]
    
    print()
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} checks passed! You're ready to go.")
        print()
        print("Try running:")
        print("  python src/hybrid_pipeline.py -v 'Einstein discovered relativity'")
        print()
        print("Or:")
        print("  python src/hybrid_pipeline.py  # interactive mode")
    else:
        print(f"⚠ {passed}/{total} checks passed. Please fix the issues above.")
        print()
        print("Install missing packages:")
        print("  pip install -r requirements.txt")
        print()
        print("Start Ollama:")
        print("  ollama serve")
        print("  ollama pull qwen2.5:7b")
    
    print("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
