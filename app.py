"""
Flask Web Application for LLM Wikidata Grounding

A visual demo interface for the fact-checking pipeline.
Run with: python app.py
"""

import re
import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request, jsonify

from hybrid_pipeline import HybridFactChecker, FactCheckResult

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Regex patterns for cleaning evidence format
_QID_PATTERN = re.compile(r'\s*\(Q\d+\)')
_PID_PATTERN = re.compile(r'\s*\(P\d+\)')
_QID_EXTRACT = re.compile(r'\(Q(\d+)\)')


def format_evidence(raw: str) -> dict:
    """Parse raw evidence string into structured, human-readable format.
    
    Input:  'Albert Einstein (Q937) | date of death (P570) | 1955-04-18'
    Output: {'entity': 'Albert Einstein', 'property': 'date of death',
             'value': '1955-04-18', 'entity_url': 'https://www.wikidata.org/wiki/Q937'}
    """
    parts = [p.strip() for p in raw.split('|')]
    if len(parts) >= 3:
        entity_raw = parts[0]
        prop_raw = parts[1]
        value_raw = ' | '.join(parts[2:])  # rejoin if value had pipes
    elif len(parts) == 2:
        entity_raw = parts[0]
        prop_raw = ''
        value_raw = parts[1]
    else:
        return {'entity': raw, 'property': '', 'value': '', 'entity_url': ''}
    
    # Extract QID for link
    qid_match = _QID_EXTRACT.search(entity_raw)
    entity_url = f'https://www.wikidata.org/wiki/Q{qid_match.group(1)}' if qid_match else ''
    
    # Strip Q/P IDs from display text
    entity = _QID_PATTERN.sub('', entity_raw).strip()
    prop = _PID_PATTERN.sub('', prop_raw).strip()
    value = _QID_PATTERN.sub('', value_raw).strip()
    
    return {'entity': entity, 'property': prop, 'value': value, 'entity_url': entity_url}


# Register as Jinja filter
app.jinja_env.filters['format_evidence'] = format_evidence

# Lazy-loaded checker
_checker = None


def get_checker():
    """Get or create the fact checker instance."""
    global _checker
    if _checker is None:
        logger.info("Initializing fact checker (loading models)...")
        _checker = HybridFactChecker(verbose=False)
        logger.info("Fact checker ready.")
    return _checker


# Sample claims for the demo
SAMPLE_CLAIMS = [
    {
        "claim": "Aziz Sancar won the Nobel Prize in Chemistry in 2015",
        "hint": "🧪 Turkish scientist — should be SUPPORTED",
    },
    {
        "claim": "Rembrandt was born in Amsterdam",
        "hint": "🎨 Actually born in Leiden — should be REFUTED",
    },
    {
        "claim": "Marie Curie won two Nobel Prizes",
        "hint": "⚛️ Physics (1903) and Chemistry (1911) — should be SUPPORTED",
    },
    {
        "claim": "The Mona Lisa is displayed in the Rijksmuseum",
        "hint": "🖼️ It's in the Louvre — should be REFUTED",
    },
    {
        "claim": "Albert Einstein was born in Ulm",
        "hint": "🔬 Born in Ulm, Germany — should be SUPPORTED",
    },
    {
        "claim": "Leonardo da Vinci invented the telephone",
        "hint": "📞 Alexander Graham Bell — should be REFUTED",
    },
]


@app.route("/")
def index():
    """Main page with claim input form."""
    return render_template("index.html", samples=SAMPLE_CLAIMS)


@app.route("/check", methods=["POST"])
def check_claim():
    """Check a claim and return results."""
    claim = request.form.get("claim", "").strip()

    if not claim:
        return render_template("index.html", samples=SAMPLE_CLAIMS, error="Please enter a claim.")

    try:
        checker = get_checker()
        start_time = time.time()
        result = checker.check(claim)
        elapsed = time.time() - start_time

        return render_template(
            "index.html",
            samples=SAMPLE_CLAIMS,
            result=result,
            elapsed=f"{elapsed:.1f}",
            claim=claim,
        )
    except Exception as e:
        logger.error("Error checking claim: %s", e)
        return render_template(
            "index.html",
            samples=SAMPLE_CLAIMS,
            error=f"Error: {e}",
            claim=claim,
        )


@app.route("/api/check", methods=["GET", "POST"])
def api_check():
    """JSON API endpoint for programmatic use."""
    if request.method == "GET":
        claim = request.args.get("claim", "").strip()
    else:
        data = request.get_json(silent=True) or {}
        claim = data.get("claim", request.form.get("claim", "")).strip()

    if not claim:
        return jsonify({"error": "Missing 'claim' parameter"}), 400

    try:
        checker = get_checker()
        start_time = time.time()
        result = checker.check(claim)
        elapsed = time.time() - start_time

        return jsonify({
            "claim": result.claim,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "evidence": result.evidence[:5],
            "entities_found": result.entities_found,
            "elapsed_seconds": round(elapsed, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LLM Wikidata Grounding — Web Demo")
    print("  http://localhost:5001")
    print("=" * 60 + "\n")
    # debug=False + use_reloader=False to prevent loading models twice
    app.run(debug=False, host="0.0.0.0", port=5001)
