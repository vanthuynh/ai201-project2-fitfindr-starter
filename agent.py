"""
agent.py

The FitFindr planning loop. Orchestrates the three tools in response to a
natural language user query, passing state between them via a session dict.

Complete tools.py and test each tool in isolation before implementing this file.

Usage (once implemented):
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import json
import re

from tools import search_listings, suggest_outfit, create_fit_card, call_llm, _MODEL


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.

    You may add fields to this dict as needed for your implementation.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "notice": None,              # set by retry ladder when filters are loosened
        "selected_item": None,       # top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "error": None,               # set if the interaction ended early
    }


# ── query parsing ─────────────────────────────────────────────────────────────

def _regex_parse(query: str) -> dict:
    """Regex fallback: extract price, size, and use raw query as description."""
    price_match = re.search(
        r'\$\s*(\d+(?:\.\d+)?)|under\s+(\d+(?:\.\d+)?)', query, re.IGNORECASE
    )
    max_price = None
    if price_match:
        max_price = float(price_match.group(1) or price_match.group(2))

    size_match = re.search(r'\bsize\s+([A-Za-z0-9/]+)', query, re.IGNORECASE)
    size = size_match.group(1) if size_match else None

    return {"description": query, "size": size, "max_price": max_price}


def _parse_query(query: str) -> dict:
    """
    Extract {description, size, max_price} from the user query.
    Primary: single LLM call returning JSON.
    Fallback: regex parser.
    """
    system_msg = (
        "Extract search parameters from the user's fashion query. "
        "Return ONLY valid JSON with exactly these three fields:\n"
        '{"description": "item keywords", "size": "size string or null", "max_price": number_or_null}\n'
        "Set size/max_price to null if not mentioned. No extra text, no markdown."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]
    try:
        raw = call_llm(messages, model=_MODEL, temperature=0.0)
        # Strip optional markdown code fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:]).rstrip("`").strip()
        parsed = json.loads(raw)
        if "description" not in parsed:
            raise ValueError("missing description field")
        return {
            "description": str(parsed["description"]),
            "size": parsed.get("size") or None,
            "max_price": float(parsed["max_price"]) if parsed.get("max_price") is not None else None,
        }
    except Exception:
        return _regex_parse(query)


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — use get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and the other
        output fields (outfit_suggestion, fit_card) will be None.
    """
    # Step 1: initialise session and parse query
    session = _new_session(query, wardrobe)
    session["parsed"] = _parse_query(query)

    description = session["parsed"]["description"]
    size        = session["parsed"]["size"]
    max_price   = session["parsed"]["max_price"]

    # Step 2: initial search
    results = search_listings(description, size, max_price)

    # Step 3: retry ladder
    if not results:
        # ① Drop price ceiling
        results = search_listings(description, size, None)
        if results:
            price_str = f"${max_price:g}" if max_price is not None else "budget"
            session["notice"] = f"Nothing under {price_str} — showing over-budget matches."
        else:
            # ② Drop price and size
            results = search_listings(description, None, None)
            if results:
                size_str = size if size is not None else "that size"
                session["notice"] = f"Nothing in size {size_str} either — showing all sizes."
            else:
                # ③ Final failure — skip Tools 2 & 3
                session["error"] = (
                    f"No {description} found, even after dropping size and price. "
                    f"Try different keywords."
                )
                return session

    # Step 4: store results and select top match (highest score, cheapest on ties)
    session["search_results"] = results
    session["selected_item"]  = results[0]

    # Step 5: suggest outfit
    print("Verify State 1", session["selected_item"])
    session["outfit_suggestion"] = suggest_outfit(session["selected_item"], wardrobe)

    # Step 6: create fit card
    print("Verify State 2", session["outfit_suggestion"])
    session["fit_card"] = create_fit_card(
        session["outfit_suggestion"], session["selected_item"]
    )

    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        print(f"Found: {session['selected_item']['title']}")
        print(f"\nOutfit: {session['outfit_suggestion']}")
        print(f"\nFit card: {session['fit_card']}")

    print("\n\n=== No-results path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error message: {session2['error']}")
