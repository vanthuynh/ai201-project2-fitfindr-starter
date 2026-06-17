"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os
import time

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── LLM helper ────────────────────────────────────────────────────────────────

_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def call_llm(messages: list[dict], model: str = _MODEL, temperature: float = 0.7) -> str:
    """
    Call the Groq chat completion API with up to 3 attempts (exponential backoff).
    Returns the response content string on success.
    Raises the last exception after all retries are exhausted.
    """
    client = _get_groq_client()
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1 s, then 2 s
    raise last_exc  # type: ignore[misc]


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    listings = load_listings()

    # Apply filters
    if max_price is not None:
        listings = [l for l in listings if l["price"] <= max_price]
    if size is not None:
        listings = [
            l for l in listings
            if size.lower() in l["size"].lower()
        ]

    # Tokenize description by whitespace
    tokens = description.lower().split()

    def score(listing: dict) -> int:
        total = 0
        for token in tokens:
            # style_tags: 3x weight
            for tag in listing.get("style_tags", []):
                if token in tag.lower():
                    total += 3
            # title: 2x weight
            if token in listing.get("title", "").lower():
                total += 2
            # description, brand, category: 1x weight each
            if token in listing.get("description", "").lower():
                total += 1
            if token in (listing.get("brand") or "").lower():
                total += 1
            if token in listing.get("category", "").lower():
                total += 1
        return total

    scored = [(score(l), l) for l in listings]
    scored = [(s, l) for s, l in scored if s > 0]
    scored.sort(key=lambda x: (-x[0], x[1]["price"]))

    return [l for _, l in scored]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    item_line = (
        f"{new_item.get('title', 'item')} "
        f"({new_item.get('category', '')}; "
        f"{', '.join(new_item.get('style_tags', []))}; "
        f"colors: {', '.join(new_item.get('colors', []))})"
    )

    wardrobe_items = wardrobe.get("items", [])

    if not wardrobe_items:
        prompt = (
            f"You are a personal stylist specializing in thrifted fashion.\n\n"
            f"A user just found this thrifted item: {item_line}\n\n"
            f"They have not added any wardrobe items yet. "
            f"Suggest 1-2 complete outfits they could build around this piece using "
            f"common wardrobe staples. Do not invent or assume specific items they own. "
            f"Be specific about what styles and types of pieces complement it and why."
        )
    else:
        wardrobe_lines = "\n".join(
            f"- {w['name']} ({w['category']}; {', '.join(w.get('style_tags', []))}; "
            f"colors: {', '.join(w.get('colors', []))})"
            for w in wardrobe_items
        )
        prompt = (
            f"You are a personal stylist specializing in thrifted fashion.\n\n"
            f"A user just found this thrifted item: {item_line}\n\n"
            f"Their existing wardrobe includes:\n{wardrobe_lines}\n\n"
            f"Suggest 1-2 complete outfits that pair the new item with specific named "
            f"pieces from their wardrobe above. Reference each wardrobe piece by name."
        )

    messages = [{"role": "user", "content": prompt}]
    try:
        result = call_llm(messages, model=_MODEL)
        if not result.strip():
            return (
                "Couldn't generate outfit suggestions right now. "
                "The item still looks great — try pairing it with basics in similar colors."
            )
        return result
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code:
            return f"Groq error {status_code}"
        return (
            "Couldn't generate outfit suggestions right now. "
            "The item still looks great — try pairing it with basics in similar colors."
        )


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    if not outfit or not outfit.strip():
        return "[fit card unavailable — outfit suggestion was empty]"

    title = new_item.get("title", "this item")
    price = new_item.get("price", "")
    platform = new_item.get("platform", "")

    prompt = (
        f"You are writing a casual, authentic OOTD caption for social media.\n\n"
        f"The thrifted item is: {title} — ${price} on {platform}\n\n"
        f"The outfit suggestion is:\n{outfit}\n\n"
        f"Write a 2–4 sentence caption that:\n"
        f"- Captures the specific vibe of this outfit\n"
        f"- Mentions the item name, price (${price}), and platform ({platform}) exactly once each\n"
        f"- Sounds like a real person posting an OOTD, not a product description\n"
        f"- Is casual, specific, and energetic\n\n"
        f"Return only the caption text, nothing else."
    )

    messages = [{"role": "user", "content": prompt}]
    try:
        result = call_llm(messages, model=_MODEL, temperature=0.9)
        if not result.strip():
            return "[fit card unavailable — LLM error]"
        return result
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code:
            return f"Groq error {status_code}"
        return "[fit card unavailable — LLM error]"
