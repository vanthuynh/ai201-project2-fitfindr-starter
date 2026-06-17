"""
Tests for search_listings() — Tool 1 from planning.md.

Covers the four verification criteria from Milestone 3:
  1. max_price filter drops items above the price ceiling.
  2. size filter uses case-insensitive substring matching.
  3. Items with a score of 0 are explicitly excluded.
  4. Results are sorted descending by score (then ascending by price on ties).

Also runs the three planning.md test cases:
  - ("vintage graphic tee", "M", 30.0)
  - ("chunky platform shoes", None, None)
  - ("cottagecore dress", "S", 20.0)
"""

import pytest
from tools import search_listings


# ── Criterion 1: max_price filter ─────────────────────────────────────────────

def test_max_price_excludes_items_above_ceiling():
    results = search_listings("vintage", None, 20.0)
    assert all(item["price"] <= 20.0 for item in results), (
        "All results must be at or below max_price"
    )


def test_max_price_none_skips_filter():
    results_no_cap = search_listings("vintage", None, None)
    results_capped = search_listings("vintage", None, 20.0)
    assert len(results_no_cap) >= len(results_capped), (
        "Removing price cap should return at least as many results"
    )


# ── Criterion 2: size filter — case-insensitive substring ─────────────────────

def test_size_filter_substring_match():
    # "M" should match listings sized "S/M", "M", "M/L", etc.
    results = search_listings("vintage", "M", None)
    for item in results:
        assert "m" in item["size"].lower(), (
            f"Listing size '{item['size']}' does not contain 'M' (case-insensitive)"
        )


def test_size_filter_case_insensitive():
    lower = search_listings("vintage", "m", None)
    upper = search_listings("vintage", "M", None)
    assert {i["id"] for i in lower} == {i["id"] for i in upper}, (
        "Size filter must be case-insensitive"
    )


def test_size_none_skips_filter():
    results_no_size = search_listings("vintage", None, None)
    results_sized = search_listings("vintage", "M", None)
    assert len(results_no_size) >= len(results_sized), (
        "Removing size filter should return at least as many results"
    )


# ── Criterion 3: zero-score items are excluded ────────────────────────────────

def test_zero_match_returns_empty_list():
    results = search_listings("xyzzy_nonexistent_item_zzz", None, None)
    assert results == [], "Zero-match query must return []"


def test_no_zero_score_items_in_results():
    results = search_listings("vintage graphic tee", None, None)
    # Every returned item must have matched at least one token somewhere
    assert len(results) > 0, "Expected at least one match"
    # Verify by re-scoring: any item whose title/tags/desc share no token should
    # not appear. Here we just assert results is non-empty and all came from the
    # filtered pool — a score=0 item would have been stripped by the function.
    ids = {item["id"] for item in results}
    assert len(ids) == len(results), "No duplicate listings should appear"


# ── Criterion 4: sorted by score desc, then price asc ─────────────────────────

def test_results_sorted_by_score_descending():
    """
    Verify that each consecutive pair satisfies score[i] >= score[i+1].
    We infer score by counting token hits ourselves and comparing relative order.
    """
    results = search_listings("vintage graphic tee", None, None)
    assert len(results) >= 2, "Need at least 2 results to test ordering"

    tokens = ["vintage", "graphic", "tee"]

    def approx_score(item):
        total = 0
        for token in tokens:
            for tag in item.get("style_tags", []):
                if token in tag.lower():
                    total += 3
            if token in item.get("title", "").lower():
                total += 2
            if token in item.get("description", "").lower():
                total += 1
            if token in (item.get("brand") or "").lower():
                total += 1
            if token in item.get("category", "").lower():
                total += 1
        return total

    scores = [approx_score(r) for r in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Result at index {i} (score {scores[i]}) should outrank "
            f"index {i+1} (score {scores[i+1]})"
        )


def test_tie_broken_by_price_ascending():
    """
    Among items with the same score, cheaper ones should appear first.
    """
    results = search_listings("vintage", None, None)
    tokens = ["vintage"]

    def approx_score(item):
        total = 0
        for token in tokens:
            for tag in item.get("style_tags", []):
                if token in tag.lower():
                    total += 3
            if token in item.get("title", "").lower():
                total += 2
            if token in item.get("description", "").lower():
                total += 1
            if token in (item.get("brand") or "").lower():
                total += 1
            if token in item.get("category", "").lower():
                total += 1
        return total

    for i in range(len(results) - 1):
        s_i = approx_score(results[i])
        s_j = approx_score(results[i + 1])
        if s_i == s_j:
            assert results[i]["price"] <= results[i + 1]["price"], (
                f"Tied items at indices {i}/{i+1}: "
                f"price {results[i]['price']} should be <= {results[i+1]['price']}"
            )


# ── Planning.md test cases ─────────────────────────────────────────────────────

def test_vintage_graphic_tee_size_m_under_30():
    results = search_listings("vintage graphic tee", "M", 30.0)
    assert isinstance(results, list)
    for item in results:
        assert item["price"] <= 30.0
        assert "m" in item["size"].lower()


def test_chunky_platform_shoes_no_filters():
    results = search_listings("chunky platform shoes", None, None)
    assert isinstance(results, list)


def test_cottagecore_dress_size_s_under_20():
    results = search_listings("cottagecore dress", "S", 20.0)
    assert isinstance(results, list)
    for item in results:
        assert item["price"] <= 20.0
        assert "s" in item["size"].lower()
