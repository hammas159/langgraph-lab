"""Tests for project 02.

The ticket-corpus tests matter most: a cross-cutting ticket must genuinely have more than one
defensible category, or the "confident misroute" finding would just be measuring a mislabelled
ground truth. The parser tests pin the format the router is asked to reply in.
"""

from __future__ import annotations

import pytest

from projects.p02_router_misroute.graph import RunResult, parse
from projects.p02_router_misroute.tickets import (
    CATEGORIES,
    TICKETS,
    Ticket,
    clean,
    cross_cutting,
    is_acceptable,
)
from shared.llm import server_is_up

# --- the corpus ----------------------------------------------------------------------------


def test_ticket_ids_are_unique():
    assert len({t.id for t in TICKETS}) == len(TICKETS)


def test_every_primary_is_a_valid_category():
    for t in TICKETS:
        assert t.primary in CATEGORIES, t.id


def test_clean_tickets_have_no_secondary_category():
    for t in clean():
        assert t.also == (), t.id


def test_cross_cutting_tickets_have_at_least_one_secondary_category():
    for t in cross_cutting():
        assert len(t.also) >= 1, t.id


def test_cross_cutting_secondary_categories_are_valid_and_distinct_from_primary():
    for t in cross_cutting():
        for c in t.also:
            assert c in CATEGORIES, t.id
            assert c != t.primary, f"{t.id} lists its own primary as a secondary"


def test_there_are_both_clean_and_cross_cutting_tickets():
    assert len(clean()) >= 4
    assert len(cross_cutting()) >= 4


def test_is_acceptable_matches_primary_or_a_secondary():
    t = next(t for t in cross_cutting())
    assert is_acceptable(t, t.primary)
    if t.also:
        assert is_acceptable(t, t.also[0])


def test_is_acceptable_rejects_an_unrelated_category():
    t = Ticket(id="z", text="x", primary="billing", kind="clean")
    assert not is_acceptable(t, "technical")


# --- parsing -------------------------------------------------------------------------------


def test_parses_category_and_confidence():
    c = parse("CATEGORY: billing\nCONFIDENCE: HIGH")
    assert c.category == "billing"
    assert c.confidence == "HIGH"


def test_parsing_is_case_insensitive():
    c = parse("category: Technical\nconfidence: low")
    assert c.category == "technical"
    assert c.confidence == "LOW"


def test_an_invalid_category_is_dropped_rather_than_kept():
    """A category outside the fixed set must not be silently trusted."""
    c = parse("CATEGORY: shipping\nCONFIDENCE: HIGH")
    assert c.category == ""


def test_an_invalid_confidence_is_dropped():
    c = parse("CATEGORY: billing\nCONFIDENCE: SURE")
    assert c.confidence == ""


def test_unparseable_text_yields_empty_fields():
    c = parse("I'm not sure how to categorise this.")
    assert c.category == ""
    assert c.confidence == ""


# --- RunResult derived properties -----------------------------------------------------------


def test_correct_when_decision_matches_primary():
    t = TICKETS[0]
    r = RunResult(ticket=t, strategy="x", decision=t.primary)
    assert r.correct


def test_escalated_when_decision_is_review():
    t = TICKETS[0]
    r = RunResult(ticket=t, strategy="x", decision="review")
    assert r.escalated
    assert not r.correct


def test_wrong_and_confident_requires_all_three_conditions():
    t = TICKETS[0]
    wrong_category = next(c for c in CATEGORIES if c != t.primary)

    confident_wrong = RunResult(
        ticket=t, strategy="x", decision=wrong_category, top_confidence="HIGH"
    )
    assert confident_wrong.wrong_and_confident

    unconfident_wrong = RunResult(
        ticket=t, strategy="x", decision=wrong_category, top_confidence="LOW"
    )
    assert not unconfident_wrong.wrong_and_confident

    confident_right = RunResult(ticket=t, strategy="x", decision=t.primary, top_confidence="HIGH")
    assert not confident_right.wrong_and_confident

    escalated = RunResult(ticket=t, strategy="x", decision="review", top_confidence="HIGH")
    assert not escalated.wrong_and_confident, "escalating is not a confident misroute"


# --- live ----------------------------------------------------------------------------------------

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_single_pass_routes_a_clean_ticket_correctly():
    from projects.p02_router_misroute.graph import run
    from projects.p02_router_misroute.tickets import BY_ID
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip("model not pulled")

    t = BY_ID["c1"]  # unambiguous double-charge -> billing
    assert run(t, strategy="single_pass", model=tag).correct
