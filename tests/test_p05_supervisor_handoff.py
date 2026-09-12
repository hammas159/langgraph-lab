"""Tests for project 05.

The scoring functions here went through three real false positives while this project was
built — the bare word "peanut" flagging a safe answer, "$1" matching "$100", and "egg" matching
"eggplant" — so each has a regression test, not just a happy-path one.
"""

from __future__ import annotations

import pytest

from projects.p05_supervisor_handoff.cases import BY_ID, CASES, on_topic, violates
from shared.llm import server_is_up

# --- the corpus ----------------------------------------------------------------------------


def test_case_ids_are_unique():
    assert len({c.id for c in CASES}) == len(CASES)


def test_every_case_declares_exactly_one_check_type():
    for c in CASES:
        checks = [bool(c.forbidden), bool(c.must_mention), c.budget_limit is not None]
        assert sum(checks) == 1, f"{c.id} declares {sum(checks)} check types, expected 1"


def test_every_case_has_domain_terms():
    """Without these, a topic-losing response cannot be told apart from a safe one."""
    for c in CASES:
        assert len(c.domain_terms) >= 3, c.id


def test_the_followup_never_repeats_the_constraint():
    """If the followup restates the constraint, the handoff-loss finding is not being tested."""
    for c in CASES:
        assert c.constraint.split()[0].lower() not in c.followup.lower(), c.id


# --- violates(): forbidden-terms cases ------------------------------------------------------


def test_a_safe_response_mentioning_the_allergy_by_name_is_not_a_violation():
    """Regression: the bare word "peanut" flagged a response that correctly said "peanut-free"."""
    c = BY_ID["a1"]
    safe = "Given your peanut allergy, I'd suggest fruit and hummus, both peanut-free."
    assert not violates(safe, c)


def test_recommending_a_forbidden_dish_is_a_violation():
    c = BY_ID["a1"]
    assert violates("A classic choice would be chicken satay skewers.", c)


def test_asana_named_anywhere_is_a_violation():
    c = BY_ID["a3"]
    assert violates("Asana remains a strong choice for a team your size.", c)


# --- violates(): the budget check -----------------------------------------------------------


def test_a_price_under_the_limit_is_not_a_violation():
    """Regression: a banned substring like "$1" also matches "$100", which is not over budget."""
    c = BY_ID["a2"]
    assert not violates("This model is available for around $100.", c)


def test_a_price_over_the_limit_is_a_violation():
    c = BY_ID["a2"]
    assert violates("This premium model runs about $1,200.", c)


def test_a_shorthand_k_price_over_the_limit_is_a_violation():
    c = BY_ID["a2"]
    assert violates("Expect to pay roughly $1.2k for this configuration.", c)


def test_a_shorthand_k_price_under_the_limit_is_not_a_violation():
    c = BY_ID["a2"]
    assert not violates("This comes in at about $0.7k.", c)


# --- violates(): the word-boundary check ------------------------------------------------------


def test_eggplant_is_not_flagged_as_containing_egg():
    """Regression: a bare substring match on "egg" also matched "eggplant", a vegan vegetable."""
    c = BY_ID["a4"]
    assert not violates("Try a grilled eggplant and lentil dish.", c)


def test_a_genuine_egg_dish_is_flagged():
    c = BY_ID["a4"]
    assert violates("A scrambled egg breakfast bowl would work well.", c)


# --- violates(): must_mention -----------------------------------------------------------------


def test_no_mention_of_accessibility_is_a_violation():
    c = BY_ID["a5"]
    assert violates("The coastal resort has a lovely pool and spa.", c)


def test_mentioning_wheelchair_access_satisfies_the_requirement():
    c = BY_ID["a5"]
    assert not violates("This resort has step-free access throughout.", c)


# --- on_topic ----------------------------------------------------------------------------------


def test_a_relevant_response_is_on_topic():
    c = BY_ID["a1"]
    assert on_topic("Here are some great snack options for your event.", c)


def test_a_response_about_something_else_entirely_is_off_topic():
    c = BY_ID["a1"]
    assert not on_topic("Consider booking a large conference room for the event.", c)


def test_off_topic_response_cannot_also_be_scored_as_a_violation_that_matters():
    """The scenario this project is built to catch: silence is not the same as safety."""
    c = BY_ID["a1"]
    off_topic_response = "I'd suggest booking a banquet hall for your event."
    assert not on_topic(off_topic_response, c)
    assert not violates(off_topic_response, c)  # true, but for the wrong reason


# --- live ----------------------------------------------------------------------------------------

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_full_transcript_keeps_the_specialist_on_topic():
    from projects.p05_supervisor_handoff.graph import run
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip("model not pulled")

    c = BY_ID["a1"]
    assert run(c, strategy="full_transcript", model=tag).stayed_on_topic
