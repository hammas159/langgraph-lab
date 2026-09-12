"""Tests for project 04.

The `_book` registry is the one thing this project's numbers actually depend on being correct.
An earlier version put a live `Ledger` inside checkpointed graph state; the checkpointer
serialises state between invocations, so every resume operated on a disconnected deserialised
copy and the real call count was silently under-reported. The registry keyed by a plain string
is what a benchmark can trust, and `test_book_returns_the_same_instance_for_the_same_run_id`
is the regression test for that.
"""

from __future__ import annotations

import pytest

from projects.p04_checkpoint_resume.graph import _book
from projects.p04_checkpoint_resume.scenarios import BY_ID, SCENARIOS
from shared.llm import server_is_up

# --- the corpus ----------------------------------------------------------------------------


def test_scenario_ids_are_unique():
    assert len({s.id for s in SCENARIOS}) == len(SCENARIOS)


def test_at_least_one_scenario_needs_no_revision():
    assert any(len(s.revision_rounds) == 0 for s in SCENARIOS)


def test_at_least_one_scenario_needs_multiple_revisions():
    assert any(len(s.revision_rounds) >= 2 for s in SCENARIOS)


def test_revision_feedback_is_never_empty_text():
    for s in SCENARIOS:
        for feedback in s.revision_rounds:
            assert feedback.strip(), s.id


# --- the ledger registry ---------------------------------------------------------------------


def test_book_returns_the_same_instance_for_the_same_run_id():
    a = _book("run-x")
    b = _book("run-x")
    assert a is b


def test_book_returns_different_instances_for_different_run_ids():
    a = _book("run-y")
    b = _book("run-z")
    assert a is not b


def test_a_fresh_book_starts_at_zero_calls():
    assert _book("run-fresh-one").n_calls == 0


# --- live ----------------------------------------------------------------------------------------
#
# Every meaningful assertion in this project is about a *count of real model calls*, which by
# construction only exists once a model actually runs it — there is nothing here that can be
# checked as a pure function the way `is_monotonic` or `represented()` can be in the other
# projects. All of it lives behind the `live` mark.

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_checkpointed_drafts_exactly_once_per_revision_round_plus_one():
    from projects.p04_checkpoint_resume.graph import run_checkpointed
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip("model not pulled")

    s = BY_ID["s3"]  # two revision rounds
    r = run_checkpointed(s, model=tag)
    assert r.draft_calls == len(s.revision_rounds) + 1
    assert r.finalize_calls == 1


@live
@pytest.mark.live
def test_naive_requeue_drafts_one_extra_time_on_the_approval_call():
    from projects.p04_checkpoint_resume.graph import run_naive
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip("model not pulled")

    s = BY_ID["s1"]  # zero revision rounds — isolates the approval-call overhead alone
    r = run_naive(s, model=tag)
    assert r.draft_calls == 2  # the initial draft, plus one wasted redraft on approval
    assert r.finalize_calls == 1


@live
@pytest.mark.live
def test_naive_costs_exactly_one_more_draft_call_than_checkpointed():
    """The headline comparison, for a scenario with revisions in it too."""
    from projects.p04_checkpoint_resume.graph import run_checkpointed, run_naive
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip("model not pulled")

    s = BY_ID["s2"]
    ck = run_checkpointed(s, model=tag)
    nv = run_naive(s, model=tag)
    assert nv.draft_calls == ck.draft_calls + 1
