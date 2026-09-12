"""Tests for project 01.

`RunResult`'s derived properties are what the whole benchmark's numbers rest on — `regressed`,
`lost_facts`, `is_monotonic` — so they get pure-function coverage independent of any live graph
run, built directly from fabricated history.
"""

from __future__ import annotations

import pytest

from projects.p01_revision_loops.graph import Round, RunResult
from projects.p01_revision_loops.tasks import BY_ID, TASKS, present, score
from shared.llm import server_is_up

# --- the corpus ----------------------------------------------------------------------------


def test_task_ids_are_unique():
    assert len({t.id for t in TASKS}) == len(TASKS)


def test_every_task_has_a_checklist_of_at_least_four_items():
    for t in TASKS:
        assert len(t.checklist) >= 4, t.id


def test_the_seed_answer_is_deliberately_incomplete():
    """If the seed already satisfied the checklist there would be nothing to revise."""
    for t in TASKS:
        assert score(t.seed, t) < 0.5, f"{t.id} seed scores {score(t.seed, t):.2f}"


def test_score_is_zero_for_empty_text():
    assert score("", TASKS[0]) == 0.0


def test_score_is_one_when_every_checklist_item_is_present():
    t = BY_ID["t1"]
    text = " ".join(t.checklist)
    assert score(text, t) == 1.0


def test_score_ignores_case():
    t = BY_ID["t1"]
    text = " ".join(item.upper() for item in t.checklist)
    assert score(text, t) == 1.0


def test_present_lists_exactly_the_matched_items():
    t = BY_ID["t1"]
    text = t.checklist[0] + " and " + t.checklist[1]
    assert set(present(text, t)) == {t.checklist[0], t.checklist[1]}


# --- RunResult derived properties -----------------------------------------------------------


def _round(draft: str, iteration: int, task) -> Round:
    return {
        "iteration": iteration,
        "draft": draft,
        "critique": "",
        "checklist_present": present(draft, task),
    }


def test_monotonic_when_score_only_increases():
    t = BY_ID["t1"]
    partial = " ".join(t.checklist[:2])
    full = " ".join(t.checklist)
    r = RunResult(
        task=t,
        strategy="x",
        final_draft=full,
        final_critique="",
        iterations=2,
        history=[_round(partial, 1, t), _round(full, 2, t)],
    )
    assert r.is_monotonic
    assert not r.regressed
    assert r.lost_facts == ()


def test_regression_detected_when_score_drops_then_recovers_incompletely():
    t = BY_ID["t1"]
    full = " ".join(t.checklist)
    partial = " ".join(t.checklist[:2])
    r = RunResult(
        task=t,
        strategy="x",
        final_draft=partial,
        final_critique="",
        iterations=2,
        history=[_round(full, 1, t), _round(partial, 2, t)],
    )
    assert not r.is_monotonic
    assert r.regressed


def test_lost_facts_reports_what_was_present_and_then_dropped():
    t = BY_ID["t1"]
    full = " ".join(t.checklist)
    without_first = " ".join(t.checklist[1:])
    r = RunResult(
        task=t,
        strategy="x",
        final_draft=without_first,
        final_critique="",
        iterations=1,
        history=[_round(full, 1, t)],
    )
    assert r.lost_facts == (t.checklist[0],)


def test_no_history_means_no_score_trace_and_no_regression():
    """fixed_1 never enters the loop, so history is empty by construction."""
    t = BY_ID["t1"]
    r = RunResult(task=t, strategy="fixed_1", final_draft=t.seed, final_critique="", iterations=0)
    assert r.score_trace == []
    assert r.is_monotonic  # vacuously true — nothing to compare
    assert not r.regressed


def test_final_score_matches_the_scoring_function():
    t = BY_ID["t1"]
    full = " ".join(t.checklist)
    r = RunResult(task=t, strategy="x", final_draft=full, final_critique="", iterations=0)
    assert r.final_score == score(full, t)


# --- live ---------------------------------------------------------------------------------------

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_the_graph_runs_and_produces_a_scoreable_draft():
    from projects.p01_revision_loops.graph import run
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip("model not pulled")

    t = BY_ID["t1"]
    result = run(t, strategy="fixed_1", max_iterations=0, stop_on_complete=False, model=tag)
    assert result.final_draft
    assert result.iterations == 0
    assert result.calls >= 2  # generate + critique
