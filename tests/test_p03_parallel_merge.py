"""Tests for project 03.

The `_merge_dicts` reducer is the load-bearing piece: without it, four parallel branches
writing to `findings` in the same superstep raise `InvalidUpdateError`, which was confirmed
directly against the LangGraph API rather than assumed — an earlier version of this project
assumed the failure mode was a silent overwrite, and it was not.
"""

from __future__ import annotations

import pytest

from projects.p03_parallel_merge.briefs import ASPECTS, BRIEFS, BY_ID, represented
from projects.p03_parallel_merge.graph import RunResult, _merge_dicts, build_graph
from shared.llm import server_is_up

# --- the corpus ----------------------------------------------------------------------------


def test_brief_ids_are_unique():
    assert len({b.id for b in BRIEFS}) == len(BRIEFS)


def test_every_brief_covers_all_four_aspects_with_facts_and_signatures():
    for b in BRIEFS:
        assert set(b.facts) == set(ASPECTS), b.id
        assert set(b.signature) == set(ASPECTS), b.id


def test_every_signature_has_at_least_one_phrasing():
    for b in BRIEFS:
        for aspect, phrasings in b.signature.items():
            assert len(phrasings) >= 1, f"{b.id}/{aspect}"


def test_a_fact_contains_at_least_one_of_its_own_signature_phrases():
    """The signature should describe the fact it was drawn from, not something unrelated."""
    for b in BRIEFS:
        for aspect in ASPECTS:
            fact_lower = b.facts[aspect].lower()
            assert any(p.lower() in fact_lower for p in b.signature[aspect]), f"{b.id}/{aspect}"


# --- represented() ---------------------------------------------------------------------------


def test_represented_matches_any_accepted_phrasing():
    b = BY_ID["b2"]
    text = "the audit found no critical issues at all"
    assert "technical" in represented(text, b)


def test_represented_is_empty_when_nothing_matches():
    b = BY_ID["b1"]
    assert represented("a completely unrelated sentence", b) == ()


def test_represented_only_counts_the_aspects_actually_present():
    b = BY_ID["b1"]
    text = f"Financial note: {b.signature['financial'][0]}."
    assert represented(text, b) == ("financial",)


# --- the merge reducer -----------------------------------------------------------------------


def test_merge_dicts_combines_disjoint_keys():
    assert _merge_dicts({"a": "1"}, {"b": "2"}) == {"a": "1", "b": "2"}


def test_merge_dicts_lets_the_second_argument_win_on_overlap():
    assert _merge_dicts({"a": "old"}, {"a": "new"}) == {"a": "new"}


def test_the_graph_builds_without_raising():
    """This is the regression test for the InvalidUpdateError the first version hit."""
    build_graph()


# --- RunResult derived properties -----------------------------------------------------------


def test_no_failure_means_disclosure_question_does_not_apply():
    b = BY_ID["b1"]
    r = RunResult(brief=b, strategy="all_succeed", fail_aspect="", report="anything", findings={})
    assert not r.missing_aspect_disclosed
    assert not r.reads_as_complete


def test_disclosure_requires_naming_the_failed_aspect_and_a_gap_marker():
    b = BY_ID["b1"]
    named_and_flagged = RunResult(
        brief=b,
        strategy="x",
        fail_aspect="technical",
        report="The technical section could not be completed due to missing data.",
        findings={},
    )
    assert named_and_flagged.missing_aspect_disclosed
    assert not named_and_flagged.reads_as_complete


def test_naming_the_aspect_without_a_gap_marker_is_not_disclosure():
    """Mentioning the word 'technical' in passing is not the same as flagging a gap."""
    b = BY_ID["b1"]
    r = RunResult(
        brief=b,
        strategy="x",
        fail_aspect="technical",
        report="The technical outlook for this sector is generally positive.",
        findings={},
    )
    assert not r.missing_aspect_disclosed


def test_a_smooth_report_with_no_disclosure_reads_as_complete():
    b = BY_ID["b1"]
    r = RunResult(
        brief=b,
        strategy="fail_silent",
        fail_aspect="technical",
        report="Everything about this company looks strong across the board.",
        findings={},
    )
    assert r.reads_as_complete


# --- live ----------------------------------------------------------------------------------------

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_all_branches_run_and_produce_a_report():
    from projects.p03_parallel_merge.graph import run
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip("model not pulled")

    b = BY_ID["b1"]
    r = run(b, strategy="all_succeed", fail_aspect="", flag_gaps=False, model=tag)
    assert r.report
    assert r.calls == 5  # 4 analysts + 1 synthesis
