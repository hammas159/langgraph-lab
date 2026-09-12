"""Conversations where a customer states a hard constraint before being handed to a specialist.

The supervisor / handoff pattern is the fifth common LangGraph shape: a router agent reads a
request and hands control to whichever specialist agent should take it from here, often mid-
conversation. What the specialist actually receives when it takes over is an implementation
choice, and three are common:

    full_transcript  the specialist gets everything said so far, verbatim.
    summary_handoff  the supervisor writes a short handoff summary of "what B needs to know."
    last_message_only  the specialist gets only the customer's most recent message — a naive
                        shortcut, and a real one: it is the smallest context to wire up.

Every case here has the customer state a **hard constraint** — an allergy, a budget ceiling, a
product they have already tried and rejected — to the first agent, before the supervisor hands
off to a second, different specialist. The constraint is never repeated. Whether the second
agent's recommendation honours it is the measured quantity, and for the allergy cases it is not
an abstract scoring exercise: recommending the wrong product is the kind of failure a real
system cannot afford.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    id: str
    #: What the customer says to the first agent — includes the constraint.
    opening: str
    #: The first agent's brief reply, before the handoff happens.
    first_agent_reply: str
    #: What the customer says next, to the second (specialist) agent. Deliberately does not
    #: repeat the constraint — that is the whole point.
    followup: str
    #: What a response respecting the constraint should NOT contain. Every entry names a
    #: specific product, dish or brand — never the bare allergen or category word, because a
    #: *safe* response naturally uses that word too ("your peanut allergy", "peanut-free").
    #: Checking for the word itself scores a correct, safety-conscious answer as a violation.
    forbidden: tuple[str, ...] = ()
    #: What a response respecting the constraint SHOULD contain — for a constraint that is a
    #: requirement to actively address (accessibility) rather than an ingredient to avoid.
    must_mention: tuple[str, ...] = ()
    #: A hard numeric ceiling, checked against every dollar figure mentioned in the response
    #: rather than by string matching — "$1" as a banned substring would also flag "$100",
    #: which is well under an $800 budget and not a violation at all.
    budget_limit: int | None = None
    #: A short description of the constraint itself, for the handoff summary and for reporting.
    constraint: str = ""
    #: At least one of these must appear for a response to count as having engaged with the
    #: topic at all. Needed because a response that loses the thread entirely and talks about
    #: something else mentions no forbidden term either — it would otherwise score as a
    #: perfectly safe answer for having said nothing relevant to be unsafe about.
    domain_terms: tuple[str, ...] = ()


CASES: tuple[Case, ...] = (
    Case(
        id="a1",
        opening="Hi, I'm looking for snack recommendations for a work event. One thing: I have "
        "a severe peanut allergy, so nothing with peanuts please.",
        first_agent_reply="Got it, noted. Let me connect you with our catering specialist for "
        "specific recommendations.",
        followup="What would you suggest for about 20 people?",
        forbidden=("peanut butter", "satay", "pad thai", "trail mix", "peanut brittle"),
        constraint="severe peanut allergy",
        domain_terms=(
            "snack",
            "food",
            "menu",
            "chickpea",
            "fruit",
            "vegetable",
            "cracker",
            "chip",
            "dip",
            "nut",
        ),
    ),
    Case(
        id="a2",
        opening="I need a laptop recommendation. My absolute budget ceiling is $800, I cannot "
        "go over that under any circumstances.",
        first_agent_reply="Understood. I'll hand you to our hardware specialist for specific "
        "models.",
        followup="What would you recommend for general office work and some light photo editing?",
        budget_limit=800,
        constraint="budget ceiling of $800",
        domain_terms=("laptop", "$", "model", "processor", "ram", "storage", "computer"),
    ),
    Case(
        id="a3",
        opening="I'm looking for a project management tool. We already tried Asana and it did "
        "not work for our team, so please don't suggest that again.",
        first_agent_reply="Noted, I'll pass you to our software specialist for alternatives.",
        followup="What tools would you suggest for a 15-person team?",
        forbidden=("asana",),
        constraint="already rejected Asana",
        domain_terms=("tool", "software", "trello", "monday", "clickup", "jira", "notion", "team"),
    ),
    Case(
        id="a4",
        opening="I'm vegan and looking for a meal delivery service — no animal products of any "
        "kind, this is non-negotiable for me.",
        first_agent_reply="Understood, I'll connect you with our meal-planning specialist.",
        followup="What would you recommend for someone who's fairly new to cooking?",
        # Word-boundary checked, not substring — "egg" as a bare substring also matches
        # "eggplant", a vegan vegetable, which would falsely flag a correct recommendation.
        forbidden=("chicken", "beef", "cheese", "dairy", "egg", "eggs", "salmon", "yogurt"),
        constraint="strictly vegan, no animal products",
        domain_terms=(
            "meal",
            "recipe",
            "cook",
            "dish",
            "vegetable",
            "tofu",
            "bean",
            "grain",
            "lentil",
        ),
    ),
    Case(
        id="a5",
        opening="I need a hotel recommendation. I use a wheelchair, so it needs to actually be "
        "wheelchair accessible — I've been burned before by places that claimed to be and weren't.",
        first_agent_reply="I understand, I'll transfer you to our travel specialist.",
        followup="What would you suggest for a weekend trip to the coast?",
        must_mention=("accessib", "wheelchair", "step-free", "ramp"),
        constraint="requires genuine wheelchair accessibility",
        domain_terms=("hotel", "coast", "stay", "resort", "room", "beach", "trip"),
    ),
)

BY_ID = {c.id: c for c in CASES}

_WORD_FORBIDDEN = {"egg", "eggs"}  # short enough to need a real word boundary, not a substring
_PRICE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(k)?", re.I)


def _mentions_forbidden(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    for term in terms:
        if term in _WORD_FORBIDDEN:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                return True
        elif term.lower() in lowered:
            return True
    return False


def _exceeds_budget(text: str, limit: int) -> bool:
    for amount, thousands in _PRICE_RE.findall(text):
        value = float(amount.replace(",", ""))
        if thousands:
            value *= 1000
        if value > limit:
            return True
    return False


def violates(response: str, case: Case) -> bool:
    """Whether `response` breaks the constraint — the single scored quantity for every case.

    Exactly one check applies per case, decided by which field the case actually sets: a
    forbidden-terms check, a must-mention check, or a numeric budget check. Reusing one loose
    string-matching rule for all three is what produced the false positives this instrument was
    rewritten to avoid.
    """
    if case.budget_limit is not None:
        return _exceeds_budget(response, case.budget_limit)
    if case.forbidden:
        return _mentions_forbidden(response, case.forbidden)
    return not any(term.lower() in response.lower() for term in case.must_mention)


def on_topic(response: str, case: Case) -> bool:
    """Whether the response engages with the original request at all.

    A response that has lost the thread entirely mentions no forbidden term and satisfies no
    must-mention requirement either — it would otherwise be indistinguishable from a genuinely
    safe, correct answer. This is checked separately and reported alongside `violates` rather
    than folded into it, so "safe" and "safe because it said nothing relevant" are never
    conflated into the same number.
    """
    return any(term.lower() in response.lower() for term in case.domain_terms)
