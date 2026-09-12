"""Explanation tasks with a checklist of required facts, for scoring a revision loop.

The generate -> critique -> revise cycle is the graph LangGraph is demonstrated with most
often, because it is exactly the shape LCEL cannot express: the critique node has to be able to
send control back to the generator, arbitrarily many times, based on what it finds.

Every demo of this pattern assumes more iterations help. This project tests that assumption by
scoring each revision against a **fixed checklist of facts the answer should contain** — the
same instrument as `langchain-lab` project 01's nullable fields, adapted to a different failure
mode. Where project 01 asked whether a model invents a value it lacks evidence for, this asks
whether a revision loop can *lose* a fact it already had, which a monotonic-improvement
assumption has no way to notice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    #: Facts a complete, correct answer should state. Checked by substring, case-insensitive —
    #: deliberately simple, so what is being measured is the loop's behaviour, not the scorer's.
    checklist: tuple[str, ...]
    #: A deliberately incomplete first draft the loop starts from, so every task begins with
    #: room to improve and the critique node has real work on iteration one.
    seed: str


TASKS: tuple[Task, ...] = (
    Task(
        id="t1",
        question="Explain what a race condition is and how to prevent one.",
        checklist=(
            "shared",
            "concurrent",
            "order",
            "lock",
            "atomic",
        ),
        seed="A race condition is a bug that happens with threads.",
    ),
    Task(
        id="t2",
        question="Explain the CAP theorem.",
        checklist=(
            "consistency",
            "availability",
            "partition",
            "network",
            "trade-off",
        ),
        seed="CAP theorem is about databases and what you can guarantee.",
    ),
    Task(
        id="t3",
        question="Explain why database migrations should be backward compatible.",
        checklist=(
            "old code",
            "deploy",
            "rollback",
            "downtime",
            "column",
        ),
        seed="Migrations change the database schema.",
    ),
    Task(
        id="t4",
        question="Explain the difference between authentication and authorization.",
        checklist=(
            "identity",
            "who you are",
            "permission",
            "access",
            "after",
        ),
        seed="Authentication and authorization are both security concepts.",
    ),
    Task(
        id="t5",
        question="Explain why you should not store passwords in plaintext.",
        checklist=(
            "hash",
            "breach",
            "salt",
            "reuse",
            "irreversible",
        ),
        seed="Storing passwords in plaintext is a bad security practice.",
    ),
    Task(
        id="t6",
        question="Explain what backpressure means in a streaming system.",
        checklist=(
            "producer",
            "consumer",
            "faster",
            "buffer",
            "slow down",
        ),
        seed="Backpressure is a term used in streaming systems.",
    ),
)

BY_ID = {t.id: t for t in TASKS}


def score(text: str, task: Task) -> float:
    """Fraction of the checklist the text contains, checked as a case-insensitive substring."""
    lowered = text.lower()
    hits = sum(1 for item in task.checklist if item.lower() in lowered)
    return hits / len(task.checklist)


def present(text: str, task: Task) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(item for item in task.checklist if item.lower() in lowered)
