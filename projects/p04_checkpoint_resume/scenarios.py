"""Refund requests that need human approval, with a scripted reviewer.

Human-in-the-loop is the fourth common LangGraph pattern: a graph drafts an action, pauses for
a person to approve it, and continues once they do. LangGraph ships a real mechanism for this —
`interrupt()` plus a checkpointer — and the trap is that it is easy to build the same-looking
behaviour without it: a node that simply checks an "approved" flag and ends the run early if it
is not set, with the calling code re-invoking the graph from the original input once approval
arrives. That looks identical from the outside. It is not identical underneath, because
nothing has paused — the second invocation restarts the graph and re-runs the drafting node it
already ran once.

The reviewer here is scripted rather than a real person, so a run is deterministic and
reproducible: each scenario specifies how many rounds of revision the reviewer asks for before
approving, standing in for a slow or picky human.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    request: str
    #: Feedback the scripted reviewer gives on each round before finally approving. An empty
    #: tuple means approved immediately, on the first draft.
    revision_rounds: tuple[str, ...]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="s1",
        request="Customer wants a full refund for a laptop that arrived with a cracked screen, "
        "48 days after purchase (past our 30-day window).",
        revision_rounds=(),
    ),
    Scenario(
        id="s2",
        request="Customer wants a refund for a software subscription they forgot to cancel, "
        "used for 3 of the 12 months billed.",
        revision_rounds=(
            "Cite our actual partial-refund policy for unused months, not a generic apology.",
        ),
    ),
    Scenario(
        id="s3",
        request="Customer wants a refund for a conference ticket; the event was cancelled by "
        "the organiser, not by us.",
        revision_rounds=(
            "State clearly whether this falls under our terms or the organiser's.",
            "Add the specific clause number from our terms of service.",
        ),
    ),
)

BY_ID = {s.id: s for s in SCENARIOS}
