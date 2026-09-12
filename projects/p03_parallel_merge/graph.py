"""A four-way parallel fan-out into one synthesis node, built as an actual LangGraph graph.

Four analyst nodes run as true parallel branches — LangGraph schedules nodes with no edge
between them in the same superstep — each writing its finding to its **own** state key. That
detail is load-bearing: an earlier version of this project had every analyst write to one
shared key, and LangGraph refused to run it at all, raising `InvalidUpdateError` rather than
silently picking a winner. Separate keys sidestep that error entirely, which is also how a
fan-out like this is actually written in practice.

What separate keys do not prevent is one branch **failing** — simulated here by forcing a named
analyst to return nothing, standing in for a timed-out tool call or an errored API response —
while the graph carries on regardless and hands the synthesis node whatever it got, missing
section included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from projects.p03_parallel_merge.briefs import ASPECTS, Brief, represented
from shared.llm import Ledger, chat

ANALYST_SYSTEM = (
    "You are a due-diligence analyst covering exactly one aspect of a company for an "
    "investment memo. Write 1-2 sentences summarising the facts you were given. State the "
    "specific figures. Do not speculate beyond what you were told."
)

SYNTHESIS_SYSTEM = (
    "You write a short investment memo combining findings from four analysts: financial, "
    "technical, market and team. Combine their findings into one coherent paragraph covering "
    "all four aspects, in 4-6 sentences."
)

SYNTHESIS_SYSTEM_FLAGGED = (
    SYNTHESIS_SYSTEM
    + " If any analyst's section is empty or missing, you must explicitly say in your memo "
    "that this section could not be completed and name which one — never write around a gap "
    "as though it were not there."
)


def _merge_dicts(a: dict[str, str], b: dict[str, str]) -> dict[str, str]:
    """The reducer `findings` needs to be writable from four parallel branches at once.

    Each analyst returns a dict holding only its own key, so the four concurrent updates in
    one superstep are genuinely different values — LangGraph will not silently pick one, it
    needs to be told how to combine them. Without this, the graph raises the same
    `InvalidUpdateError` that a shared scalar key does; a dict is not automatically merge-safe
    just because it looks like a container.
    """
    return {**a, **b}


class MergeState(TypedDict):
    brief: Brief
    fail_aspect: str  # "" for none
    flag_gaps: bool
    model: str
    ledgers: list[Ledger]
    findings: Annotated[dict[str, str], _merge_dicts]
    report: str


def _analyst(aspect: str):
    def node(state: MergeState) -> dict:
        # Returning only the changed key matters here, not just for tidiness. Four branches
        # running in the same superstep each returning `{**state, ...}` would hand LangGraph
        # four "updates" to every unchanged field too — `brief`, `model`, `ledgers` — and it
        # rejects that as a conflicting concurrent write even though the values are identical.
        # A parallel branch must return only its own delta.
        if aspect == state["fail_aspect"]:
            # Standing in for a timed-out tool call or an errored upstream API: the branch
            # completes (the graph does not crash) but produces nothing. The reducer merges
            # this single-key dict with the other three branches' single-key dicts.
            return {"findings": {aspect: ""}}

        fact = state["brief"].facts[aspect]
        llm = chat(state["model"], callbacks=state["ledgers"])
        reply = llm.invoke(
            [
                SystemMessage(content=ANALYST_SYSTEM),
                HumanMessage(content=f"Aspect: {aspect}\nFacts: {fact}"),
            ]
        )
        return {"findings": {aspect: str(reply.content).strip()}}

    return node


def synthesise(state: MergeState) -> dict:
    system = SYNTHESIS_SYSTEM_FLAGGED if state["flag_gaps"] else SYNTHESIS_SYSTEM
    sections = "\n\n".join(
        f"{a.upper()}: {state['findings'].get(a) or '(no content received)'}" for a in ASPECTS
    )
    llm = chat(state["model"], callbacks=state["ledgers"])
    reply = llm.invoke([SystemMessage(content=system), HumanMessage(content=sections)])
    return {"report": str(reply.content).strip()}


def build_graph():
    graph = StateGraph(MergeState)
    for aspect in ASPECTS:
        graph.add_node(aspect, _analyst(aspect))
        graph.add_edge(aspect, "synthesise")
    graph.add_node("synthesise", synthesise)
    graph.add_edge("synthesise", END)
    # An edge directly from START into each analyst, rather than a single `set_entry_point`,
    # is what makes all four run in the same superstep — LangGraph schedules every node
    # reachable from START with no unmet dependency together, so this is genuine parallelism,
    # not four sequential calls that merely look tidy in the source.
    for aspect in ASPECTS:
        graph.add_edge(START, aspect)
    return graph.compile()


@dataclass
class RunResult:
    brief: Brief
    strategy: str
    fail_aspect: str
    report: str
    findings: dict[str, str]
    calls: int = 0
    seconds: float = 0.0

    @property
    def represented_aspects(self) -> tuple[str, ...]:
        return represented(self.report, self.brief)

    @property
    def missing_aspect_disclosed(self) -> bool:
        """When a branch failed, did the report say so, by name, rather than staying silent?"""
        if not self.fail_aspect:
            return False
        lowered = self.report.lower()
        markers = ("could not be completed", "missing", "no content", "unable to", "not available")
        return self.fail_aspect.lower() in lowered and any(m in lowered for m in markers)

    @property
    def reads_as_complete(self) -> bool:
        """A failed branch produced no disclosure and the report still covers all 4 by name."""
        return bool(self.fail_aspect) and not self.missing_aspect_disclosed


def run(
    brief: Brief,
    *,
    strategy: str,
    fail_aspect: str,
    flag_gaps: bool,
    model: str,
    ledger: Ledger | None = None,
) -> RunResult:
    graph = build_graph()
    book = Ledger()
    ledgers = [book, ledger] if ledger else [book]
    state: MergeState = {
        "brief": brief,
        "fail_aspect": fail_aspect,
        "flag_gaps": flag_gaps,
        "model": model,
        "ledgers": ledgers,
        "findings": {},
        "report": "",
    }
    final = graph.invoke(state)
    return RunResult(
        brief=brief,
        strategy=strategy,
        fail_aspect=fail_aspect,
        report=final["report"],
        findings=final["findings"],
        calls=book.n_calls,
        seconds=round(book.seconds, 2),
    )
