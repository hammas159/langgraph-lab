"""Two ways to build "pause for human approval," one of which cannot avoid redoing work.

`checkpointed` uses LangGraph's actual mechanism: `interrupt()` inside a node, a checkpointer
attached to the compiled graph, and `Command(resume=...)` to continue. The graph genuinely
suspends mid-run; resuming it does not re-enter nodes that already completed.

`naive_requeue` looks the same from outside a request handler and is not the same underneath.
There is no checkpointer. "Pausing" is implemented the way it is easy to reach for without
knowing `interrupt()` exists: the graph runs to completion every time it is invoked, an early
node checks whether approval was passed in, and if not it ends the run immediately with a
"pending" result. Getting a decision means invoking the **whole graph again from the original
request**, which re-executes the drafting node — a real LLM call, standing in for anything a
real system would not want to redo, from an actual API call to a temporary resource hold.

Both are driven by `run_to_approval`, which plays a scripted sequence of reviewer rounds
against the graph and counts how many times the drafting node actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from projects.p04_checkpoint_resume.scenarios import Scenario
from shared.llm import Ledger, chat

DRAFT_SYSTEM = (
    "You draft a short refund decision memo for internal approval: state the decision "
    "(approve/deny/partial) and a one-sentence reason citing policy. 2-3 sentences total. "
    "If reviewer feedback is provided, revise the previous draft to address it."
)


class GraphState(TypedDict):
    request: str
    feedback: str  # the latest reviewer feedback, "" if none yet
    draft: str
    approved: bool
    pending: bool  # naive-only: true once a draft exists and awaits approval
    model: str
    run_id: str


#: A live `Ledger` cannot live inside checkpointed state: `checkpointed` persists state between
#: invocations by serialising it, and the first version of this project put the ledger there.
#: Every resume deserialised a *disconnected copy* — real model calls kept happening, the copy
#: kept counting them, and the external variable being inspected never saw the increments,
#: silently under-reporting by every call after the first. Call counts are kept here instead,
#: keyed by a plain string that checkpoints without incident.
_BOOKS: dict[str, Ledger] = {}


def _book(run_id: str) -> Ledger:
    return _BOOKS.setdefault(run_id, Ledger())


def _draft(state: GraphState) -> dict:
    llm = chat(state["model"], callbacks=[_book(state["run_id"])])
    content = f"Request: {state['request']}"
    if state.get("feedback"):
        content += (
            f"\n\nPrevious draft:\n{state['draft']}\n\nReviewer feedback: {state['feedback']}"
        )
    reply = llm.invoke([SystemMessage(content=DRAFT_SYSTEM), HumanMessage(content=content)])
    return {"draft": str(reply.content).strip()}


def _finalize(state: GraphState) -> dict:
    return {"approved": True}


# --- checkpointed: interrupt() genuinely suspends the run --------------------------------------


def _checkpoint_gate(state: GraphState) -> dict:
    decision = interrupt({"draft": state["draft"]})
    if decision == "approve":
        return {"feedback": ""}
    return {"feedback": decision}


def _checkpoint_route(state: GraphState) -> str:
    return "finalize" if not state["feedback"] else "draft"


def build_checkpointed():
    graph = StateGraph(GraphState)
    graph.add_node("draft", _draft)
    graph.add_node("gate", _checkpoint_gate)
    graph.add_node("finalize", _finalize)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "gate")
    graph.add_conditional_edges(
        "gate", _checkpoint_route, {"finalize": "finalize", "draft": "draft"}
    )
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=MemorySaver())


# --- naive: "pause" is an early return; resuming means starting over --------------------------


def _naive_gate(state: GraphState) -> dict:
    return {"pending": not state["approved"]}


def _naive_route(state: GraphState) -> str:
    return "finalize" if state["approved"] else END


def build_naive():
    graph = StateGraph(GraphState)
    graph.add_node("draft", _draft)
    graph.add_node("gate", _naive_gate)
    graph.add_node("finalize", _finalize)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "gate")
    graph.add_conditional_edges("gate", _naive_route, {"finalize": "finalize", END: END})
    graph.add_edge("finalize", END)
    return graph.compile()  # no checkpointer: nothing persists between invocations


@dataclass
class RunResult:
    scenario: Scenario
    strategy: str
    draft_calls: int
    finalize_calls: int
    final_draft: str
    seconds: float = 0.0
    rounds: list[str] = field(default_factory=list)


def run_checkpointed(scenario: Scenario, *, model: str) -> RunResult:
    graph = build_checkpointed()
    run_id = f"ckpt-{scenario.id}-{id(graph)}"
    cfg = {"configurable": {"thread_id": run_id}}

    state: GraphState = {
        "request": scenario.request,
        "feedback": "",
        "draft": "",
        "approved": False,
        "pending": False,
        "model": model,
        "run_id": run_id,
    }
    result = graph.invoke(state, config=cfg)
    for feedback in scenario.revision_rounds:
        result = graph.invoke(Command(resume=feedback), config=cfg)
    result = graph.invoke(Command(resume="approve"), config=cfg)

    book = _book(run_id)
    return RunResult(
        scenario=scenario,
        strategy="checkpointed",
        draft_calls=book.n_calls,
        finalize_calls=1 if result.get("approved") else 0,
        final_draft=result.get("draft", ""),
        seconds=round(book.seconds, 2),
        rounds=list(scenario.revision_rounds),
    )


def run_naive(scenario: Scenario, *, model: str) -> RunResult:
    graph = build_naive()
    run_id = f"naive-{scenario.id}-{id(graph)}"

    # Every round is a fresh invocation from the original request plus whatever feedback has
    # accumulated so far — there is no persisted state to resume, so the caller must resupply
    # everything, and the naive graph re-runs `draft` unconditionally on every single
    # invocation, including the one that only exists to say "go ahead, finalize this."
    def invoke(feedback: str, approved: bool, prior_draft: str) -> dict:
        state: GraphState = {
            "request": scenario.request,
            "feedback": feedback,
            "draft": prior_draft,
            "approved": approved,
            "pending": False,
            "model": model,
            "run_id": run_id,
        }
        return graph.invoke(state)

    # Phase 1: the initial draft, with no decision yet.
    result = invoke(feedback="", approved=False, prior_draft="")
    # Phase 2: one fresh full invocation per round of reviewer feedback.
    for feedback in scenario.revision_rounds:
        result = invoke(feedback=feedback, approved=False, prior_draft=result.get("draft", ""))
    # Phase 3: the approval call. The naive graph has no way to skip straight to `finalize`
    # with the already-approved draft — it re-runs `draft` one more time regardless.
    result = invoke(feedback="", approved=True, prior_draft=result.get("draft", ""))

    book = _book(run_id)
    return RunResult(
        scenario=scenario,
        strategy="naive_requeue",
        draft_calls=book.n_calls,
        finalize_calls=1 if result.get("approved") else 0,
        final_draft=result.get("draft", ""),
        seconds=round(book.seconds, 2),
        rounds=list(scenario.revision_rounds),
    )
