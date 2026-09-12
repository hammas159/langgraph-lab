"""A two-agent handoff, built as an actual LangGraph graph, with three handoff strategies.

The graph itself is a `supervisor -> specialist` handoff: `intake` receives the opening message
and hands control to `specialist`, which answers the follow-up. What varies between strategies
is what `specialist` sees of the conversation that happened before it took over — this is a
genuine architectural choice every supervisor-pattern implementation has to make, not a bug
being planted:

    full_transcript     specialist gets the opening message, the intake reply, and the followup.
    summary_handoff      a supervisor node writes a short handoff note — real LLM call — and
                         the specialist gets that plus the followup, not the raw transcript.
    last_message_only    the specialist gets only the followup. No handoff content at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from projects.p05_supervisor_handoff.cases import Case, on_topic, violates
from shared.llm import Ledger, chat

INTAKE_SYSTEM = (
    "You are a front-line intake agent. Acknowledge the customer's request briefly in one "
    "sentence and say you are connecting them with a specialist. Do not answer the request "
    "yourself."
)

SUMMARY_SYSTEM = (
    "You are a supervisor handing off a customer conversation to a specialist agent. Write a "
    "one-sentence handoff note stating anything the specialist must know to serve this "
    "customer safely and correctly — especially any hard constraint, allergy, budget limit, "
    "or thing they explicitly do not want. If there is nothing critical, say so."
)

SPECIALIST_SYSTEM = (
    "You are a specialist agent picking up a conversation from an intake agent. Answer the "
    "customer's question directly and helpfully in 2-3 sentences, using whatever context you "
    "have been given about them."
)


class HandoffState(TypedDict):
    case: Case
    strategy: str
    model: str
    run_id: str
    handoff_context: str
    response: str


_BOOKS: dict[str, Ledger] = {}


def _book(run_id: str) -> Ledger:
    return _BOOKS.setdefault(run_id, Ledger())


def _build_context(state: HandoffState) -> dict:
    case, strategy = state["case"], state["strategy"]

    if strategy == "full_transcript":
        context = (
            f"Customer: {case.opening}\nIntake agent: {case.first_agent_reply}\n"
            f"Customer: {case.followup}"
        )
    elif strategy == "summary_handoff":
        llm = chat(state["model"], callbacks=[_book(state["run_id"])])
        reply = llm.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM),
                HumanMessage(content=f"Customer's opening message: {case.opening}"),
            ]
        )
        note = str(reply.content).strip()
        context = f"Handoff note from intake: {note}\nCustomer: {case.followup}"
    else:  # last_message_only
        context = f"Customer: {case.followup}"

    return {"handoff_context": context}


def _specialist(state: HandoffState) -> dict:
    llm = chat(state["model"], callbacks=[_book(state["run_id"])])
    reply = llm.invoke(
        [
            SystemMessage(content=SPECIALIST_SYSTEM),
            HumanMessage(content=state["handoff_context"]),
        ]
    )
    return {"response": str(reply.content).strip()}


def build_graph():
    graph = StateGraph(HandoffState)
    graph.add_node("build_context", _build_context)
    graph.add_node("specialist", _specialist)
    graph.set_entry_point("build_context")
    graph.add_edge("build_context", "specialist")
    graph.add_edge("specialist", END)
    return graph.compile()


@dataclass
class RunResult:
    case: Case
    strategy: str
    handoff_context: str
    response: str
    calls: int = 0
    seconds: float = 0.0

    @property
    def violated(self) -> bool:
        return violates(self.response, self.case)

    @property
    def stayed_on_topic(self) -> bool:
        return on_topic(self.response, self.case)

    @property
    def safe_and_relevant(self) -> bool:
        """The only reading of "worked" that cannot be gamed by losing the thread entirely."""
        return self.stayed_on_topic and not self.violated


def run(case: Case, *, strategy: str, model: str) -> RunResult:
    graph = build_graph()
    run_id = f"{strategy}-{case.id}-{id(graph)}"
    state: HandoffState = {
        "case": case,
        "strategy": strategy,
        "model": model,
        "run_id": run_id,
        "handoff_context": "",
        "response": "",
    }
    final = graph.invoke(state)
    book = _book(run_id)
    return RunResult(
        case=case,
        strategy=strategy,
        handoff_context=final["handoff_context"],
        response=final["response"],
        calls=book.n_calls,
        seconds=round(book.seconds, 2),
    )
