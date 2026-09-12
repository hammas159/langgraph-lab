"""A router node built as an actual LangGraph graph, with three dispatch strategies.

The graph is the supervisor pattern LangGraph is demonstrated with constantly: one node reads
the incoming request and decides which specialist node should handle it. The thing this project
tests is not whether the router can classify a clear-cut ticket — it can — but whether its own
stated **confidence** means anything, and whether asking it more than once catches what asking
it once misses.

    single_pass       classify once, dispatch to whatever it says, regardless of confidence.
    confidence_gate    classify once; if confidence is not HIGH, dispatch to `review` instead
                       of guessing.
    multi_vote         classify three times with independently worded prompts; dispatch to the
                       majority category, or to `review` on a three-way tie.

All three share one router call; `confidence_gate` and `multi_vote` differ only in what they do
with the result, so any difference between them and `single_pass` is attributable to the
gating policy, not to a different classifier.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from projects.p02_router_misroute.tickets import CATEGORIES, Ticket
from shared.llm import Ledger, chat

ROUTER_SYSTEM = (
    "You triage support tickets into exactly one category: billing, technical, account, or "
    "general. Reply in exactly this format on two lines:\n"
    "CATEGORY: <one of billing, technical, account, general>\n"
    "CONFIDENCE: <LOW, MEDIUM, or HIGH>\n"
    "Confidence reflects how clearly the ticket belongs to a single category. If the ticket "
    "genuinely touches more than one area, say so with LOW or MEDIUM confidence rather than "
    "picking one and claiming certainty."
)

#: Three independently worded framings of the same task, for the multi-vote strategy. If they
#: were paraphrases of one prompt, agreement between them would say nothing that asking once
#: three times over would not already say.
VOTE_SYSTEMS = (
    ROUTER_SYSTEM,
    (
        "You are the first line of a support queue. Read the ticket and decide which team "
        "should own it: billing, technical, account, or general. Reply in exactly this format "
        "on two lines:\nCATEGORY: <billing, technical, account, or general>\n"
        "CONFIDENCE: <LOW, MEDIUM, or HIGH>"
    ),
    (
        "Classify the following customer message for internal routing. The valid destinations "
        "are billing, technical, account, and general. If in doubt, prefer the category the "
        "customer would most likely say the issue is about. Reply in exactly this format on "
        "two lines:\nCATEGORY: <billing, technical, account, or general>\n"
        "CONFIDENCE: <LOW, MEDIUM, or HIGH>"
    ),
)

_CATEGORY_RE = re.compile(r"CATEGORY:\s*(\w+)", re.I)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\w+)", re.I)


@dataclass
class Classification:
    category: str  # "" if unparseable
    confidence: str  # "LOW" | "MEDIUM" | "HIGH" | ""
    raw: str


def parse(text: str) -> Classification:
    cat_match = _CATEGORY_RE.search(text)
    conf_match = _CONFIDENCE_RE.search(text)
    category = cat_match.group(1).lower() if cat_match else ""
    confidence = conf_match.group(1).upper() if conf_match else ""
    if category not in CATEGORIES:
        category = ""
    if confidence not in ("LOW", "MEDIUM", "HIGH"):
        confidence = ""
    return Classification(category=category, confidence=confidence, raw=text)


def classify(text: str, system: str, model: str, ledgers: list[Ledger]) -> Classification:
    llm = chat(model, callbacks=ledgers)
    reply = llm.invoke([SystemMessage(content=system), HumanMessage(content=text)])
    return parse(str(reply.content).strip())


class RouteState(TypedDict):
    ticket_text: str
    strategy: str
    model: str
    ledgers: list[Ledger]
    votes: list[Classification]
    decision: str  # final category, or "review"
    top_confidence: str


def route_node(state: RouteState) -> RouteState:
    strategy = state["strategy"]
    text, model, ledgers = state["ticket_text"], state["model"], state["ledgers"]

    if strategy == "multi_vote":
        votes = [classify(text, sys, model, ledgers) for sys in VOTE_SYSTEMS]
        counts = Counter(v.category for v in votes if v.category)
        if not counts:
            decision = "review"
        else:
            top_count = max(counts.values())
            winners = [c for c, n in counts.items() if n == top_count]
            decision = winners[0] if len(winners) == 1 else "review"
        return {**state, "votes": votes, "decision": decision, "top_confidence": ""}

    votes = [classify(text, ROUTER_SYSTEM, model, ledgers)]
    v = votes[0]

    if strategy == "confidence_gate":
        decision = v.category if v.confidence == "HIGH" and v.category else "review"
    else:  # single_pass
        decision = v.category or "review"

    return {**state, "votes": votes, "decision": decision, "top_confidence": v.confidence}


def dispatch(state: RouteState) -> str:
    return state["decision"] if state["decision"] in CATEGORIES else "review"


def build_graph():
    graph = StateGraph(RouteState)
    graph.add_node("route", route_node)
    graph.set_entry_point("route")

    # Specialist nodes are trivial: routing correctness is what is under test, not what a
    # specialist would then do with the ticket.
    for category in (*CATEGORIES, "review"):
        graph.add_node(category, lambda s: s)
        graph.add_edge(category, END)

    graph.add_conditional_edges(
        "route", dispatch, {**{c: c for c in CATEGORIES}, "review": "review"}
    )
    return graph.compile()


@dataclass
class RunResult:
    ticket: Ticket
    strategy: str
    decision: str
    votes: list[Classification] = field(default_factory=list)
    top_confidence: str = ""
    calls: int = 0
    seconds: float = 0.0

    @property
    def correct(self) -> bool:
        return self.decision == self.ticket.primary

    @property
    def escalated(self) -> bool:
        return self.decision == "review"

    @property
    def wrong_and_confident(self) -> bool:
        """Misrouted while claiming HIGH confidence. The dangerous failure, not the safe one."""
        return not self.correct and not self.escalated and self.top_confidence == "HIGH"


def run(ticket: Ticket, *, strategy: str, model: str, ledger: Ledger | None = None) -> RunResult:
    graph = build_graph()
    book = Ledger()
    ledgers = [book, ledger] if ledger else [book]
    state: RouteState = {
        "ticket_text": ticket.text,
        "strategy": strategy,
        "model": model,
        "ledgers": ledgers,
        "votes": [],
        "decision": "",
        "top_confidence": "",
    }
    final = graph.invoke(state)
    return RunResult(
        ticket=ticket,
        strategy=strategy,
        decision=final["decision"],
        votes=final["votes"],
        top_confidence=final["top_confidence"],
        calls=book.n_calls,
        seconds=round(book.seconds, 2),
    )
