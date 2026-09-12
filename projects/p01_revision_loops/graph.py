"""A generate -> critique -> revise cycle, built as an actual LangGraph graph with cycles.

This is the shape LangGraph exists for: LCEL composes a fixed pipeline, and a critique node
that can send control back to the reviser an unknown number of times is a cycle, not a chain.
The graph itself is almost trivial —

    generate -> critique -> (loop back to revise, or stop) -> revise -> critique -> ...

— which is deliberate. The thing under test is not the graph's plumbing, it is the **stopping
strategy**: fixed iteration counts against a self-reported "I'm done" signal from the critique
node, scored against the ground-truth checklist the loop itself never sees.

State carries the running answer and the iteration's history, so a benchmark can inspect the
score trajectory after the fact without re-deriving it from the model's raw output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from projects.p01_revision_loops.tasks import Task, present, score
from shared.llm import Ledger, chat

GENERATE_SYSTEM = (
    "You write clear, technically accurate explanations. Answer the question directly in "
    "2-4 sentences. Do not pad the answer with filler."
)

CRITIQUE_SYSTEM = (
    "You review a draft answer for a technical question and list what is missing or wrong. "
    "Be specific and brief: a short bullet list of concrete gaps, or the single word COMPLETE "
    "if the draft is accurate and does not omit anything a correct answer should cover. "
    "Do not praise the draft; only report problems, or say COMPLETE."
)

REVISE_SYSTEM = (
    "You revise a draft answer using the reviewer's feedback. Rewrite the full answer, "
    "incorporating every point raised. Keep it to 2-5 sentences. Do not lose anything correct "
    "that was already in the draft while fixing what was flagged."
)


class Round(TypedDict):
    """One trip through the loop, kept for post-hoc inspection."""

    iteration: int
    draft: str
    critique: str
    checklist_present: tuple[str, ...]


class LoopState(TypedDict):
    task_question: str
    draft: str
    critique: str
    iteration: int
    max_iterations: int
    stop_on_complete: bool
    history: list[Round]
    model: str
    ledger: list[Ledger]


def _invoke(system: str, content: str, state: LoopState) -> str:
    llm = chat(state["model"], callbacks=state["ledger"])
    reply = llm.invoke([SystemMessage(content=system), HumanMessage(content=content)])
    return str(reply.content).strip()


def generate(state: LoopState) -> LoopState:
    draft = _invoke(GENERATE_SYSTEM, f"Question: {state['task_question']}", state)
    return {**state, "draft": draft, "iteration": 0}


def critique(state: LoopState) -> LoopState:
    text = _invoke(
        CRITIQUE_SYSTEM,
        f"Question: {state['task_question']}\n\nDraft:\n{state['draft']}",
        state,
    )
    return {**state, "critique": text}


def revise(state: LoopState) -> LoopState:
    revised = _invoke(
        REVISE_SYSTEM,
        (
            f"Question: {state['task_question']}\n\nDraft:\n{state['draft']}\n\n"
            f"Reviewer feedback:\n{state['critique']}"
        ),
        state,
    )
    return {**state, "draft": revised, "iteration": state["iteration"] + 1}


def record(state: LoopState) -> LoopState:
    """A bookkeeping node: append this round to history before the router decides."""
    round_: Round = {
        "iteration": state["iteration"],
        "draft": state["draft"],
        "critique": state["critique"],
        "checklist_present": (),  # filled in by the benchmark, which has the ground truth
    }
    return {**state, "history": [*state["history"], round_]}


def route(state: LoopState) -> Literal["revise", "end"]:
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    if state["stop_on_complete"] and state["critique"].strip().upper().startswith("COMPLETE"):
        return "end"
    return "revise"


def build_graph():
    graph = StateGraph(LoopState)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("record", record)
    graph.add_node("revise", revise)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "record")
    graph.add_conditional_edges("record", route, {"revise": "revise", "end": END})
    graph.add_edge("revise", "critique")

    return graph.compile()


@dataclass
class RunResult:
    task: Task
    strategy: str
    final_draft: str
    final_critique: str
    iterations: int
    history: list[Round] = field(default_factory=list)
    calls: int = 0
    seconds: float = 0.0

    @property
    def score_trace(self) -> list[float]:
        """Score after each recorded round, in order. This is what shows monotonicity."""
        return [score(round_["draft"], self.task) for round_ in self.history]

    @property
    def final_score(self) -> float:
        return score(self.final_draft, self.task)

    @property
    def is_monotonic(self) -> bool:
        """Did the score never go down across iterations."""
        trace = self.score_trace
        return all(b >= a for a, b in zip(trace, trace[1:], strict=False))

    @property
    def regressed(self) -> bool:
        return not self.is_monotonic

    @property
    def lost_facts(self) -> tuple[str, ...]:
        """Facts present at some point in the loop but absent from the final draft.

        This is the concrete thing "the score can go down" means: a fact the model *had*, that
        a later revision dropped while fixing something else.
        """
        ever_present: set[str] = set()
        for round_ in self.history:
            ever_present.update(present(round_["draft"], self.task))
        final_present = set(present(self.final_draft, self.task))
        return tuple(sorted(ever_present - final_present))


def run(
    task: Task,
    *,
    strategy: str,
    max_iterations: int,
    stop_on_complete: bool,
    model: str,
    ledger: Ledger | None = None,
) -> RunResult:
    graph = build_graph()
    book = Ledger()
    handlers = [book, ledger] if ledger else [book]

    state: LoopState = {
        "task_question": task.question,
        "draft": "",
        "critique": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "stop_on_complete": stop_on_complete,
        "history": [],
        "model": model,
        "ledger": handlers,
    }
    final: LoopState = graph.invoke(state)

    # Fill in the checklist hits per round now that we have the task's ground truth.
    history = [{**r, "checklist_present": present(r["draft"], task)} for r in final["history"]]

    return RunResult(
        task=task,
        strategy=strategy,
        final_draft=final["draft"],
        final_critique=final["critique"],
        iterations=final["iteration"],
        history=history,
        calls=book.n_calls,
        seconds=round(book.seconds, 2),
    )
