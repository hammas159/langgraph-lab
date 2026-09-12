"""The UI for project 01.

Runs the actual LangGraph graph live and shows the score after every iteration as a trace, so
a regression — the score going down between two revisions — is visible as a dip rather than
hidden inside a final number.

    uv run uvicorn projects.p01_revision_loops.web:app --reload --port 8111
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p01_revision_loops.benchmark import STRATEGIES
from projects.p01_revision_loops.graph import run
from projects.p01_revision_loops.tasks import BY_ID, TASKS
from shared.llm import installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Does the revision loop actually converge?")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "tasks": TASKS,
        "strategies": list(STRATEGIES),
        "models": _models(),
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p01_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(
    request: Request,
    task_id: str = Form(...),
    strategy: str = Form("__all__"),
    model: str = Form(...),
):
    task = BY_ID[task_id]
    chosen = list(STRATEGIES) if strategy == "__all__" else [strategy]

    results = []
    for label in chosen:
        kwargs = STRATEGIES[label]
        try:
            r = run(task, strategy=label, model=model, **kwargs)
        except Exception as exc:
            results.append({"label": label, "crashed": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(
            {
                "label": label,
                "crashed": None,
                "r": r,
                "trace": r.score_trace,
                "final_score": r.final_score,
                "regressed": r.regressed,
                "lost_facts": r.lost_facts,
            }
        )

    return templates.TemplateResponse(
        request,
        "p01_index.html",
        _context(
            request,
            results=results,
            task=task,
            chosen_task=task_id,
            chosen_strategy=strategy,
            chosen_model=model,
        ),
    )
