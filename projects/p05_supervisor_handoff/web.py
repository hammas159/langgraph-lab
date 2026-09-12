"""The UI for project 05.

Runs the real handoff graph live under any strategy and shows exactly what context the
specialist received beside its response, so a dropped constraint is visible as a genuine gap
between "what the customer said" and "what the second agent was actually given."

    uv run uvicorn projects.p05_supervisor_handoff.web:app --reload --port 8115
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p05_supervisor_handoff.benchmark import STRATEGIES
from projects.p05_supervisor_handoff.cases import BY_ID, CASES
from projects.p05_supervisor_handoff.graph import run
from shared.llm import installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Does the constraint survive the handoff?")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "cases": CASES,
        "strategies": STRATEGIES,
        "models": _models(),
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p05_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(
    request: Request,
    case_id: str = Form(...),
    strategy: str = Form("__all__"),
    model: str = Form(...),
):
    case = BY_ID[case_id]
    chosen = list(STRATEGIES) if strategy == "__all__" else [strategy]

    results = []
    for label in chosen:
        try:
            r = run(case, strategy=label, model=model)
        except Exception as exc:
            results.append({"label": label, "crashed": f"{type(exc).__name__}: {exc}"})
            continue
        results.append({"label": label, "crashed": None, "r": r})

    return templates.TemplateResponse(
        request,
        "p05_index.html",
        _context(
            request,
            results=results,
            case=case,
            chosen_case=case_id,
            chosen_strategy=strategy,
            chosen_model=model,
        ),
    )
