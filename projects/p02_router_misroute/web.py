"""The UI for project 02.

Runs the router live and shows every vote (for multi_vote) or the single classification (for
the other strategies) beside the confidence it reported, so a confident misroute is visible as
exactly that — a HIGH-confidence tag sitting next to a wrong category.

    uv run uvicorn projects.p02_router_misroute.web:app --reload --port 8112
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p02_router_misroute.benchmark import STRATEGIES
from projects.p02_router_misroute.graph import run
from projects.p02_router_misroute.tickets import BY_ID, TICKETS
from shared.llm import installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Does the router know when it's guessing?")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "tickets": TICKETS,
        "strategies": STRATEGIES,
        "models": _models(),
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p02_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(
    request: Request,
    ticket_id: str = Form(...),
    strategy: str = Form("__all__"),
    model: str = Form(...),
):
    ticket = BY_ID[ticket_id]
    chosen = list(STRATEGIES) if strategy == "__all__" else [strategy]

    results = []
    for label in chosen:
        try:
            r = run(ticket, strategy=label, model=model)
        except Exception as exc:
            results.append({"label": label, "crashed": f"{type(exc).__name__}: {exc}"})
            continue
        results.append({"label": label, "crashed": None, "r": r})

    return templates.TemplateResponse(
        request,
        "p02_index.html",
        _context(
            request,
            results=results,
            ticket=ticket,
            chosen_ticket=ticket_id,
            chosen_strategy=strategy,
            chosen_model=model,
        ),
    )
