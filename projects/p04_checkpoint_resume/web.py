"""The UI for project 04.

Runs both resume strategies live on the same scenario and shows every drafting call each one
made, in order, so the naive strategy's extra call at the very end — the one that only exists
to say "go ahead" and re-drafts anyway — is visible as a distinct, wasted step.

    uv run uvicorn projects.p04_checkpoint_resume.web:app --reload --port 8114
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p04_checkpoint_resume.graph import run_checkpointed, run_naive
from projects.p04_checkpoint_resume.scenarios import BY_ID, SCENARIOS
from shared.llm import installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Does resuming re-run work that already happened?")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "scenarios": SCENARIOS,
        "models": _models(),
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p04_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(request: Request, scenario_id: str = Form(...), model: str = Form(...)):
    scenario = BY_ID[scenario_id]
    try:
        ck = run_checkpointed(scenario, model=model)
        nv = run_naive(scenario, model=model)
        result = {"crashed": None, "ck": ck, "nv": nv, "extra": nv.draft_calls - ck.draft_calls}
    except Exception as exc:
        result = {"crashed": f"{type(exc).__name__}: {exc}"}

    return templates.TemplateResponse(
        request,
        "p04_index.html",
        _context(
            request,
            result=result,
            scenario=scenario,
            chosen_scenario=scenario_id,
            chosen_model=model,
        ),
    )
