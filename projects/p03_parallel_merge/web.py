"""The UI for project 03.

Runs the real fan-out graph live: four analyst branches in parallel, then a synthesis node.
Shows every branch's finding (empty ones rendered visibly empty) beside the synthesised report,
so a silently dropped branch and a smooth-reading report sit next to each other on one screen.

    uv run uvicorn projects.p03_parallel_merge.web:app --reload --port 8113
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p03_parallel_merge.briefs import ASPECTS, BRIEFS, BY_ID
from projects.p03_parallel_merge.graph import run
from shared.llm import installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Does the report admit a branch failed?")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "briefs": BRIEFS,
        "aspects": ASPECTS,
        "models": _models(),
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p03_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(
    request: Request,
    brief_id: str = Form(...),
    fail_aspect: str = Form(""),
    flag_gaps: bool = Form(False),
    model: str = Form(...),
):
    brief = BY_ID[brief_id]
    try:
        r = run(brief, strategy="ui", fail_aspect=fail_aspect, flag_gaps=flag_gaps, model=model)
        result = {
            "crashed": None,
            "r": r,
            "represented": r.represented_aspects,
            "disclosed": r.missing_aspect_disclosed,
            "reads_complete": r.reads_as_complete,
        }
    except Exception as exc:
        result = {"crashed": f"{type(exc).__name__}: {exc}"}

    return templates.TemplateResponse(
        request,
        "p03_index.html",
        _context(
            request,
            result=result,
            brief=brief,
            chosen_brief=brief_id,
            chosen_fail=fail_aspect,
            chosen_flag=flag_gaps,
            chosen_model=model,
        ),
    )
