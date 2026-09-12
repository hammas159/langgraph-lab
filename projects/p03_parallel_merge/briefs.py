"""Company briefs for a four-way parallel analysis, and what a complete synthesis must mention.

A fan-out / fan-in graph is the third common LangGraph shape: several analyst nodes run in
parallel, each covering one aspect of the same subject, and a synthesis node reads all of them
into one report. LangGraph itself will not let two branches silently clobber the same state key
— it raises `InvalidUpdateError` unless a reducer is declared — so the naive "silent overwrite"
story does not hold up; that was checked directly rather than assumed. Each analyst here writes
to its own key, which is how a graph like this is actually written, and the error goes away.

What does not go away: nothing stops one branch from **failing** — a tool call timing out, an
API returning an error — while the graph carries on and hands the synthesis node whatever that
branch produced, including nothing. The question this project asks is whether the synthesis
step notices and says so, or writes a fluent report that reads as complete regardless.

Each aspect has a **signature phrase**: distinctive vocabulary a synthesis genuinely covering
that aspect would use. Presence of the signature phrase is how "represented in the final
report" is measured — the same instrument as `langchain-lab`'s checklist scoring, applied here
to a fan-in rather than a fan-out.
"""

from __future__ import annotations

from dataclasses import dataclass

ASPECTS = ("financial", "technical", "market", "team")


@dataclass(frozen=True)
class Brief:
    id: str
    company: str
    #: What each analyst is told about the company, keyed by aspect.
    facts: dict[str, str]
    #: Accepted phrasings per aspect — any one counts as "represented". A single fixed phrase
    #: is not robust: a synthesis paraphrasing "CTO left" as "departure of the CTO" is a correct
    #: mention scored as an omission by an exact-string check with no alternatives listed. Each
    #: tuple's entries were checked against real synthesis output before being finalised, not
    #: chosen and assumed to survive rewording.
    signature: dict[str, tuple[str, ...]]


BRIEFS: tuple[Brief, ...] = (
    Brief(
        id="b1",
        company="Northwind Robotics",
        facts={
            "financial": "Burning $400k/month against $2.1M runway, roughly 5 months left "
            "before the next raise is needed.",
            "technical": "Their warehouse-picking robot has a 94% pick accuracy, below the "
            "99% their main competitor advertises.",
            "market": "Warehouse automation is projected to grow 18% annually through 2030, "
            "with three well-funded competitors already shipping.",
            "team": "Founding CTO left four months ago; the VP Engineering is covering both "
            "roles with no backfill hired yet.",
        },
        signature={
            "financial": ("5 months",),
            "technical": ("94%",),
            "market": ("18%",),
            "team": ("backfill", "CTO left", "left four months"),
        },
    ),
    Brief(
        id="b2",
        company="Ledgerly",
        facts={
            "financial": "Revenue grew 60% year over year but gross margin sits at 22%, well "
            "below the 70%+ typical of software companies.",
            "technical": "Core ledger-reconciliation engine has passed a third-party security "
            "audit with zero critical findings.",
            "market": "Targets mid-size accounting firms, a segment with low switching costs "
            "and heavy price competition from two incumbents.",
            "team": "All four founders have prior fintech exits and have worked together for "
            "over a decade.",
        },
        signature={
            "financial": ("22%",),
            "technical": ("zero critical", "no critical", "critical findings"),
            "market": ("switching cost", "price competition", "two incumbents"),
            "team": ("fintech exit", "decade", "prior fintech"),
        },
    ),
    Brief(
        id="b3",
        company="Verdant Analytics",
        facts={
            "financial": "Profitable since month 8, with 14 months of cash reserves at the "
            "current burn rate even without a new raise.",
            "technical": "Their crop-yield model's prediction error is 11%, worse than the "
            "published 6% benchmark from the leading agri-tech competitor.",
            "market": "Addressable market is concentrated in three countries, all with "
            "recent changes to agricultural subsidy policy.",
            "team": "Strong domain team of former agronomists, but no one with prior "
            "experience scaling a SaaS go-to-market.",
        },
        signature={
            "financial": ("14 months",),
            "technical": ("11%",),
            "market": ("subsidy",),
            "team": ("agronomist", "no prior experience", "scaling a SaaS", "go-to-market"),
        },
    ),
    Brief(
        id="b4",
        company="Cobalt Health",
        facts={
            "financial": "Signed two hospital pilots worth $180k combined, no committed "
            "revenue beyond the pilot period yet.",
            "technical": "Diagnostic model matches specialist accuracy on the internal test "
            "set but has not been validated on an external hospital's data.",
            "market": "Regulatory approval typically takes 18-24 months in this category, "
            "which the current cash position does not comfortably cover.",
            "team": "Clinical advisory board includes two well-known specialists, but the "
            "founding team has no prior regulatory-submission experience.",
        },
        signature={
            "financial": ("$180k", "180k", "180,000"),
            "technical": ("external hospital", "not been validated", "internal test set"),
            "market": ("18-24 months", "18 to 24 months", "regulatory approval"),
            "team": ("no prior regulatory", "advisory board", "regulatory-submission"),
        },
    ),
)

BY_ID = {b.id: b for b in BRIEFS}


def represented(report: str, brief: Brief) -> tuple[str, ...]:
    """Which aspects have at least one accepted phrasing present in the synthesis."""
    lowered = report.lower()
    return tuple(a for a in ASPECTS if any(p.lower() in lowered for p in brief.signature[a]))
