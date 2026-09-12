"""Support tickets with a ground-truth category, split into clean and cross-cutting.

Supervisor / router graphs are the second most common LangGraph pattern after the revision
cycle: a node classifies an incoming request and dispatches it to one of several specialist
nodes. Every demo shows it on requests where the category is obvious. Real tickets are not
all obvious — a customer can be billed incorrectly *and* locked out at the same time — and the
question this project asks is what the router does with those, and whether it *knows* it is
unsure when it is.

Tickets are split into two groups on purpose:

    clean          a single category clearly applies. The router should get these right, and
                   a router that does not is simply broken.
    cross_cutting  more than one category has a real claim. There is no dishonest trick here —
                   these are ordinary support tickets a real inbox receives — but there is a
                   `primary` category a support team would actually route it to first, which is
                   the label the router is scored against.
"""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = ("billing", "technical", "account", "general")


@dataclass(frozen=True)
class Ticket:
    id: str
    text: str
    primary: str
    kind: str  # "clean" | "cross_cutting"
    #: For cross-cutting tickets, every category with a genuine claim on it. Empty for clean
    #: tickets, where only `primary` applies.
    also: tuple[str, ...] = ()


TICKETS: tuple[Ticket, ...] = (
    Ticket(
        id="c1",
        text="I was charged twice for my subscription this month. Can I get a refund for the duplicate charge?",
        primary="billing",
        kind="clean",
    ),
    Ticket(
        id="c2",
        text="The app crashes every time I try to open the settings page. It just closes immediately.",
        primary="technical",
        kind="clean",
    ),
    Ticket(
        id="c3",
        text="I want to change the email address on my account to a new one.",
        primary="account",
        kind="clean",
    ),
    Ticket(
        id="c4",
        text="What are your business hours and how do I contact support by phone?",
        primary="general",
        kind="clean",
    ),
    Ticket(
        id="c5",
        text="My invoice from last month shows the wrong amount and I'd like it corrected.",
        primary="billing",
        kind="clean",
    ),
    Ticket(
        id="c6",
        text="The export feature has been broken since the last update and produces empty files.",
        primary="technical",
        kind="clean",
    ),
    Ticket(
        id="x1",
        text="I was charged for the premium plan but I can't log in to actually use any of the "
        "features I'm paying for.",
        primary="billing",
        kind="cross_cutting",
        also=("technical",),
    ),
    Ticket(
        id="x2",
        text="I tried to update my payment card and now my whole account is locked and I can't "
        "access anything, including billing.",
        primary="account",
        kind="cross_cutting",
        also=("billing", "technical"),
    ),
    Ticket(
        id="x3",
        text="Your app keeps signing me out every few minutes, and each time it happens I get "
        "charged again for what looks like a new subscription.",
        primary="technical",
        kind="cross_cutting",
        also=("billing",),
    ),
    Ticket(
        id="x4",
        text="I cancelled my plan last week but I was still billed, and now I also can't reset "
        "my password to get into the account to check anything.",
        primary="billing",
        kind="cross_cutting",
        also=("account", "technical"),
    ),
    Ticket(
        id="x5",
        text="Is there a way to downgrade my plan, and if I do, will I lose access to my saved "
        "files immediately or at the end of the billing period?",
        primary="billing",
        kind="cross_cutting",
        also=("account",),
    ),
    Ticket(
        id="x6",
        text="Someone else seems to have access to my account and changed my email, and now I'm "
        "worried they can see my payment details too.",
        primary="account",
        kind="cross_cutting",
        also=("billing", "technical"),
    ),
)

BY_ID = {t.id: t for t in TICKETS}


def clean() -> tuple[Ticket, ...]:
    return tuple(t for t in TICKETS if t.kind == "clean")


def cross_cutting() -> tuple[Ticket, ...]:
    return tuple(t for t in TICKETS if t.kind == "cross_cutting")


def is_acceptable(ticket: Ticket, chosen: str) -> bool:
    """Whether routing to `chosen` is defensible, not just exactly `primary`.

    Used only for reporting a looser "plausible" rate alongside the strict one — the strict
    `chosen == ticket.primary` check is what every other number in this project uses.
    """
    return chosen == ticket.primary or chosen in ticket.also
