"""The local model fleet, and what each one is actually good for.

Everything in this repo runs on ollama against models small enough to sit on a 16 GB card
with room left over. That is a deliberate constraint, not a compromise: the failures these
projects are built around are *easier* to observe on a small model, and a reader can
reproduce every number without an API key or a bill.

The registry below is the single source of truth for which models exist. Projects ask for a
capability ("something that can call tools") rather than hard-coding a tag, so adding a model
to the fleet does not mean editing five projects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


@dataclass(frozen=True)
class ModelSpec:
    """One model in the fleet.

    `params_b` and `disk_gb` are recorded because several projects report results per model
    size, and reading that off the tag ("3b") breaks the moment a tag stops containing it.
    """

    tag: str
    params_b: float
    disk_gb: float
    context: int
    tools: bool
    embedding: bool = False
    notes: str = ""
    roles: frozenset[str] = field(default_factory=frozenset)


FLEET: tuple[ModelSpec, ...] = (
    ModelSpec(
        tag="qwen2.5:3b-instruct",
        params_b=3.1,
        disk_gb=1.9,
        context=32768,
        tools=True,
        notes="The default. Native tool calling, and quick enough to keep a UI responsive.",
        roles=frozenset({"general", "tools", "default"}),
    ),
    ModelSpec(
        tag="qwen2.5:7b-instruct",
        params_b=7.6,
        disk_gb=4.7,
        context=32768,
        tools=True,
        notes="The quality ceiling here. Used as the reference answer when a project needs one.",
        roles=frozenset({"general", "tools", "reference"}),
    ),
    ModelSpec(
        tag="llama3.2:3b",
        params_b=3.2,
        disk_gb=2.0,
        context=131072,
        tools=True,
        notes="A second family at the same size. Without it, every cross-model result is "
        "really a result about Qwen.",
        roles=frozenset({"general", "tools", "contrast"}),
    ),
    ModelSpec(
        tag="qwen2.5-coder:3b",
        params_b=3.1,
        disk_gb=1.9,
        context=32768,
        tools=True,
        notes="Code and structured text. Noticeably better at emitting valid JSON.",
        roles=frozenset({"code", "tools", "structured"}),
    ),
    ModelSpec(
        tag="granite3.3:2b",
        params_b=2.5,
        disk_gb=1.5,
        context=131072,
        tools=True,
        notes="The smallest tool-capable model in the fleet. It is here to fail: it marks the "
        "floor where a technique stops working.",
        roles=frozenset({"general", "tools", "floor"}),
    ),
    ModelSpec(
        tag="nomic-embed-text",
        params_b=0.14,
        disk_gb=0.27,
        context=8192,
        tools=False,
        embedding=True,
        notes="768-dimension embeddings. Used by the semantic cache and the retrieval chain.",
        roles=frozenset({"embedding"}),
    ),
)

BY_TAG: dict[str, ModelSpec] = {m.tag: m for m in FLEET}


def with_role(role: str) -> tuple[ModelSpec, ...]:
    """Every model advertising `role`, smallest first.

    Smallest-first matters: benchmarks that iterate the fleet should hit the cheap models
    before the expensive ones, so a run that is going to fail fails quickly.
    """
    return tuple(sorted((m for m in FLEET if role in m.roles), key=lambda m: m.params_b))


def default_chat_model() -> str:
    """The tag used when a project does not care which model it gets."""
    override = os.environ.get("LANGCHAIN_LAB_MODEL")
    if override:
        return override
    return with_role("default")[0].tag


def embedding_model() -> str:
    return with_role("embedding")[0].tag
