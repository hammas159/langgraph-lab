"""Getting a chat model, and counting what it cost to use.

Two things live here that every project needs:

1. `chat()` — a `ChatOllama` built from the fleet registry, with the sampling settings pinned.
   Temperature defaults to 0. Every measurement in this repo is a comparison, and a
   comparison against a randomly sampled baseline measures the sampler.

2. `Ledger` — a callback that records each call: model, latency, token counts. Projects report
   "this technique costs 2.4x more calls" and that claim has to come from somewhere other
   than an estimate.

There is deliberately no retry-with-backoff wrapper. A local model does not rate-limit you,
and hiding failures behind retries is how a project ends up reporting a success rate that
belongs to the retry loop rather than the model.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_ollama import ChatOllama, OllamaEmbeddings

from shared.models import BY_TAG, OLLAMA_HOST, default_chat_model, embedding_model


@dataclass
class Call:
    model: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(eq=False)
class Ledger(BaseCallbackHandler):
    """Records every model call made while it is attached.

    Attach one per experiment, not one per process: the numbers are only meaningful when the
    boundary of what is being counted is explicit.

    `eq=False` is load-bearing. A plain `@dataclass` generates `__eq__` from the fields, so two
    freshly created Ledgers — both with an empty `calls` list — compare **equal**. LangChain's
    callback manager deduplicates handlers, so passing `callbacks=[inner, outer]` silently
    dropped the second one and the outer ledger reported zero calls for work that had actually
    happened. Identity comparison is what a callback handler needs; equality by contents is
    meaningless for it.
    """

    calls: list[Call] = field(default_factory=list)
    _starts: dict[Any, float] = field(default_factory=dict)

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id=None, **kwargs) -> None:
        started = self._starts.pop(run_id, None)
        elapsed = time.perf_counter() - started if started is not None else 0.0

        model = ""
        prompt_tokens = completion_tokens = 0
        info = response.llm_output or {}
        model = info.get("model_name") or info.get("model") or ""

        # Ollama reports usage on the message itself rather than in llm_output.
        for generation in response.generations:
            for item in generation:
                message = getattr(item, "message", None)
                usage = getattr(message, "usage_metadata", None) if message else None
                if usage:
                    prompt_tokens += usage.get("input_tokens", 0) or 0
                    completion_tokens += usage.get("output_tokens", 0) or 0
                meta = getattr(message, "response_metadata", {}) if message else {}
                model = model or meta.get("model", "")

        self.calls.append(
            Call(
                model=model,
                seconds=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        self._starts.pop(run_id, None)

    # --- reporting -----------------------------------------------------------------

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def seconds(self) -> float:
        return sum(c.seconds for c in self.calls)

    @property
    def tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    def reset(self) -> None:
        self.calls.clear()
        self._starts.clear()


@contextmanager
def ledger():
    """`with ledger() as book:` — scope a measurement to a block."""
    book = Ledger()
    try:
        yield book
    finally:
        book._starts.clear()


def chat(
    model: str | None = None,
    *,
    temperature: float = 0.0,
    num_ctx: int | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    **kwargs: Any,
) -> ChatOllama:
    """A chat model from the fleet.

    `num_ctx` defaults to the context length the registry records for the model. Ollama's own
    default is 2048 regardless of what the model supports, which silently truncates long
    prompts — that is the single most common reason a local-model experiment produces a
    result that cannot be reproduced against a hosted model.
    """
    tag = model or default_chat_model()
    spec = BY_TAG.get(tag)
    if num_ctx is None and spec is not None:
        num_ctx = min(spec.context, 8192)

    return ChatOllama(
        model=tag,
        base_url=OLLAMA_HOST,
        temperature=temperature,
        num_ctx=num_ctx,
        callbacks=callbacks,
        **kwargs,
    )


def embeddings(model: str | None = None) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model or embedding_model(), base_url=OLLAMA_HOST)


def server_is_up(timeout: float = 2.0) -> bool:
    """Whether ollama is reachable. Used to skip live tests rather than fail them."""
    import httpx

    try:
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def installed_tags() -> set[str]:
    """Model tags present locally, each in both its bare and `:latest` spelling.

    Ollama reports a model pulled as `nomic-embed-text` under the name
    `nomic-embed-text:latest`. Registry entries use the bare form, so a plain set membership
    test silently reports a pulled model as missing — and the benchmark then skips it without
    saying anything, which is the worst possible failure mode for a results table.
    """
    import httpx

    try:
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        response.raise_for_status()
    except Exception:
        return set()

    tags: set[str] = set()
    for model in response.json().get("models", []):
        name = model["name"]
        tags.add(name)
        if name.endswith(":latest"):
            tags.add(name.removesuffix(":latest"))
        else:
            tags.add(f"{name}:latest")
    return tags
