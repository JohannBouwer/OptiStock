"""Model + output-mode wiring — provider-agnostic via the OpenAI-compatible interface.

Any OpenAI-compatible ``/v1`` endpoint works unchanged: a local server (Ollama,
LM Studio, vLLM) or a hosted one (OpenRouter, Google AI Studio). Only env vars change.

Output mode matters for small open models
-----------------------------------------
PydanticAI defaults to *tool* mode for structured output — it asks the model to call a
function. Models without native tool tokens (notably **Gemma**) can't do that reliably.
``native`` mode instead uses the provider's JSON-schema constrained decoding, which
enforces the schema during sampling and needs no tool support. Hence ``native`` is the
default here.

That choice constrains the output models in :mod:`~optistock.ingest.agents`: providers
restrict what constrained decoding accepts and reject the *whole request* rather than
ignoring a keyword they dislike. Keep ``minimum``/``maximum``/``pattern``/``minLength``
out of those schemas and validate in Python instead.

One model per agent
-------------------
The three agents do very different amounts of work, so they get their own slots and
each falls back to ``INGEST_MODEL``:

    agent 1 describe  INGEST_DESCRIBE_MODEL   120 words of prose — put the cheapest
                                              model you have here
    agent 2 map       INGEST_MODEL            where mapping quality is decided
    agent 3 review    INGEST_REVIEW_MODEL     point this somewhere other than the
                                              mapper; reviewing a model with itself
                                              inherits the blind spots that produced
                                              the mapping

All three share ``INGEST_BASE_URL`` and ``INGEST_API_KEY``, so by default they select
different models on the *same* endpoint. For a genuinely different provider, build the
model yourself and pass it in::

    ingest(df, describe_model=build_model("m", base_url="https://other.example/v1"))

Environment variables
----------------------
INGEST_MODEL           model name (default: ``gemma3:4b``)
INGEST_BASE_URL        OpenAI-compatible endpoint (default: local Ollama)
INGEST_API_KEY         API key, if the endpoint needs one
INGEST_OUTPUT_MODE     ``native`` | ``tool`` | ``prompted`` (default: ``native``)
INGEST_DESCRIBE_MODEL  model used by agent 1 (default: the same as INGEST_MODEL)
INGEST_REVIEW_MODEL    model used by agent 3 (default: the same as INGEST_MODEL)
"""

from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic_ai import NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load a local .env if present so API keys never have to be pasted into code.
load_dotenv()

DEFAULT_MODEL = "gemma3:4b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
DEFAULT_OUTPUT_MODE = "native"

OutputMode = Literal["native", "tool", "prompted"]

# Handy presets — pass to build_model(base_url=...) or set INGEST_BASE_URL.
KNOWN_ENDPOINTS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_MODE",
    "KNOWN_ENDPOINTS",
    "OutputMode",
    "build_describe_model",
    "build_model",
    "build_review_model",
    "describe_model_name",
    "review_model_name",
    "wrap_output",
]


def build_model(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAIChatModel:
    """Construct the chat model from explicit args or environment variables."""
    model_name = model_name or os.getenv("INGEST_MODEL", DEFAULT_MODEL)
    base_url = base_url or os.getenv("INGEST_BASE_URL", DEFAULT_BASE_URL)
    # Allow shorthand: INGEST_BASE_URL=openrouter
    base_url = KNOWN_ENDPOINTS.get(base_url, base_url)
    # Local servers accept any non-empty key; keep a harmless placeholder default.
    api_key = api_key or os.getenv("INGEST_API_KEY", "not-needed")

    return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


def wrap_output(output_type: type, mode: OutputMode | None = None) -> Any:
    """Wrap a structured output type in the configured output mode.

    - ``native``   → provider-side JSON-schema constrained decoding (best for Gemma and
      other models lacking tool tokens).
    - ``tool``     → function calling (PydanticAI's default; needs tool support).
    - ``prompted`` → schema described in the prompt, response parsed as JSON.
    """
    mode = mode or os.getenv("INGEST_OUTPUT_MODE", DEFAULT_OUTPUT_MODE)  # type: ignore[assignment]
    if mode == "native":
        return NativeOutput(output_type)
    if mode == "tool":
        return ToolOutput(output_type)
    if mode == "prompted":
        return PromptedOutput(output_type)
    raise ValueError(
        f"Unknown INGEST_OUTPUT_MODE {mode!r} (expected 'native', 'tool' or 'prompted')"
    )


def _agent_model_name(env_var: str) -> str:
    """Resolve one agent's model, falling back to the mapper's.

    ``or`` rather than a ``getenv`` default, so an empty value in a ``.env`` falls back
    instead of being sent to the provider as an empty model name — a commented-out line
    left as ``INGEST_REVIEW_MODEL=`` is the usual way that happens.
    """
    return os.getenv(env_var) or os.getenv("INGEST_MODEL", DEFAULT_MODEL)


def describe_model_name() -> str:
    """The model agent 1 uses. Its whole job is 120 words of prose."""
    return _agent_model_name("INGEST_DESCRIBE_MODEL")


def review_model_name() -> str:
    """The model agent 3 uses."""
    return _agent_model_name("INGEST_REVIEW_MODEL")


def build_describe_model(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAIChatModel:
    """Construct the describing model (agent 1)."""
    return build_model(model_name or describe_model_name(), base_url, api_key)


def build_review_model(
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAIChatModel:
    """Construct the reviewing model (agent 3)."""
    return build_model(model_name or review_model_name(), base_url, api_key)
