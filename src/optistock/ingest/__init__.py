"""Agent-assisted mapping of a messy source table onto the canonical schema.

Installed by ``pip install optistock[ingest]``. Importing this package is an explicit
opt-in — ``import optistock`` never reaches it, so the core library stays free of an
LLM SDK.

The pipeline, and where the boundary sits::

    read_table ─▶ profile_dataframe ─▶ [describe] ─▶ [map] ─▶ ColumnMapping
                                        agent 1     agent 2         │
                                                                    ▼
                                                              to_canonical
                                                                    │
                                        [review] ◀──────────────────┤
                                        agent 3                     ▼
                                                       optistock.preprocessing

The agents see *only* the profile — column names, dtypes, null rates, cardinality and a
few truncated sample values — plus, for agent 3, the profile and relational facts of the
frame the mapping produced. They never receive the data and never emit it. Everything
downstream of ``ColumnMapping`` is plain pandas, so the whole apply-and-validate half
runs with no model, no network and no API key.

The deterministic half is importable without ``pydantic-ai``; the agent entry points
are loaded lazily on first access.
"""

from .agents import ColumnMapping, FieldMapping, NumericFormat, Review, Strategy
from .apply import ValidationResult, apply_mapping, to_canonical
from .profile import (
    ColumnProfile,
    TableProfile,
    profile_dataframe,
    read_table,
    relational_facts,
)
from .synthetic import make_canonical, scramble

__all__ = [
    # Deterministic — no model needed
    "ColumnMapping",
    "ColumnProfile",
    "FieldMapping",
    "NumericFormat",
    "Review",
    "Strategy",
    "TableProfile",
    "ValidationResult",
    "apply_mapping",
    "make_canonical",
    "profile_dataframe",
    "read_table",
    "relational_facts",
    "scramble",
    "to_canonical",
    # Agents — lazy, need pydantic-ai
    "IngestResult",
    "SourceDescription",
    "build_describer",
    "build_mapper",
    "build_reviewer",
    "describe_source",
    "ingest",
    "map_columns",
    "review_result",
]

_LAZY = {
    "SourceDescription": "agents",
    "build_describer": "agents",
    "build_mapper": "agents",
    "build_reviewer": "agents",
    "describe_source": "agents",
    "map_columns": "agents",
    "review_result": "agents",
    "IngestResult": "run",
    "ingest": "run",
}


def __getattr__(name: str):
    """Load the agent entry points on first use (PEP 562).

    Keeps ``pydantic-ai`` off the import path for anyone using only the deterministic
    transforms, and turns a missing extra into an actionable message rather than a bare
    ModuleNotFoundError.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from importlib import import_module

        module = import_module(f".{module_name}", __name__)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"optistock.ingest.{name} needs pydantic-ai. Install it with "
            "`pip install optistock[ingest]` (or `uv sync --extra ingest`), or use "
            "to_canonical() with a hand-written ColumnMapping for the deterministic "
            "path."
        ) from exc
    return getattr(module, name)
