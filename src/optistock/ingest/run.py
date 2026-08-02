"""The pipeline: profile → describe → map → apply → review, with one retry.

    profile_dataframe   code    what the table looks like, column by column
    describe_source     agent 1 what it appears to be, in prose
    map_columns         agent 2 how each canonical field is derived
    to_canonical        code    execute the mapping, validate the result
    review_result       agent 3 does the result hold what its names claim

Three model calls per round, six at most. Which round is returned is decided here, by
code, never by the model: a revision that fixes the flagged field while breaking another
must not win merely by being last.

There is no cleverer termination rule than the round cap, on purpose. The previous
version had six stop conditions and a scoring function, and a hard cap of two rounds
makes all of them unnecessary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..preprocessing.transforms import available_transforms
from .agents import (
    ColumnMapping,
    Review,
    SourceDescription,
    describe_source,
    describe_source_async,
    map_columns,
    map_columns_async,
    render_mapping,
    review_result,
    review_result_async,
)
from .apply import ValidationResult, to_canonical
from .profile import profile_dataframe, render_facts

__all__ = ["IngestResult", "Round", "ingest", "ingest_async"]


@dataclass
class Round:
    """One propose-apply-review cycle, kept whole so a failed run is diagnosable."""

    mapping: ColumnMapping | None
    result: ValidationResult | None
    review: Review | None = None
    error: str | None = None


@dataclass
class IngestResult:
    """Everything the run produced. ``frame`` is the canonical table, or None."""

    description: SourceDescription | None
    mapping: ColumnMapping | None
    result: ValidationResult | None
    rounds: tuple[Round, ...] = ()
    transforms: dict[str, str | None] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok

    @property
    def frame(self) -> pd.DataFrame | None:
        return None if self.result is None else self.result.frame

    def render(self) -> str:
        lines = [f"ok={self.ok}  rounds={len(self.rounds)}"]
        if self.description is not None:
            lines.append("\n--- source ---\n" + self.description.render())
        for i, round_ in enumerate(self.rounds, 1):
            lines.append(f"\n--- round {i} ---")
            if round_.error:
                lines.append(f"  error: {round_.error}")
            if round_.mapping is not None:
                lines.append(render_mapping(round_.mapping))
            if round_.review is not None:
                lines.append(f"  verdict: {round_.review.verdict}")
                lines += [f"    - {issue}" for issue in round_.review.issues]
        if self.result is not None:
            lines.append("\n--- chosen ---\n" + self.result.render())
        if self.transforms:
            lines.append("\n--- transforms ---")
            lines += [f"  {n:24} {b or 'OK'}" for n, b in self.transforms.items()]
        return "\n".join(lines)


def _better(new: ValidationResult | None, old: ValidationResult | None) -> bool:
    """Is *new* worth keeping over *old*?

    Validity first, then rows. A round that validates beats one that does not; between
    two that both validate, the one that kept more rows wins, because the usual way a
    "correction" goes wrong is mapping quantity to a column full of zeros and losing
    half the order book to ``drop_invalid_rows``.
    """
    if new is None:
        return False
    if old is None:
        return True
    if new.ok != old.ok:
        return new.ok
    # `frame or ()` would call bool() on a DataFrame, which raises.
    return _n_rows(new) > _n_rows(old)


def _n_rows(result: ValidationResult) -> int:
    return 0 if result.frame is None else len(result.frame)


def ingest(
    df: pd.DataFrame,
    *,
    max_rounds: int = 2,
    model=None,
    describe_model=None,
    review_model=None,
    output_mode=None,
    drop_invalid: bool = True,
    describe: bool = True,
) -> IngestResult:
    """Map a raw source table onto the canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        The raw source table. Never mutated, and never sent to a model — only its
        profile is.
    max_rounds : int
        Mapper attempts. ``1`` disables the retry while still returning a review.
    model
        Agent 2, the mapper — where mapping quality is actually decided. Defaults to
        ``INGEST_MODEL``.
    describe_model
        Agent 1. Defaults to ``INGEST_DESCRIBE_MODEL``, else the mapper's model. Its
        output is 120 words of prose that nothing parses, so this is the natural home
        for the cheapest model available.
    review_model
        Agent 3. Defaults to ``INGEST_REVIEW_MODEL``, else the mapper's model.
        Reviewing a model with itself is the weakest form of review — it inherits the
        blind spots that produced the mapping.
    describe : bool
        Run agent 1 at all. Off saves a call per run at the cost of the mapper's
        context.

    Returns
    -------
    IngestResult
        With every round on ``rounds``, the best round's mapping and result, and
        ``transforms`` naming which of to_sales/to_items/to_clv/to_returns the frame
        can feed.

    Notes
    -----
    Synchronous, so it cannot run inside an event loop that is already going — which
    is every Jupyter cell. Use :func:`ingest_async` with ``await`` there.
    """
    return _drive(
        df,
        _SyncCalls(model, describe_model, review_model, output_mode),
        max_rounds=max_rounds,
        drop_invalid=drop_invalid,
        describe=describe,
    )


async def ingest_async(
    df: pd.DataFrame,
    *,
    max_rounds: int = 2,
    model=None,
    describe_model=None,
    review_model=None,
    output_mode=None,
    drop_invalid: bool = True,
    describe: bool = True,
) -> IngestResult:
    """Async variant of :func:`ingest` — the one to use in Jupyter::

        out = await ingest_async(raw)

    ``ingest`` calls PydanticAI's ``run_sync``, which raises
    ``RuntimeError: This event loop is already running`` in a notebook. Identical
    behaviour otherwise: the loop, the round cap and the choice of round are shared
    with the sync entry point, so the two cannot drift.
    """
    calls = _AsyncCalls(model, describe_model, review_model, output_mode)
    return await _adrive(
        df, calls, max_rounds=max_rounds, drop_invalid=drop_invalid, describe=describe
    )


# ---------------------------------------------------------------------------
# The loop, written once
# ---------------------------------------------------------------------------
#
# The sync and async entry points differ only in how the three agent calls are awaited,
# so each supplies a small object with `describe`/`map`/`review` and the loop is written
# against that.
#
# `_drive` and `_adrive` are then deliberate near-duplicates: `_adrive` is `_drive` with
# `async`/`await` added and nothing else. Python offers no clean way to write one body
# that works both ways, and the alternatives (running the sync path in a thread, or
# `asyncio.run` from the async one) either block a notebook's loop or nest loops. The
# duplication is the least bad option, and the risk it carries — the termination logic
# drifting between the two — is guarded by
# tests/ingest/test_agents.py::test_the_sync_and_async_drivers_have_identical_logic,
# which normalises the awaits away and compares the bodies. Change one, change both.


@dataclass
class _SyncCalls:
    model: object = None
    describe_model: object = None
    review_model: object = None
    output_mode: object = None

    def describe(self, source):
        return describe_source(source, self.describe_model, self.output_mode)

    def map(self, source, description, previous, issues):
        return map_columns(
            source, self.model, self.output_mode,
            description=description, previous=previous, issues=issues,
        )

    def review(self, source, mapping, result):
        return review_result(
            source, mapping, result,
            model=self.review_model, output_mode=self.output_mode,
        )


@dataclass
class _AsyncCalls:
    model: object = None
    describe_model: object = None
    review_model: object = None
    output_mode: object = None

    async def describe(self, source):
        return await describe_source_async(source, self.describe_model, self.output_mode)

    async def map(self, source, description, previous, issues):
        return await map_columns_async(
            source, self.model, self.output_mode,
            description=description, previous=previous, issues=issues,
        )

    async def review(self, source, mapping, result):
        return await review_result_async(
            source, mapping, result,
            model=self.review_model, output_mode=self.output_mode,
        )


def _apply_crash_feedback(exc: Exception) -> list[str]:
    return [
        f"Applying your mapping raised {type(exc).__name__}: {exc}. "
        "Return a corrected mapping that does not raise."
    ]


def _finish(description, rounds, best) -> IngestResult:
    chosen = best.result if best else None
    return IngestResult(
        description=description,
        mapping=best.mapping if best else (rounds[-1].mapping if rounds else None),
        result=chosen,
        rounds=tuple(rounds),
        transforms=_transforms(chosen),
    )


def _drive(df, calls, *, max_rounds, drop_invalid, describe) -> IngestResult:
    source = profile_dataframe(df)
    description = calls.describe(source) if describe else None

    rounds: list[Round] = []
    best: Round | None = None
    previous: ColumnMapping | None = None
    issues: list[str] | None = None

    for _ in range(max(max_rounds, 1)):
        try:
            mapping = calls.map(source, description, previous, issues)
        except Exception as exc:  # noqa: BLE001 - a failed proposal ends the run
            rounds.append(Round(None, None, error=f"{type(exc).__name__}: {exc}"))
            break

        try:
            result = to_canonical(df, mapping, drop_invalid=drop_invalid)
        except Exception as exc:  # noqa: BLE001 - a crash is a defect, not the end
            rounds.append(Round(mapping, None, error=f"{type(exc).__name__}: {exc}"))
            previous, issues = mapping, _apply_crash_feedback(exc)
            continue

        try:
            review = calls.review(source, mapping, result)
        except Exception as exc:  # noqa: BLE001 - no reviewer, but the mapping stands
            rounds.append(
                Round(mapping, result, error=f"review: {type(exc).__name__}: {exc}")
            )
            if _better(result, best.result if best else None):
                best = rounds[-1]
            break

        rounds.append(Round(mapping, result, review))
        if _better(result, best.result if best else None):
            best = rounds[-1]

        if review.verdict == "approve" or not review.issues:
            break
        previous, issues = mapping, list(review.issues)

    return _finish(description, rounds, best)


async def _adrive(df, calls, *, max_rounds, drop_invalid, describe) -> IngestResult:
    """Mirror of :func:`_drive` with the three agent calls awaited."""
    source = profile_dataframe(df)
    description = await calls.describe(source) if describe else None

    rounds: list[Round] = []
    best: Round | None = None
    previous: ColumnMapping | None = None
    issues: list[str] | None = None

    for _ in range(max(max_rounds, 1)):
        try:
            mapping = await calls.map(source, description, previous, issues)
        except Exception as exc:  # noqa: BLE001
            rounds.append(Round(None, None, error=f"{type(exc).__name__}: {exc}"))
            break

        try:
            result = to_canonical(df, mapping, drop_invalid=drop_invalid)
        except Exception as exc:  # noqa: BLE001
            rounds.append(Round(mapping, None, error=f"{type(exc).__name__}: {exc}"))
            previous, issues = mapping, _apply_crash_feedback(exc)
            continue

        try:
            review = await calls.review(source, mapping, result)
        except Exception as exc:  # noqa: BLE001
            rounds.append(
                Round(mapping, result, error=f"review: {type(exc).__name__}: {exc}")
            )
            if _better(result, best.result if best else None):
                best = rounds[-1]
            break

        rounds.append(Round(mapping, result, review))
        if _better(result, best.result if best else None):
            best = rounds[-1]

        if review.verdict == "approve" or not review.issues:
            break
        previous, issues = mapping, list(review.issues)

    return _finish(description, rounds, best)


def _transforms(result: ValidationResult | None) -> dict[str, str | None]:
    """Which transforms the chosen frame can feed.

    ``to_returns`` is corrected from ``result.returns`` rather than read off the frame:
    ``drop_invalid`` has already removed every negative quantity by this point, so
    asking the frame would always answer "no returns here" on exactly the sources that
    have them.
    """
    if result is None or result.frame is None:
        return {}
    out = available_transforms(result.frame)
    if result.returns is not None and len(result.returns):
        out["to_returns"] = None
    return out
