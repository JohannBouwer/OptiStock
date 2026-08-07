"""Shared machinery for the ``prepare_*.py`` scripts.

Not runnable. Every prep script does the same four things — fetch a csv from Kaggle,
profile it, run the ingest agents over it, write what they produced to ``data/<name>/``
— so it lives here once and each script is reduced to a dataset slug and an output
directory.

The five files written are the same for every source, and notebooks 11b and 11c read
them::

    orders.csv          the canonical order lines the mapping produced
    returns.csv         the return lines to_canonical held back before dropping them
    source_profile.txt  the column profile — the entire input the agents were given
    ingest_report.txt   the full run report: every round, every verdict
    ingest_summary.json ok, error, confidence, review, relational facts, transforms, mapping

**They are written even when the run fails.** A source that cannot support the canonical
schema is a result worth keeping, not an error to swallow: ``to_canonical`` hands back
the frame it built alongside ``ok=False``, and that frame is the most interesting thing
about the run. So the exit code answers "did the run happen", not "was it any good" —
1 means the download or the endpoint failed and there is nothing on disk to read, and a
failed *mapping* still returns 0 with a banner on stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import kagglehub
import pandas as pd

from optistock.ingest import ingest, profile_dataframe, read_table

ENDPOINT_HINT = (
    "The ingest agents need a reachable OpenAI-compatible endpoint. Check yours with\n"
    "  optistock-ingest --check\n"
    "and see docs/ingest.md for the INGEST_* environment variables (a .env works)."
)


def summarise(out) -> dict:
    """The parts of the run the notebooks print, as plain JSON-able data.

    Every field is optional-tolerant: a mapper crash leaves ``result`` and ``mapping``
    as None (``run.py`` ``_finish``), and a summary that raises on the one run worth
    reading would defeat the point of writing it.
    """
    review = out.rounds[-1].review if out.rounds else None
    result = out.result
    return {
        "ok": out.ok,
        "error": None if result is None else result.error,
        "n_rows": 0 if out.frame is None else len(out.frame),
        "rounds": len(out.rounds),
        "confidence": {} if result is None else result.confidence,
        "low_confidence": [] if result is None else list(result.low_confidence),
        "review": {
            "verdict": None if review is None else review.verdict,
            "issues": [] if review is None else list(review.issues),
        },
        "facts": {} if result is None else result.facts,
        "transforms": out.transforms,
        "mapping": None if out.mapping is None else out.mapping.model_dump(),
    }


def _write(out, profile, out_dir: Path) -> int:
    """Write the five artifacts. Returns the number of canonical rows written."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # `frame` is not None even when ok=False — it is what the mapping actually built,
    # which is exactly what someone diagnosing the failure wants to look at.
    # An empty frame only when the mapper crashed before proposing anything at all.
    frame = pd.DataFrame() if out.frame is None else out.frame
    frame.to_csv(out_dir / "orders.csv", index=False)

    # Written even when empty, so the notebooks' read is unconditional.
    returns = None if out.result is None else out.result.returns
    if returns is None:
        returns = frame.iloc[:0]
    returns.to_csv(out_dir / "returns.csv", index=False)

    # Two files rather than one: the profile render contains blank lines, so a notebook
    # splitting a combined report back apart would have to guess where the seam is.
    (out_dir / "source_profile.txt").write_text(
        f"{profile.render()}\n", encoding="utf-8"
    )
    (out_dir / "ingest_report.txt").write_text(f"{out.render()}\n", encoding="utf-8")
    # default=str: `facts` carries Timestamps, which json cannot serialise.
    (out_dir / "ingest_summary.json").write_text(
        json.dumps(summarise(out), indent=2, default=str), encoding="utf-8"
    )

    print(
        f"\nWrote to {out_dir}:\n"
        f"  orders.csv           {len(frame):,} canonical order lines\n"
        f"  returns.csv          {len(returns):,} return lines\n"
        f"  source_profile.txt\n"
        f"  ingest_report.txt\n"
        f"  ingest_summary.json"
    )
    return len(frame)


def prepare(*, dataset: str, out_dir: Path, filename: str | None = None) -> int:
    """Download, map with the agents, and write the artifacts. Returns an exit code.

    Parameters
    ----------
    dataset : str
        Kaggle dataset slug, e.g. ``"olistbr/brazilian-ecommerce"``.
    out_dir : Path
        Where the five artifacts go. Created if absent.
    filename : str | None
        The csv to read from the download. ``None`` takes the only one, which is right
        for a single-file dataset and wrong for a multi-table one — name the file
        whenever the dataset has more than one.
    """
    try:
        dataset_dir = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:  # noqa: BLE001 - surface whatever kagglehub said
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    csv_path = dataset_dir / filename if filename else next(dataset_dir.glob("*.csv"))
    if not csv_path.exists():
        print(f"{csv_path} not found in the download.", file=sys.stderr)
        return 1
    print(f"Downloaded to: {csv_path}")

    raw = read_table(csv_path)
    profile = profile_dataframe(raw)
    print(f"Read {len(raw):,} rows x {raw.shape[1]} columns. Running the agents...\n")

    try:
        out = ingest(raw)
    except Exception as exc:  # noqa: BLE001 - surface whatever the endpoint said
        print(f"{type(exc).__name__}: {exc}\n\n{ENDPOINT_HINT}", file=sys.stderr)
        return 1

    print(out.render())
    _write(out, profile, out_dir)

    if not out.ok:
        print(
            "\nNo valid canonical frame was produced. The artifacts above record why —"
            "\nread the 'error' key of ingest_summary.json and the chosen round of"
            "\ningest_report.txt. orders.csv holds the frame the mapping did build.",
            file=sys.stderr,
        )
    return 0
