"""Download the French bakery POS export and map it with ``optistock.ingest``.

Run once, from the repository root::

    python scripts/prepare_bakery.py

It writes five files into ``data/bakery/`` (git-ignored — the dataset has no open
licence), which notebook 11b then reads:

    orders.csv          the canonical order lines, straight from the agents
    returns.csv         the return lines to_canonical held back before dropping them
    source_profile.txt  the column profile — the entire input the agents were given
    ingest_report.txt   the full run report: every round, every verdict
    ingest_summary.json confidence, review verdict, relational facts, transforms, mapping

The agents live here rather than in the notebook for two reasons: the run costs up to
six model calls and its output drifts between runs, and a notebook that reads files is
reproducible by someone with no endpoint configured at all.

The ``"."`` placeholder article is deliberately *not* dropped here. That is a fact about
this one bakery, not about the schema, and section 1.6 of the notebook makes the point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import kagglehub

from optistock.ingest import ingest, profile_dataframe, read_table

DATASET = "matthieugimbert/french-bakery-daily-sales"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "bakery"

ENDPOINT_HINT = (
    "The ingest agents need a reachable OpenAI-compatible endpoint. Check yours with\n"
    "  optistock-ingest --check\n"
    "and see docs/ingest.md for the INGEST_* environment variables (a .env works)."
)


def _summary(out) -> dict:
    """The parts of the run the notebook prints, as plain JSON-able data."""
    review = out.rounds[-1].review if out.rounds else None
    return {
        "n_rows": len(out.frame),
        "rounds": len(out.rounds),
        "confidence": out.result.confidence,
        "low_confidence": list(out.result.low_confidence),
        "review": {
            "verdict": None if review is None else review.verdict,
            "issues": [] if review is None else list(review.issues),
        },
        "facts": out.result.facts,
        "transforms": out.transforms,
        "mapping": out.mapping.model_dump(),
    }


def main() -> int:
    dataset_dir = Path(kagglehub.dataset_download(DATASET))
    csv_path = next(dataset_dir.glob("*.csv"))
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
    if not out.ok:
        print("\nNo valid canonical frame was produced.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.frame.to_csv(OUT_DIR / "orders.csv", index=False)

    # Written even when empty, so the notebook's read is unconditional.
    returns = out.result.returns
    if returns is None:
        returns = out.frame.iloc[:0]
    returns.to_csv(OUT_DIR / "returns.csv", index=False)

    # Two files rather than one: the profile render contains blank lines, so a notebook
    # splitting a combined report back apart would have to guess where the seam is.
    (OUT_DIR / "source_profile.txt").write_text(
        f"{profile.render()}\n", encoding="utf-8"
    )
    (OUT_DIR / "ingest_report.txt").write_text(f"{out.render()}\n", encoding="utf-8")
    # default=str: `facts` carries Timestamps, which json cannot serialise.
    (OUT_DIR / "ingest_summary.json").write_text(
        json.dumps(_summary(out), indent=2, default=str), encoding="utf-8"
    )

    print(
        f"\nWrote to {OUT_DIR}:\n"
        f"  orders.csv           {len(out.frame):,} canonical order lines\n"
        f"  returns.csv          {len(returns):,} return lines\n"
        f"  source_profile.txt\n"
        f"  ingest_report.txt\n"
        f"  ingest_summary.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
