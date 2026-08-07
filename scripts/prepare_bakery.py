"""Download the French bakery POS export and map it with ``optistock.ingest``.

Run once, from the repository root::

    python scripts/prepare_bakery.py

It writes the five artifacts described in ``_ingest_artifacts`` into ``data/bakery/``
(git-ignored — the dataset has no open licence), which notebooks 11b and 11c read.

The agents live here rather than in the notebook for two reasons: the run costs up to
six model calls and its output drifts between runs, and a notebook that reads files is
reproducible by someone with no endpoint configured at all.

The ``"."`` placeholder article is deliberately *not* dropped here. That is a fact about
this one bakery, not about the schema, and section 1.6 of notebook 11b makes the point.

Expected outcome: a clean pass. Every canonical field except ``customer_id`` and
``unit_cost`` maps, the ``'0,90 €'`` prices parse via ``numeric_format``, and roughly
1,300 negative-quantity lines are held back as returns.
"""

from __future__ import annotations

from pathlib import Path

from _ingest_artifacts import prepare

DATASET = "matthieugimbert/french-bakery-daily-sales"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "bakery"


if __name__ == "__main__":
    raise SystemExit(prepare(dataset=DATASET, out_dir=OUT_DIR))
