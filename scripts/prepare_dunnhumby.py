"""Download the dunnhumby transaction table and map it with ``optistock.ingest``.

Run once, from the repository root::

    python scripts/prepare_dunnhumby.py

Writes the five artifacts described in ``_ingest_artifacts`` into ``data/dunnhumby/``
for notebook 11c. Git-ignored: the dataset is distributed under dunnhumby's own terms.

**Source:** "The Complete Journey" (Kaggle mirror), 2.6M grocery transaction lines from
2,500 households.

This is the source the demo set includes on purpose because it **cannot** be mapped. The
table has no calendar date anywhere: ``DAY`` is an integer 1-711 counting from an origin
the file never states, and ``WEEK_NO`` is the same information coarsened. ``date`` is a
required canonical field, so there is no correct mapping to find.

Expected outcome: ``ok=False``. A mapper that behaves should return ``date`` as
``strategy='missing'`` rather than inventing an anchor; the column is then absent,
``validate`` raises naming it, and ``transforms`` comes back empty. The script still
writes every artifact and still exits 0 — the failure is the result, not a crash.

The failure this guards against is the other one: mapping ``DAY`` straight into ``date``
produces a frame that validates cleanly, reports 2.6M rows, and places two years of
grocery shopping in January 1970. Compare ``first_date`` in ``facts`` if you ever see
this run come back ``ok=True``.
"""

from __future__ import annotations

from pathlib import Path

from _ingest_artifacts import prepare

DATASET = "frtgnn/dunnhumby-the-complete-journey"
FILENAME = "transaction_data.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dunnhumby"


if __name__ == "__main__":
    raise SystemExit(prepare(dataset=DATASET, filename=FILENAME, out_dir=OUT_DIR))
