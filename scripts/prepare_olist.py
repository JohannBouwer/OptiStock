"""Download the Olist order-items table and map it with ``optistock.ingest``.

Run once, from the repository root::

    python scripts/prepare_olist.py

Writes the five artifacts described in ``_ingest_artifacts`` into ``data/olist/`` for
notebook 11c.

**Source:** Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC BY-NC-SA 4.0).
Git-ignored: the licence is non-commercial share-alike, so the data stays local.

Only ``olist_order_items_dataset.csv`` is fed to the agents, deliberately. The dataset
is relational and the purchase timestamp lives in a *different* table, which is what
makes this the interesting case: this table's only date-like column is
``shipping_limit_date``, a seller handover deadline, and there is no quantity column at
all — one row is one unit.

Expected outcome: a pass that validates cleanly and is still not what you want. The
mapper should reach for ``shipping_limit_date`` because nothing else is a date, and
should return ``quantity`` as ``strategy='constant'``. Watch agent 3: on the runs behind
``docs/ingest_internals.md`` it named the date mismatch precisely, observed that no
mapping of this table can repair it, and approved anyway — which is the correct verdict
and the sharpest illustration of the warning in ``docs/ingest.md`` that a mapping can be
well-formed and still wrong.
"""

from __future__ import annotations

from pathlib import Path

from _ingest_artifacts import prepare

DATASET = "olistbr/brazilian-ecommerce"
FILENAME = "olist_order_items_dataset.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "olist"


if __name__ == "__main__":
    raise SystemExit(prepare(dataset=DATASET, filename=FILENAME, out_dir=OUT_DIR))
