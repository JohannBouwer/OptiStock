"""Download the Online Retail II order book and map it with ``optistock.ingest``.

Run once, from the repository root::

    python scripts/prepare_online_retail.py

Writes the five artifacts described in ``_ingest_artifacts`` into ``data/online_retail/``
for notebook 11c.

**Source:** Online Retail II (UCI, CC BY 4.0), via the Kaggle mirror. Two years of
transactions from a UK online gift retailer, 1.07M lines. Unlike the bakery export this
one carries a real buyer identifier, which is what makes it the only source in the demo
set where the full canonical schema is reachable.

Expected outcome: a clean pass, and the widest one. ``customer_id`` maps, so
``to_clv`` and ``to_sales(split=True)`` both become runnable. The cancellation lines —
invoices prefixed ``'C'``, carrying negative quantities — are held back as returns rather
than silently dropped. The directory is git-ignored despite the permissive licence: a
1.07M-row csv does not belong in a repository.
"""

from __future__ import annotations

from pathlib import Path

from _ingest_artifacts import prepare

DATASET = "mashlyn/online-retail-ii-uci"
FILENAME = "online_retail_II.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "online_retail"


if __name__ == "__main__":
    raise SystemExit(prepare(dataset=DATASET, filename=FILENAME, out_dir=OUT_DIR))
