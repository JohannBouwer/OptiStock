"""Order-line preprocessing: the canonical schema and the deterministic transforms.

Two modules, one contract:

    canonical.py    the field registry, the pandera gate, validate/drop/rename
    transforms.py   to_sales, to_items, to_clv, to_returns, available_transforms

Everything downstream consumes one of these frames, and all of them derive from the
same order-line table, so the model families can never disagree about what happened.

Pure pandas + pandera. ``pymc-marketing`` is imported lazily inside ``to_clv``, and
nothing here touches an LLM — see :mod:`optistock.ingest` for the agents that produce
the canonical frame upstream.
"""

from .canonical import (
    CUSTOMER_ID,
    CUSTOMER_SCHEMA,
    DATE,
    FIELD_NAMES,
    FIELDS,
    ITEM_ID,
    ORDER_ID,
    ORDER_SCHEMA,
    QUANTITY,
    REQUIRED_COLUMNS,
    UNIT_COST,
    UNIT_PRICE,
    FieldSpec,
    OrderValidationError,
    describe_fields,
    drop_invalid_rows,
    rename_to_canonical,
    validate,
)
from .transforms import (
    OPTISTOCK_CORE_COLS,
    available_transforms,
    split_new_returning,
    to_clv,
    to_items,
    to_returns,
    to_sales,
)

__all__ = [
    # Schema
    "CUSTOMER_ID",
    "CUSTOMER_SCHEMA",
    "DATE",
    "FIELDS",
    "FIELD_NAMES",
    "ITEM_ID",
    "ORDER_ID",
    "ORDER_SCHEMA",
    "QUANTITY",
    "REQUIRED_COLUMNS",
    "UNIT_COST",
    "UNIT_PRICE",
    "FieldSpec",
    "describe_fields",
    # Validation
    "OrderValidationError",
    "drop_invalid_rows",
    "rename_to_canonical",
    "validate",
    # Transforms
    "OPTISTOCK_CORE_COLS",
    "available_transforms",
    "split_new_returning",
    "to_clv",
    "to_items",
    "to_returns",
    "to_sales",
]
