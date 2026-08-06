"""The three agents, their output types, and their prompts.

    describe  a short, neutral account of what the source table appears to be
    map       how each canonical field is derived from the source columns
    review    whether the frame that mapping produced holds what its names claim

None of them calls a tool. Each is one request with a JSON schema enforced during
sampling (see :mod:`~optistock.ingest.config`), and each returns a pydantic model that
:mod:`~optistock.ingest.run` and :mod:`~optistock.ingest.apply` execute deterministically.

The output types live here rather than in a separate module because the validators on
them are part of the prompt engineering: a model that mislabels a strategy or invents a
field name gets the ``ValueError`` fed back and retries. On a small cheap model those
repairs matter more, not less.

**This module imports at module scope with pure pydantic only.** ``pydantic-ai`` is
imported inside the three ``build_*`` functions, so ``apply.py`` — which needs the
types but never runs an agent — stays importable without the ``[ingest]`` extra. Adding
a top-level ``from pydantic_ai import ...`` here would silently break that.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..preprocessing.canonical import FIELD_NAMES, REQUIRED_COLUMNS, describe_fields
from .profile import TableProfile, render_facts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic_ai import Agent

__all__ = [
    "ColumnMapping",
    "FieldMapping",
    "NumericFormat",
    "Review",
    "SourceDescription",
    "Strategy",
    "build_describer",
    "build_mapper",
    "build_reviewer",
    "describe_source",
    "map_columns",
    "render_mapping",
    "review_result",
]

Strategy = Literal["direct", "expression", "constant", "missing"]
NumericFormat = Literal["currency", "comma_decimal"]

# An "expression" with none of these is just a column name.
_IS_ARITHMETIC = re.compile(r"[+\-*/()]")

#: Longest source/expression/constant shown by :func:`render_mapping`. A model that
#: overflows its reasoning into ``source_column`` puts it here too — one run produced a
#: 1,410-character value that swallowed the whole table, in the render *and* in the
#: feedback prompt sent back to the mapper.
_MAX_DETAIL = 60

#: Appended to every instruction block, and load-bearing on some providers.
#:
#: When an endpoint serves a model in ``response_format: json_object`` mode rather than
#: true JSON-schema mode, the DashScope-family API *requires* the literal word "json"
#: to appear somewhere in the messages and returns
#: ``400 'messages' must contain the word 'json' in some form`` otherwise. Observed on
#: ``qwen/qwen3.7-flash`` via OpenRouter, where it blocked all seven trial datasets at
#: the first call. Harmless everywhere else — the sentence is true regardless.
_JSON_REMINDER = "\nReturn your answer as JSON conforming to the given schema.\n"


# ---------------------------------------------------------------------------
# Agent 1 — describe
# ---------------------------------------------------------------------------


class SourceDescription(BaseModel):
    """A neutral reading of the source table, as context for the mapper."""

    summary: str = Field(
        description="2-3 sentences: what kind of table this appears to be."
    )
    grain: str = Field(description="What one row represents, in one sentence.")
    caveats: list[str] = Field(
        default_factory=list,
        description="At most 3 short notes on anything odd about the columns.",
    )

    def render(self) -> str:
        lines = [self.summary, f"Grain: {self.grain}"]
        lines += [f"Caveat: {c}" for c in self.caveats[:3]]
        return "\n".join(lines)


DESCRIBE_INSTRUCTIONS = """\
You are shown a profile of an unfamiliar table: column names, dtypes, null rates,
cardinalities and a few sample values. Describe what it appears to be, so another agent
can map it onto a fixed schema with that context in hand.

Be brief and neutral. At most 120 words in total.

Say what the table is (an order history, a point-of-sale export, a web event log, ...),
what one row represents, and note anything odd about the columns: an integer that looks
like a day counter rather than a date, prices that appear to be line totals rather than
per-unit, a suffix or separator that will need declaring, an apparently missing header
row.

Do NOT propose a mapping, name target fields, or say which column should become what.
Describe only what you see. Another agent decides the mapping, and a confident guess
here would anchor it.
""" + _JSON_REMINDER


def build_describer(model=None, output_mode=None) -> "Agent[None, SourceDescription]":
    """Construct the describing agent.

    Defaults to ``INGEST_DESCRIBE_MODEL``, falling back to the mapper's model. This is
    the cheapest agent to run cheaply: its entire output is 120 words of prose that
    nothing downstream parses, so a small model costs little and risks little.
    """
    from pydantic_ai import Agent

    from .config import build_describe_model, wrap_output

    return Agent(
        model or build_describe_model(),
        output_type=wrap_output(SourceDescription, output_mode),
        instructions=DESCRIBE_INSTRUCTIONS,
        name="source_describer",
        retries=2,
    )


def describe_source(
    profile: TableProfile, model=None, output_mode=None
) -> SourceDescription:
    """Agent 1: read the profile, return a short account of the table."""
    return build_describer(model, output_mode).run_sync(profile.render()).output


# ---------------------------------------------------------------------------
# Agent 2 — map
# ---------------------------------------------------------------------------


class FieldMapping(BaseModel):
    """How one canonical field is produced from the source table."""

    canonical_field: str = Field(description="One of the canonical field names.")
    strategy: Strategy = Field(
        description=(
            "direct = copy a single source column; "
            "expression = arithmetic over source columns (pandas df.eval syntax, "
            "e.g. 'line_total / quantity'); "
            "constant = a fixed value for every row; "
            "missing = this field cannot be produced from the source."
        )
    )
    source_column: str | None = Field(
        default=None, description="Exact source column name (strategy='direct')."
    )
    expression: str | None = Field(
        default=None,
        description="pandas df.eval expression over source columns "
        "(strategy='expression').",
    )
    constant: str | None = Field(
        default=None, description="Fixed value as a string (strategy='constant')."
    )
    date_format: str | None = Field(
        default=None,
        description=(
            "For date fields only: the strptime format of the SOURCE text, e.g. "
            "'%d/%m/%Y' for 17/08/2025. Leave null if the source is already a date "
            "or the format is unambiguous. Never parse dates in an expression."
        ),
    )
    numeric_format: NumericFormat | None = Field(
        default=None,
        description=(
            "For numeric fields whose source is TEXT rather than a number. "
            "'currency' strips symbols/thousands separators (e.g. '$1,234.50', "
            "'12 345'). 'comma_decimal' additionally treats ',' as the decimal "
            "point, for European formats like '0,90 EUR' or '1.234,50'. Leave null "
            "if the column is already numeric."
        ),
    )
    confidence: float = Field(description="Confidence in this mapping, from 0 to 1.")
    note: str = Field(default="", description="Short reasoning or caveat.")

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        """Keep the range without putting ``minimum``/``maximum`` in the JSON schema.

        ``Field(ge=0, le=1)`` is the obvious way to say this, and it makes the schema
        unusable for constrained decoding on Anthropic, which rejects the whole
        request: ``400 output_config.format.schema: For 'number' type, properties
        maximum, minimum are not supported``. Every dataset failed on the first call.
        The same applies to ``pattern`` and ``minLength`` anywhere in these models.

        Clamped rather than rejected, because confidence gates nothing — spending a
        retry to re-roll an advisory number would cost a call to change a value that
        nothing reads.
        """
        return min(1.0, max(0.0, value))

    @field_validator("canonical_field")
    @classmethod
    def _must_be_a_canonical_field(cls, value: str) -> str:
        """Reject invented field names.

        Without this the target is a free-form string, so a model returning
        ``invoice_date`` produces a frame with an ``invoice_date`` column, no ``date``
        column, and no warning — the failure only surfaces as a ``KeyError`` further
        downstream. Raising here means the error is fed back and the model corrects
        itself. Case and separator slips are repaired, since those are unambiguous.
        """
        if value in FIELD_NAMES:
            return value
        normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
        if normalised in FIELD_NAMES:
            return normalised
        raise ValueError(
            f"{value!r} is not a canonical field. Expected one of "
            f"{list(FIELD_NAMES)}. Put source columns you want to keep in "
            "'unmapped_source_columns' instead of inventing a field name."
        )

    @model_validator(mode="after")
    def _reconcile_strategy(self) -> FieldMapping:
        """Make the strategy label agree with the payload the model actually filled in.

        Small models reliably pick the right *source* but often mislabel the *strategy*
        enum. Where the intent is unambiguous (exactly one payload field populated) we
        repair it silently; where information is genuinely absent we raise, so the
        error is fed back and the model can retry.
        """
        has_col = bool(self.source_column)
        has_expr = bool(self.expression)
        has_const = self.constant is not None

        # A bare column name in the expression slot is a direct copy, however the model
        # labelled it. Repairing it matters beyond tidiness: `df.eval` parses its
        # argument as Python, so an unquoted name like 'Customer ID' raises
        # SyntaxError, while `df['Customer ID']` is fine. Observed in the wild.
        if has_expr and not has_col and not _IS_ARITHMETIC.search(self.expression):
            self.source_column = self.expression.strip()
            self.expression = None
            self.strategy = "direct"
            return self

        if self.strategy == "expression" and not has_expr:
            if has_col:
                self.strategy = "direct"
            elif has_const:
                self.strategy = "constant"
            else:
                raise ValueError(
                    f"{self.canonical_field}: strategy='expression' but no expression "
                    "was given. Provide 'expression', or use strategy='direct' with "
                    "'source_column'."
                )
        elif self.strategy == "direct" and not has_col:
            if has_expr:
                self.strategy = "expression"
            else:
                raise ValueError(
                    f"{self.canonical_field}: strategy='direct' but no source_column "
                    "was given. Provide 'source_column', or use strategy='missing'."
                )
        elif self.strategy == "constant" and not has_const:
            if has_col:
                self.strategy = "direct"
            else:
                raise ValueError(
                    f"{self.canonical_field}: strategy='constant' but no constant "
                    "value was given. Provide 'constant', or use strategy='missing'."
                )
        return self


class ColumnMapping(BaseModel):
    """The full proposed mapping — Agent 2's entire output.

    Constructible by hand for a deterministic, LLM-free run; that is how the tests
    exercise :func:`~optistock.ingest.apply.apply_mapping`.
    """

    mappings: list[FieldMapping]
    unmapped_source_columns: list[str] = Field(
        default_factory=list,
        description="Source columns not used for any canonical field (kept as "
        "covariates).",
    )
    overall_note: str = ""

    @model_validator(mode="after")
    def _must_cover_every_required_field(self) -> ColumnMapping:
        """Reject a mapping that omits a required field, or names one twice.

        Omission is a distinct failure from a mislabelled name: nothing validates a
        field that is simply absent, so ``apply_mapping`` never creates the column and
        the first sign of trouble is a ``KeyError`` in the caller's own code. A model
        may legitimately say a field is *unavailable* — that is ``strategy='missing'``,
        an explicit entry — but it may not stay silent. A duplicate is just as bad:
        the last write wins, invisibly.
        """
        seen = [m.canonical_field for m in self.mappings]

        duplicates = sorted({f for f in seen if seen.count(f) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate mapping entries for {duplicates}. Give exactly one entry "
                "per canonical field."
            )

        absent = [f for f in REQUIRED_COLUMNS if f not in seen]
        if absent:
            raise ValueError(
                f"no mapping entry for required field(s): {absent}. Every canonical "
                f"field needs an entry — use strategy='missing' to say a field cannot "
                f"be derived from this source, but do not omit it. Expected one entry "
                f"for each of {list(FIELD_NAMES)}."
            )
        return self

    def by_field(self) -> dict[str, FieldMapping]:
        return {m.canonical_field: m for m in self.mappings}


def render_mapping(mapping: ColumnMapping) -> str:
    """One line per field — compact enough to sit in a prompt without dominating it.

    The detail is truncated; see :data:`_MAX_DETAIL`. Which column is actually used is
    decided by :func:`~optistock.ingest.apply._recover_source_column`, which warns when
    it has to recover one.
    """
    lines = []
    for m in mapping.mappings:
        detail = m.source_column or m.expression or m.constant or "-"
        if len(detail) > _MAX_DETAIL:
            detail = f"{detail[:_MAX_DETAIL]}… (+{len(detail) - _MAX_DETAIL} chars)"
        extra = f" date_format={m.date_format!r}" if m.date_format else ""
        extra += f" numeric_format={m.numeric_format}" if m.numeric_format else ""
        lines.append(
            f"  {m.canonical_field} <- [{m.strategy}] {detail}"
            f" (confidence {m.confidence:.2f}){extra}"
        )
    return "\n".join(lines)


MAP_INSTRUCTIONS = f"""\
You map columns from a messy source order-history table onto a fixed canonical schema.

You do NOT transform data. You only describe, per canonical field, how it is derived
from the source columns. Deterministic code applies your mapping afterwards.

The canonical schema (target):
{describe_fields()}

Each strategy REQUIRES its own payload field. Pick the strategy that matches the
payload you are filling in:

  strategy='direct'      -> set source_column ONLY   (a plain column copy)
  strategy='expression'  -> set expression ONLY      (arithmetic over columns)
  strategy='constant'    -> set constant ONLY        (same value for every row)
  strategy='missing'     -> set none of them         (not derivable from this source)

Do NOT use strategy='expression' with a bare column name. Copying one column is
strategy='direct'.

Expressions support ARITHMETIC ONLY (+ - * /) over column names. Function calls are
NOT allowed — no strftime, no to_datetime, no round, no if/else.

DATES: always map a date field with strategy='direct' and set 'date_format' to the
strptime pattern of the source text. Never convert a date inside an expression.
  e.g. source '17/08/2025' -> direct, source_column='purchased_on', date_format='%d/%m/%Y'
  e.g. source '2018-12-01 11:40:29 UTC' -> date_format='%Y-%m-%d %H:%M:%S UTC'

TEXT NUMBERS: check the sample values of every column you map to quantity, unit_price
or unit_cost. If ANY sample contains a character other than a digit, a minus sign or a
single '.' decimal point — a currency symbol, a space, a thousands separator, or a ','
used as the decimal point — you MUST set 'numeric_format'. A dtype of 'object' on a
price column is the giveaway. Never try to clean it in an expression.
  'comma_decimal' -> ',' is the decimal point (European): '0,90 €', '1.234,50'
  'currency'      -> '.' is the decimal point (anglo):    '$1,234.50', '12 345'
Omitting this does not fail loudly — the column silently becomes empty — so when the
samples are not plain numbers, set it.

Worked example. Source columns: 'Order Ref', 'cust', 'sku', 'qty', 'line_total'.
  order_id    -> direct,     source_column='Order Ref'
  quantity    -> direct,     source_column='qty'
  unit_price  -> expression, expression='line_total / qty'
Note unit_price is PER UNIT: when the source only has a line/row total, you must
DIVIDE by the quantity column. Never map a line total straight into unit_price.

Rules:
- Produce exactly one mapping entry for every canonical field listed above.
- Use the field's meaning and the sample values to decide the source, not just the name.
- Required fields should almost always be mapped; only use strategy='missing' if the
  source genuinely lacks the information. Saying 'missing' is better than mapping a
  column that does not mean what the field means.
- ALWAYS prefer a real source column over a constant. Scan the column list for a
  plausible match before deciding a field is absent.
- A non-zero null_rate is NOT a reason to leave a field unmapped. Real exports have
  gaps; map the column anyway and let the blank rows be cleaned downstream.
- CUSTOMER_ID: map it whenever the source has a real buyer identifier — a customer
  number, account id or loyalty id — even when it is only partly populated. Use
  strategy='missing' ONLY if there is no buyer column at all. Never substitute a
  ticket/basket/invoice number, which would make every basket look like a different
  customer.
- Set confidence honestly; low confidence flags a field for human review.
- List every source column you did not use in 'unmapped_source_columns'.
""" + _JSON_REMINDER


def build_mapper(model=None, output_mode=None) -> "Agent[None, ColumnMapping]":
    from pydantic_ai import Agent

    from .config import build_model, wrap_output

    return Agent(
        model or build_model(),
        output_type=wrap_output(ColumnMapping, output_mode),
        instructions=MAP_INSTRUCTIONS,
        name="column_mapper",
        retries=2,
    )


def _map_prompt(
    profile: TableProfile,
    description: SourceDescription | None,
    previous: ColumnMapping | None,
    issues: list[str] | None,
) -> str:
    parts = []
    if description is not None:
        parts += ["ANOTHER AGENT'S READING OF THIS TABLE", description.render(), ""]
    parts += [
        "Here is the profile of the source table to map.",
        "",
        profile.render(),
        "",
        f"Map it onto the canonical fields: {', '.join(FIELD_NAMES)}.",
    ]
    if previous is not None and issues:
        parts += [
            "",
            "YOUR PREVIOUS MAPPING WAS REVIEWED AND NEEDS CORRECTION",
            render_mapping(previous),
            "",
            "PROBLEMS FOUND",
            *(f"  - {issue}" for issue in issues),
            "",
            "Return a corrected mapping. Change ONLY the fields named above; keep "
            "every other field exactly as it was.",
            "",
            # Named explicitly because a closing paragraph listing only the four
            # strategies let a mapper re-send an identical mapping when told its price
            # column was unparsed, shipping a column of NaN.
            "If the problem is that a column did not parse, the fix is NOT a different "
            "source column. Keep the same source_column and add the format that "
            "describes its text: numeric_format='comma_decimal' when ',' is the "
            "decimal point, numeric_format='currency' for symbols and thousands "
            "separators, or date_format set to the strptime pattern of the source.",
            "",
            "Only if a listed field cannot be produced from this source with direct / "
            "expression / constant and no format setting, set it to "
            "strategy='missing' rather than substituting something nearly right.",
        ]
    return "\n".join(parts)


def map_columns(
    profile: TableProfile,
    model=None,
    output_mode=None,
    *,
    description: SourceDescription | None = None,
    previous: ColumnMapping | None = None,
    issues: list[str] | None = None,
) -> ColumnMapping:
    """Agent 2: propose how every canonical field is derived from the source.

    Pass *previous* and *issues* together to ask for a correction. Without *previous*
    the mapper re-derives from scratch and regresses fields nobody complained about.
    """
    agent = build_mapper(model, output_mode)
    return agent.run_sync(_map_prompt(profile, description, previous, issues)).output


# ---------------------------------------------------------------------------
# Agent 3 — review
# ---------------------------------------------------------------------------


class Review(BaseModel):
    """Agent 3's verdict on one mapping, judged by the frame it produced."""

    verdict: Literal["approve", "revise"] = Field(
        description="approve = every field holds what its name claims; "
        "revise = at least one field does not, and the mapper could fix it."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="One sentence each, most serious first. Name the canonical field, "
        "what is wrong, and what the mapping should say instead.",
    )
    summary: str = ""


REVIEW_INSTRUCTIONS = f"""\
You audit the RESULT of a column mapping onto a canonical order-line schema. You are
shown the source table's profile, the mapping that was proposed, the profile of the
frame it produced, and factual measurements of that frame. Judge whether each canonical
column now holds what its name claims.

The canonical schema:
{describe_fields()}

Look for results that are structurally valid but semantically wrong:
- a date whose values are implausible for real orders (e.g. everything in 1970, which
  is what an integer column looks like after being read as nanoseconds)
- an order_id, customer_id or item_id that is really one of the others, or a category
- a quantity that is not a count of units
- a unit_price that is really a line total (it would rise with quantity)
- a numeric or date column that is entirely null in the result while the source column
  it came from had values — the source text was never parsed

WHAT IS NOT A DEFECT — do not report any of these:
1. The canonical schema is day-granular by design. Ignoring a separate time or hour
   column is correct.
2. order_id, customer_id and item_id deliberately keep the source dtype. A float64 or
   int64 identifier is correct: the orchestrator joins on raw value equality, so
   coercing 101 to "101" would match zero rows. Never report a dtype on an identifier.
3. strategy='missing' is a valid answer. A source with no buyer id is ordinary;
   customer_id='missing' blocks CLV and nothing else. Never demand a substitute, and
   never accept a ticket/basket/invoice number as a customer id.
4. Low confidence is not itself a defect. Judge the mapping against the source profile,
   not the mapper's self-assessment. A field can be a sound proxy at confidence 0.0
   when it is the only candidate in the table.
5. Unmapped source columns are fine — the schema is deliberately narrow. Mention one
   only if it is a BETTER candidate for a canonical field than the column chosen, and
   then say which field and why.
6. A non-zero null rate is not a defect.

Report ONLY problems that change a number downstream. For each issue, name the
canonical field, say what is wrong, and say what the mapping should be instead. If you
cannot name a downstream consequence in to_sales, to_items or to_clv, it is not an
issue — leave it out.

The mapper's vocabulary is strategy='direct' (one column), 'expression' (ARITHMETIC
ONLY over columns — no function calls, no date maths, no lookups, no joins),
'constant', and 'missing'. Two more settings sit alongside any strategy and are fully
within its reach: date_format (the strptime pattern of the source text, e.g. '%d/%m/%Y'
or '%Y-%m-%d %H:%M:%S UTC') and numeric_format ('comma_decimal' when ',' is the decimal
point, 'currency' for symbols and thousands separators). Text that will not parse into
a number or a date is the most common repairable problem there is — name the format to
set, and do not suggest strategy='missing' for a column that is present and merely
unparsed.

BEFORE you conclude that the source cannot support a correct mapping, go back to the
SOURCE TABLE profile and check every column for a better candidate. Do not write "there
is no other X column in the source" unless you have just read the column list and it is
true. A headerless file makes this easy to get wrong: its column NAMES are the first row
of data, so a column called '1' whose samples are all 1 is a quantity, and a column
called '0' whose samples are large integers is an id. Judge by the samples and the
cardinality, never by the name.

If, having checked, the only fix needs something outside all of the above — a lookup
table, a date offset, a join, a column that is not in this table — then say so in the
summary and return verdict='approve' with the issue listed, rather than sending the
mapper after something it cannot express. If a better column DOES exist, that is
verdict='revise', and name the column.
""" + _JSON_REMINDER


def build_reviewer(model=None, output_mode=None) -> "Agent[None, Review]":
    """Construct the reviewing agent.

    Defaults to ``INGEST_REVIEW_MODEL``, falling back to the mapper's model. Reviewing
    a model with itself is the weakest form of review — it shares the blind spots that
    produced the mapping — so point it somewhere else when you can.
    """
    from pydantic_ai import Agent

    from .config import build_review_model, wrap_output

    return Agent(
        model or build_review_model(),
        output_type=wrap_output(Review, output_mode),
        instructions=REVIEW_INSTRUCTIONS,
        name="result_reviewer",
        retries=2,
    )


def _review_prompt(source, mapping, result, result_profile) -> str:
    parts = [
        "SOURCE TABLE",
        source.render(),
        "",
        "PROPOSED MAPPING",
        render_mapping(mapping),
    ]
    if result_profile is not None:
        parts += ["", "RESULTING CANONICAL FRAME", result_profile.render()]
    if result.facts:
        parts += ["", "FACTS ABOUT THE RESULT", render_facts(result.facts)]
    if not result.ok and result.error:
        parts += ["", "SCHEMA VALIDATION FAILED", result.error]
    return "\n".join(parts)


def review_result(
    source: TableProfile,
    mapping: ColumnMapping,
    result,
    *,
    model=None,
    output_mode=None,
) -> Review:
    """Agent 3: judge a mapping by the frame it actually produced.

    The mapper only ever sees the source profile. This agent sees the *result*, which
    is where a semantically wrong mapping becomes visible: an integer day counter looks
    perfectly reasonable in a source profile and utterly absurd in a result that starts
    at ``1970-01-01 00:00:00.000000001``.

    The mapper's own ``note`` is deliberately not shown. It is the mapper's argument
    for its own answer, and a reviewer that reads it grades an essay instead of reading
    the data — one model's note said its mapping was wrong and shipped it anyway.
    """
    agent = build_reviewer(model, output_mode)
    return agent.run_sync(_review_prompt_for(source, mapping, result)).output


def _review_prompt_for(source: TableProfile, mapping: ColumnMapping, result) -> str:
    from .profile import profile_dataframe

    result_profile = None if result.frame is None else profile_dataframe(result.frame)
    return _review_prompt(source, mapping, result, result_profile)
