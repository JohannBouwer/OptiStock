# How the ingest pipeline works

A code-level walkthrough of the v1: six modules, three agents, and the deterministic
code between them. [`ingest.md`](ingest.md) is the user-facing guide; this is the one to
read before changing anything.

The design in one sentence: **the agents propose, code disposes, and no model output is
used without a deterministic check behind it.**

---

## 1. The six modules

```
  raw file
     │  read_table
     ▼
  DataFrame ──▶ profile_dataframe ──▶ TableProfile
     │                                     │
     │                    ┌────────────────┴────────────────┐
     │                    ▼                                 │
     │            (1) describe_source ──▶ SourceDescription  │
     │                                     │                 │
     │                                     ▼                 ▼
     │                            (2) map_columns ──▶ ColumnMapping
     │                                                       │
     └───────────────────▶ to_canonical ◀────────────────────┘
                                │   apply_mapping + validate    (no LLM)
                                ▼
                        ValidationResult ──▶ relational_facts
                                │
                                ▼
                        (3) review_result ──▶ Review
                                │  verdict == "revise" ─▶ one retry of (2)
                                ▼
                        available_transforms ─▶ to_sales / to_items / to_clv / to_returns
```

| Module | LLM? | Holds |
|---|---|---|
| `preprocessing/canonical.py` | no | `FIELDS`, the pandera gate, `validate`, `drop_invalid_rows`, `rename_to_canonical` |
| `preprocessing/transforms.py` | no | `to_sales`, `to_items`, `to_clv`, `to_returns`, `available_transforms` |
| `ingest/profile.py` | no | `read_table`, `profile_dataframe`, `relational_facts` |
| `ingest/agents.py` | **yes**, lazily | the three output models, the three prompts, the three builders |
| `ingest/apply.py` | no | `apply_mapping`, `to_canonical`, `ValidationResult` |
| `ingest/run.py` | orchestrates | `ingest()`, the retry, which round wins |
| `ingest/config.py` | — | model/endpoint/output-mode wiring |

### The import boundary is load-bearing

`apply.py` needs `ColumnMapping`, which lives in `agents.py` — but `apply.py` must be
importable without `pydantic-ai`. So **`agents.py` imports only pydantic at module
scope**, and `from pydantic_ai import Agent` happens *inside* each `build_*` function.
Adding a top-level `pydantic_ai` import to `agents.py` silently breaks the LLM-free
half of the package. `ingest/__init__.py` layers a PEP-562 `__getattr__` on top, so
`ingest`/`describe_source`/`map_columns`/`review_result` are only resolved on first
access.

---

## 2. What the agents see — and only this

`profile_dataframe` (`profile.py`) reduces a frame to per-column metadata: name, dtype,
non-null count, null rate, cardinality, and up to **5 distinct non-null values truncated
to 40 characters**. `TableProfile.render()` is exactly what goes in the prompt:

```
Source table: 112650 rows, 7 columns.

- 'order_id' | dtype=object | null_rate=0% | unique=98666 | samples: 00010242fe8c5a6d…, …
- 'order_item_id' | dtype=int64 | null_rate=0% | unique=21 | samples: 1, 2, 3, 4, 5
- 'shipping_limit_date' | dtype=object | null_rate=0% | unique=93318 | samples: 2017-09-19 09:45:35, …
- 'price' | dtype=float64 | null_rate=0% | unique=5968 | samples: 58.9, 239.9, 199.0, …
```

Three consequences worth stating plainly:

- **Token cost is independent of row count.** 112,650 rows and 112 rows cost the same.
- **No customer record leaves the machine.** The truncated samples are the only values
  that travel — a real privacy surface, but a small and inspectable one.
- **The agents are diagnosing from a shadow.** Anything invisible in a profile — whether
  `price` scales with quantity, whether `DAY` is a day counter — they will sometimes get
  wrong. Agent 3 exists because of that sentence.

`relational_facts(df)` is the second view, and the one a per-column profile cannot give:

```
n_order_lines=112650, n_orders=98666, lines_per_order=1.14, n_items=32951,
first_date=2016-09-19, last_date=2020-04-09, date_span_days=1298.9, n_distinct_dates=555,
quantity_modal_value=1.0, quantity_is_integral=True, unit_price_median=74.99, ...
```

Pure description, no judgement, so it cannot be wrong — and it is where a swapped or
collapsed column announces itself. A day counter mapped to `date` reads unremarkably
column by column and absurdly as `date_span_days=0.0` across 2.6M lines.

---

## 3. `canonical.py` — one registry, two consumers

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str | None   # None = accept the caller's dtype unchanged
    required: bool
    description: str    # the agents read this
```

`FIELDS` drives both `ORDER_SCHEMA` (which *enforces*) and `describe_fields()` (which
*describes* to all three prompts), so the gate and the prompts cannot drift apart. Add a
field here and everything follows.

Two choices in this file are load-bearing and must survive any future edit:

- **Identity fields carry `dtype=None` and are never coerced.** `BaseStockKeep` joins
  histories to `item_configs` by raw value equality (`stockkeep.py:371`, `:1167`), so
  coercing SKU `101` to `'101'` matches zero rows and fails deep inside the forecaster.
  Validation must not change what an identifier *is*.
- **Numeric columns use `coerce=False` plus a `pa.Parser`.** pandera's `coerce=True`
  uses `astype()`, which raises on `'abc'`, leaves the column as `object`, and then lets
  the `> 0` check run against strings — surfacing a `TypeError` *as if it were a
  quantity violation*.

---

## 4. Agent 1 — describe

There are no tools anywhere in this pipeline. Each agent is one request with a JSON
schema enforced during sampling; there is no tool loop and no multi-turn conversation.

```python
class SourceDescription(BaseModel):
    summary: str        # 2-3 sentences: what this table appears to be
    grain: str          # what one row represents
    caveats: list[str]  # at most 3
```

Input: `TableProfile.render()`. Output: short prose prepended to agent 2's prompt.

The prompt caps it at 120 words and **forbids proposing a mapping**. That restriction is
the whole reason it is safe to run: an agent that names target fields becomes a second
mapper whose guesses agent 2 then anchors on, and you have paid for a call that makes
the result worse. It reports what it sees — an integer that looks like a day counter,
prices that look like line totals, an apparently missing header row — and stops.

---

## 5. Agent 2 — map

`MAP_INSTRUCTIONS` is long because every paragraph was written against an observed
failure: the strategy/payload table, the ban on function calls in expressions, the
`date_format` and `numeric_format` rules, the `line_total / qty` worked example, the
customer_id rule.

| strategy | payload | meaning |
|---|---|---|
| `direct` | `source_column` | copy one column |
| `expression` | `expression` | arithmetic over columns via `df.eval` |
| `constant` | `constant` | one value for every row |
| `missing` | none | not derivable from this source |

Plus two settings that sit alongside *any* strategy: `date_format` (the strptime pattern
of the source text) and `numeric_format` (`comma_decimal` / `currency`). Those two are
the most common repair there is, and both agent 2's and agent 3's prompts name them
explicitly — an earlier version listed only the four strategies, and a mapper told its
price column was unparsed re-sent an identical mapping and shipped a column of NaN.

### The validators are prompt engineering

Constrained decoding guarantees the *shape*. Four pydantic validators guard the
*content*, and each exists because a real model got it wrong:

| Validator | Behaviour | The failure it caught |
|---|---|---|
| `_must_be_a_canonical_field` | repairs case/separator slips, **raises** on invented names | a model returned `invoice_date`; the frame then had no `date` column and no warning |
| `_reconcile_strategy` | repairs an unambiguous strategy/payload mismatch, **raises** otherwise | `strategy='expression', expression='Price'` — and `df.eval('Customer ID')` is a `SyntaxError` |
| `_must_cover_every_required_field` | **raises** on an omitted required field | six entries returned, `date` silently absent |
| — same validator | **raises** on duplicates | two entries for one field: last write wins, invisibly |

Repair only where intent is unambiguous; everything else raises, and PydanticAI's
`retries=2` feeds the message back so the model corrects itself. Those messages are
written *for the model*: they name the expected values and the alternative action.

> [!IMPORTANT]
> **Keep JSON-schema range and pattern keywords out of these models.** Providers
> restrict what constrained decoding accepts and reject the *whole request* rather than
> ignoring a keyword. `confidence: float = Field(ge=0.0, le=1.0)` emits
> `minimum`/`maximum`, and every dataset failed instantly on Anthropic with
> `400 output_config.format.schema: For 'number' type, properties maximum, minimum are
> not supported`. The bound lives in a `field_validator` that clamps instead. Same for
> `pattern`, `minLength`, `maxLength`.

---

## 6. `apply.py` — code disposes

`apply_mapping` walks the mappings and calls `_resolve` per field, then per column:
parse dates with the declared `date_format`; parse text numbers with the declared
`numeric_format`; and `_warn_if_unparseable` if a numeric/date column had data but
parsed to *all* NaN — the optional fields are nullable, so an all-NaN `unit_price`
**validates cleanly** and produces zero revenue downstream with no error at all.

Three crash guards, each from a response a real model returned:

```python
_coerce_constant   # a model returned constant='，1' (fullwidth comma)
```
It raises rather than stripping the stray character. Turning a garbled response into a
plausible `1.0` is worse than a crash, because you would never find out.

```python
_resolve           # catches TypeError from df.eval, not just ValueError
```
A model reached for `expression='TRANSACTION_DT + CUSTOMER_ID'`, trying to *concatenate*
identifiers into a composite key. `object + int64` raises `TypeError`, and the message
says so specifically — the generic "expression not evaluable" sent the model in circles.

```python
_recover_source_column
```
gemini-3-flash returned a `source_column` of 1,410 characters: the real column name
followed by an entire JSON document of reasoning, which constrained decoding happily
accepted as a string. Recovery takes the **longest** prefix that is a real column,
refuses when two candidates tie, and warns loudly when it fires. A recovery, not a
repair.

`to_canonical` then optionally drops the rows the schema would reject and validates:

```python
@dataclass
class ValidationResult:
    ok: bool                       # validate() alone decides this
    frame: pd.DataFrame | None     # present even when ok=False
    error: str | None
    confidence: dict[str, float]
    low_confidence: tuple[str, ...]  # below 0.5, excluding strategy='missing'
    facts: dict[str, object]
```

On confidence, be honest about what it is worth: across real datasets `0.00` has
appeared on the **worst** mapping seen *and* on a correct one. It is data you can
threshold, never a gate. `strategy='missing'` entries are excluded from
`low_confidence` — declining a field the source lacks is a right answer, not a doubt.

---

## 7. Agent 3 — review

```python
class Review(BaseModel):
    verdict: Literal["approve", "revise"]
    issues: list[str]   # one sentence each, most serious first
    summary: str
```

It sees what agent 2 never does — the **result**: the source profile, the proposed
mapping, the profile of the frame the mapping produced, and `relational_facts`. That is
the entire point of a third agent. An integer day counter looks perfectly reasonable in
a source profile and utterly absurd in a result whose first date is
`1970-01-01 00:00:00.000000001`.

It does **not** see row-level data, and it does **not** see the mapper's `note`. The
note is agent 2's argument for its own answer, and a reviewer that reads it grades an
essay instead of reading the data — one model's note said its own mapping was wrong and
shipped it anyway.

`REVIEW_INSTRUCTIONS` carries a six-point **"WHAT IS NOT A DEFECT"** block, which exists
because the first version of this reviewer *rejected a correct bakery mapping*:

1. day granularity is by design; ignoring an hour column is correct
2. identity dtypes are deliberately preserved — never report a dtype on an identifier
3. `strategy='missing'` is a valid answer; never demand a substitute
4. low confidence is not itself a defect
5. unmapped source columns are fine
6. a non-zero null rate is not a defect

Plus the strongest lever: *report only problems that change a number downstream, and say
which of `to_sales`/`to_items`/`to_clv` changes*. Neither "date ignores `time`" nor
"order_id is float64" can produce a non-vacuous answer there.

**`verdict` is the only signal the loop reads**, and v1 deliberately dropped the
separate `fixable_from_this_source` flag the previous design carried. The cost of that
collapse is real: "the source cannot support this" and "the mapping is fine" both come
back as `approve`, so a wrong mapping the reviewer believes is unfixable never gets a
retry. On REES46 jewelry the reviewer diagnosed the wrong quantity column precisely,
asserted *"there is no other quantity-like column in the source"* — false, the column
named `'1'` was all 1s — and approved, ending the run at 46,526 of 95,910 rows. The
prompt now requires it to re-read the column list before making that claim, and warns
that a headerless file's column *names are its first row of data*. That is a mitigation,
not a guarantee.

---

## 8. `run.py` — the loop

```python
out = ingest(raw, max_rounds=2)   # ≤ 6 model calls
```

Profile → describe → map → `to_canonical` → review. If the verdict is `revise` **and**
there are issues to act on **and** a round remains, re-map with the issues and the
previous mapping in the prompt. Failures degrade rather than propagate: a mapper crash
ends the run with the error on the round, an `apply` crash becomes the next round's
feedback, a reviewer crash leaves the mapping standing.

There is no scoring function, no no-progress guard, and no six-way `stopped_because`.
A hard cap of two rounds makes all of them unnecessary — that is the main thing this v1
removed.

**Which round is returned is decided by code:**

```python
def _better(new, old):
    if new.ok != old.ok:
        return new.ok      # validity first
    return _n_rows(new) > _n_rows(old)   # then rows
```

Rows are the tiebreak because the usual way a "correction" goes wrong is mapping
`quantity` to a column full of zeros and losing half the order book to
`drop_invalid_rows`. On REES46 jewelry the reviewer mis-identified the quantity column,
the mapper obliged, and round 2 dropped 51.5% of the rows while validating cleanly.

`IngestResult.transforms` is computed by `available_transforms(frame)` — **not** asked
of a model. A deterministic function that reads the built frame cannot be wrong about
which transforms it can feed.

---

## 9. Config, cost, and where to change things

**Model calls:** profiling 0, one round 3, `max_rounds=2` up to 6. Cost is independent
of row count. Pass `describe=False` to drop to 2 per round.

```bash
INGEST_MODEL=anthropic/claude-sonnet-5
INGEST_BASE_URL=openrouter        # or ollama | lmstudio | google | any /v1 URL
INGEST_API_KEY=sk-or-v1-...
INGEST_OUTPUT_MODE=native         # native | tool | prompted
INGEST_REVIEW_MODEL=              # empty falls back to INGEST_MODEL
```

`native` is the default because Gemma has no tool tokens and cannot do `tool` mode
reliably; `tool` mode also fails outright on Gemini via OpenRouter.

| To change… | Edit |
|---|---|
| the canonical schema | `canonical.py` `FIELDS` — gate and all three prompts follow |
| what the agents see of the source | `profile.py` |
| the strategy vocabulary | `agents.py` `Strategy` + `_reconcile_strategy` + `apply.py` `_resolve` |
| any prompt | `agents.py` `DESCRIBE_/MAP_/REVIEW_INSTRUCTIONS` |
| when the loop stops, or which round wins | `run.py` `ingest` / `_better` |
| a new transform | `transforms.py`, and add it to `available_transforms` |

---

## 10. How it is tested

149 offline tests, no API key:

```bash
uv run pytest -q
```

`pydantic_ai.models.function.FunctionModel` drives all three agents offline, covering
the retry, the round cap, a crash becoming a round, a failing reviewer, and the
row-count tiebreak. The validators, the three crash guards and the schema-portability
rule are each pinned by the actual captured strings from the failures that motivated
them. `pyproject.toml` sets `filterwarnings = ["error"]`, so a spurious warning is a
build failure.

---

## 11. What this does not do

- **There are no deterministic plausibility checks any more.** The previous version had
  seven; v1 keeps only `relational_facts`, which describes but never judges. On a strong
  model this costs nothing — Sonnet-5 declined to map dunnhumby's integer `DAY` at all —
  but on a weaker model the deterministic epoch-collapse check was the only thing that
  caught a frame which validated cleanly and produced a confident, entirely wrong CLV
  table. Reinstating it is ~20 lines in `profile.py`.
- **A mapping can be well-formed and still wrong, and nothing deterministic will say
  so.** Olist's `date ← shipping_limit_date` is a seller deadline, not a purchase date;
  it validates cleanly and every check passes. Agent 3 *did* catch it — "the source
  table contains no purchase-date column, so this mismatch cannot be fixed within this
  table" — and correctly approved anyway, since no mapping can repair it. That is the
  design working, but it rests entirely on the review model being good enough to know
  what an Olist column means.
- **The reviewer shares the mapper's blind spots** when `INGEST_REVIEW_MODEL` is unset.
- **The loop cannot make a model act.** Every layer can detect a problem and none can
  repair it if the mapper declines: gemini-3-flash returned an identical mapping 3/3
  when told, verbatim, to set `numeric_format='comma_decimal'`.
- **Model behaviour is not reproducible.** The same dataset and model can approve in
  round 1 on one run and exhaust its retries on the next.
