# Agent-assisted ingest

[`optistock.preprocessing`](preprocessing.md) needs an order-line table using the canonical column names. Getting there from a real export is the tedious part: the columns are named differently, revenue arrives as a line total, dates are `dd/mm/yyyy` strings, prices are `'0,90 €'`. `optistock.ingest` uses three small agents to propose that mapping, then executes it with plain pandas.

```bash
pip install optistock[ingest]
```

`import optistock` never reaches this package, so the core library stays free of an LLM SDK.

## One call

```python
from optistock.ingest import read_table, ingest

out = ingest(read_table("orders_export.xlsx"))

if out.ok:
    orders = out.frame
    print(out.transforms)        # which of to_sales/to_items/to_clv/to_returns will run
else:
    print(out.render())          # every round, every verdict, and why it failed
```

`out.render()` prints the whole run: what the table was taken to be, each mapping, each verdict, the chosen frame and the relational facts behind it. It is the first thing to read when a result looks wrong.

## Three agents, and code between them

```
read_table ─▶ profile_dataframe ─▶ [describe] ─▶ [map] ─▶ ColumnMapping
                                    agent 1     agent 2        │
                                                               ▼
                                                        to_canonical
                                                               │
                                    [review] ◀─────────────────┤
                                    agent 3                    ▼
                                                  optistock.preprocessing
```

| Stage | Who | What it sees |
|---|---|---|
| `profile_dataframe` | code | the whole frame |
| **describe** | agent 1 | the profile — returns 2-3 sentences of context |
| **map** | agent 2 | the profile + that context — returns a `ColumnMapping` |
| `to_canonical` | code | the whole frame + the mapping |
| **review** | agent 3 | the *result*: its profile, its relational facts, the mapping |

The agents never receive the data and never emit it. They see column names, dtypes, null rates, cardinality and at most five truncated sample values. Token cost is therefore independent of row count, no customer record is sent anywhere, and everything downstream of the mapping is deterministic.

Agent 1 is deliberately forbidden from suggesting a mapping — it reports what it sees (an integer that looks like a day counter, prices that look like line totals) and stops. Agent 3 is the one that earns its call: it sees what agent 2 never does, the frame the mapping actually produced. An integer day counter looks perfectly reasonable in a source profile and absurd in a result starting at `1970-01-01 00:00:00.000000001`.

If agent 3 says `revise` and names issues, agent 2 gets **one** retry with those issues and its previous mapping in the prompt. Which round is returned is decided by code — validity first, then row count — so a revision that fixes the flagged field while breaking another cannot win merely by being last.

Up to 6 model calls per run. Pass `describe=False` to drop to 4, or `max_rounds=1` to skip the retry.

## What the model is allowed to say

Each field gets one of four strategies, and each requires its own payload:

| Strategy | Payload | Meaning |
|---|---|---|
| `direct` | `source_column` | copy one column |
| `expression` | `expression` | arithmetic over columns, e.g. `line_total / qty` |
| `constant` | `constant` | the same value for every row |
| `missing` | none | not derivable from this source |

Expressions go through `df.eval`, which permits column references and arithmetic only — no function calls, no arbitrary code. Two conversions are code's job, declared rather than performed:

- **dates** — the model sets `date_format="%d/%m/%Y"` and `pd.to_datetime` applies it, so an ambiguous `05/03/2025` is not silently read as 3 May;
- **text numbers** — the model sets `numeric_format="comma_decimal"` or `"currency"` and a regex parser applies it.

Small models reliably pick the right *source* but often mislabel the *strategy* enum. Where the intent is unambiguous the model validator repairs it silently; where information is genuinely absent it raises, and PydanticAI feeds the error back for a retry.

## Configuration

Any OpenAI-compatible `/v1` endpoint works: a local server (Ollama, LM Studio, vLLM) or a hosted one. Only environment variables change — copy [`.env.example`](../.env.example) to `.env` and edit it.

```bash
INGEST_BASE_URL=openrouter        # or ollama | lmstudio | google | any URL
INGEST_API_KEY=sk-or-v1-...
INGEST_OUTPUT_MODE=native         # native | tool | prompted

INGEST_DESCRIBE_MODEL=qwen/qwen3.7-flash    # agent 1 — cheapest that works
INGEST_MODEL=anthropic/claude-sonnet-5      # agent 2 — the one that matters
INGEST_REVIEW_MODEL=z-ai/glm-5.2            # agent 3 — deliberately not agent 2
```

**One model per agent.** The two optional slots fall back to `INGEST_MODEL` when unset or empty, and all three share the endpoint and key — so by default they pick different models on the *same* provider. Pass a fully built model to `ingest(describe_model=..., review_model=...)` to cross providers.

Spend where it changes the answer. Measured across seven real datasets:

- **Agent 1** emits 120 words of prose that nothing downstream parses. The cheapest model available is fine; a stronger one buys nothing.
- **Agent 2 is where every wrong number comes from.** A mid-tier model handled the four easy sources identically to a frontier one and then shipped a grocery panel with two years of dates collapsed into 1970, reported as valid and CLV-ready. That is the whole budget argument: the cost of a cheap mapper is not measured in tokens.
- **Agent 3 should not be agent 2.** A reviewer from the same lineage inherits the blind spots that produced the mapping. A cheaper model from a different vendor is a better reviewer than a stronger copy of the mapper — in the trial the mid-tier reviewer correctly flagged an integer date counter and drove a quantity-column fix on retry.

`INGEST_OUTPUT_MODE` matters more than it looks. PydanticAI defaults to *tool* mode, which asks the model to call a function — and models without native tool tokens (notably Gemma) cannot do that reliably. `native` uses the provider's JSON-schema constrained decoding instead. Hence `native` is the default.

> [!TIP]
> On OpenRouter, `native` needs a model advertising **`structured_outputs`**, not merely `response_format`. Models offering only the latter need `INGEST_OUTPUT_MODE=prompted`, which is the weakest of the three — the schema is described in the prompt rather than enforced during sampling, so the `ColumnMapping` validators do correspondingly more work.

Cost is dominated by the fixed instructions, not your data: the three prompts are ~200, ~1,200 and ~1,300 tokens, a column profile is a few hundred more, and none of it scales with row count. A full two-round run is roughly 12–15k in and 2k out — about $0.05 on `anthropic/claude-sonnet-5`, under a cent on `z-ai/glm-5.2`. The expensive failure mode is a wrong mapping, not the bill.

```bash
optistock-ingest --check
```

`optistock-ingest path/to/file.csv` runs the whole pipeline; `--profile-only` prints what the agents would see without calling anything; `--rounds 1` skips the retry; `--no-describe` skips agent 1.

> [!WARNING]
> **A mapping can be well-formed and still wrong.** The schema checks shape and dtype, not meaning. Nothing stops a line total being mapped into `unit_price`, and the result validates cleanly. Read `out.result.facts` — orders per customer, date span, lines per order — which is usually the fastest way to spot it.

[Notebook 11c](../notebooks/11c_Ingest_Across_Datasets.ipynb) is the worked example: four real sources side by side, including one that validates cleanly on a date column that turns out to be a shipping deadline, and one the mapper correctly refuses to map at all.

## Returns

`to_canonical` drops the rows the schema rejects, which on most real exports means the returns: negative quantities are near-universal and would otherwise block the demand panel outright. They are not lost:

```python
out.result.returns        # the return lines, as positive counts
out.transforms["to_returns"]
```

They are captured **before** the drop, which is the only moment they exist. Note the resulting demand panel is gross of returns, and the return rate reads as 0% rather than "not recorded".

## Running it without a model

The deterministic half needs no model, no network and no API key — it is importable even without the `[ingest]` extra. Hand-build the mapping and it behaves like a scripted ETL step, which is how the tests exercise it:

```python
from optistock.ingest import ColumnMapping, FieldMapping, to_canonical

mapping = ColumnMapping(mappings=[
    FieldMapping(canonical_field="order_id",   strategy="direct",
                 source_column="Order Ref",    confidence=1.0),
    FieldMapping(canonical_field="date",       strategy="direct",
                 source_column="purchased_on", date_format="%d/%m/%Y", confidence=1.0),
    FieldMapping(canonical_field="unit_price", strategy="expression",
                 expression="line_total / qty", confidence=1.0),
    # ...
])
orders = to_canonical(source, mapping).frame
```

`apply_mapping` keeps only the mapped fields plus whatever is listed in `unmapped_source_columns`, and it resets the index. If all you need is a rename, [`rename_to_canonical`](preprocessing.md#getting-your-columns-to-match) is the better tool — it drops nothing and preserves the index.

`make_canonical` and `scramble` generate a clean frame and a messy "export" of it with known ground truth, for tests and eval cases.

## When to use this

Reach for `optistock.ingest` when a source needs *reshaping* — derived columns, ambiguous date formats, formatted numbers. When the columns are merely named differently, `rename_to_canonical` is deterministic, instant and needs no model at all.

## Going deeper

[**How the ingest pipeline works**](ingest_internals.md) is the code-level walkthrough: the exact prompts, the validators that run before you see a mapping, the crash guards, and where each design decision lives. Read it before changing anything in `optistock/ingest`.
