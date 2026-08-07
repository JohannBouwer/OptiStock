"""Command-line entrypoint: profile a file, map it, apply + validate.

Installed as ``optistock-ingest`` by ``pip install optistock[ingest]``.
"""

from __future__ import annotations

import argparse
import sys

from .profile import profile_dataframe, read_table


def check_connection() -> int:
    """Smoke-test the configured endpoint/model/output-mode with a tiny request."""
    import os

    from pydantic_ai import Agent

    from .config import (
        DEFAULT_BASE_URL,
        DEFAULT_MODEL,
        DEFAULT_OUTPUT_MODE,
        KNOWN_ENDPOINTS,
        build_model,
        describe_model_name,
        review_model_name,
    )

    base_url = os.getenv("INGEST_BASE_URL", DEFAULT_BASE_URL)
    print(f"endpoint      : {KNOWN_ENDPOINTS.get(base_url, base_url)}")
    print(f"output mode   : {os.getenv('INGEST_OUTPUT_MODE', DEFAULT_OUTPUT_MODE)}")
    print(f"1 describe    : {describe_model_name()}")
    print(f"2 map         : {os.getenv('INGEST_MODEL', DEFAULT_MODEL)}")
    print(f"3 review      : {review_model_name()}")
    print("\nsending a test request to the mapper...")

    try:
        reply = Agent(build_model(), output_type=str).run_sync("Reply with exactly: ok")
    except Exception as exc:  # noqa: BLE001 - surface whatever the endpoint said
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"response      : {reply.output.strip()[:80]}\n\nConnection OK.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="optistock-ingest",
        description="Map a raw order-history file onto the canonical schema.",
    )
    parser.add_argument("path", nargs="?", help="Path to a csv/tsv/xlsx/parquet file.")
    parser.add_argument(
        "--check", action="store_true",
        help="Verify the model endpoint is reachable, then exit.",
    )
    parser.add_argument(
        "--profile-only", action="store_true",
        help="Just print the source-table profile (no model call).",
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="Mapper attempts (default 2). Use 1 to skip the retry.",
    )
    parser.add_argument(
        "--no-describe", action="store_true",
        help="Skip agent 1, saving one call per round.",
    )
    parser.add_argument("--out", help="Path to write the validated canonical csv.")
    args = parser.parse_args()

    if args.check:
        return check_connection()
    if not args.path:
        parser.error("a file path is required (or use --check)")

    df = read_table(args.path)
    if args.profile_only:
        print(profile_dataframe(df).render())
        return 0

    # Imported here, not at module scope, so --profile-only works without the
    # [ingest] extra installed.
    from .run import ingest

    print(f"Read {len(df):,} rows x {df.shape[1]} columns. Running the agents...\n")
    out = ingest(df, max_rounds=args.rounds, describe=not args.no_describe)
    print(out.render())

    if not out.ok:
        print("\nNo valid canonical frame was produced.", file=sys.stderr)
        return 1
    if args.out:
        out.frame.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
