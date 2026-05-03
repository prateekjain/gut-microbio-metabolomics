"""
Idempotently apply trigram search indexes on `in_vivo`.

These indexes back the server-side ILIKE search in
`compare_tumor.data_functions.search_gmm_name_options` (the in_vivo
metabolite/heatmap dropdowns). They are not created by `migrate_invivo.py`,
so re-run this after a fresh restore or table swap.

Creates (each step is a no-op if already done):
  - extension pg_trgm
  - GIN(metabolite gin_trgm_ops)  → idx_in_vivo_metabolite_trgm
  - GIN(name gin_trgm_ops)        → idx_in_vivo_name_trgm

CREATE INDEX CONCURRENTLY is used so the table is never exclusively locked
(the DB is shared with the sister Yale app). Concurrent builds need
autocommit and cannot run inside a transaction.

Usage:
    python apply_invivo_indexes.py
"""
from __future__ import annotations

import os
import sys

import psycopg2
from dotenv import load_dotenv


INDEXES = [
    (
        "idx_in_vivo_metabolite_trgm",
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_in_vivo_metabolite_trgm '
        'ON in_vivo USING GIN (metabolite gin_trgm_ops)',
    ),
    (
        "idx_in_vivo_name_trgm",
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_in_vivo_name_trgm '
        'ON in_vivo USING GIN (name gin_trgm_ops)',
    ),
]


def main() -> int:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is not set; aborting.", file=sys.stderr)
        return 1

    conn = psycopg2.connect(db_url)
    conn.autocommit = True  # CREATE INDEX CONCURRENTLY can't run in a transaction
    try:
        with conn.cursor() as cur:
            print("Ensuring pg_trgm extension is installed...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

            for name, ddl in INDEXES:
                print(f"Building {name} (no-op if it already exists)...")
                cur.execute(ddl)

            cur.execute(
                """
                SELECT i.relname, x.indisvalid,
                       pg_size_pretty(pg_relation_size(i.oid)) AS size
                FROM pg_class t
                JOIN pg_index x ON x.indrelid = t.oid
                JOIN pg_class i ON i.oid = x.indexrelid
                WHERE t.relname = 'in_vivo'
                  AND i.relname = ANY(%s)
                ORDER BY i.relname
                """,
                ([n for n, _ in INDEXES],),
            )
            rows = cur.fetchall()

        print("\nFinal index status:")
        for relname, is_valid, size in rows:
            marker = "OK " if is_valid else "INVALID "
            print(f"  {marker}{relname:<32} {size}")

        invalid = [r[0] for r in rows if not r[1]]
        if invalid:
            print(
                f"\nWARNING: invalid indexes present (drop and re-run): {invalid}",
                file=sys.stderr,
            )
            return 2
        if len(rows) != len(INDEXES):
            present = {r[0] for r in rows}
            missing = [n for n, _ in INDEXES if n not in present]
            print(
                f"\nWARNING: expected indexes not visible: {missing}",
                file=sys.stderr,
            )
            return 2
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
