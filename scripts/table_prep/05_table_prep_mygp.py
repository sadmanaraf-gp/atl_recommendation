from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from dotenv import load_dotenv
import oracledb

load_dotenv()


# =========================
# Parameters
# =========================
@dataclass(frozen=True)
class Params:
    # Typically runs for the current month.
    # If today is Jul 13 2026, pred_month = 202607, dim_date = 13-JUL-26
    date_ref = datetime.now()
    pred_month: str = date_ref.strftime("%Y%m")
    dim_date: str = date_ref.strftime("%d-%b-%y").upper()


P = Params()
PM = P.pred_month

print("=" * 60)
print("MYGP TABLE PREP PARAMETERS")
print("-" * 60)
for k, v in vars(P).items():
    print(f"{k:<25} : {v}")
print("=" * 60)


# =========================
# SQL builders
# =========================
def build_statements(pm: str, latest_dim_date: str) -> List[str]:
    """Return the ordered DROP/CREATE statements for the MYGP pipeline.

    latest_dim_date is the LAST_UPDATE_DATE to filter gpbi_customer_dim on
    (today's date), formatted as e.g. '13-JUL-26'.
    """
    return [

        # -------------------------------------------------
        # Reco base: test-group MSISDNs from the prediction table
        # -------------------------------------------------
        f"DROP TABLE TBL_ATL_PREDICTION_{pm}_RECO_BASE PURGE",
        f"""
        CREATE TABLE TBL_ATL_PREDICTION_{pm}_RECO_BASE AS
        SELECT /*+ PARALLEL(8) */ DISTINCT MSISDN
        FROM TBL_ATL_PREDICTION_{pm}
        WHERE GROUP_TYPE LIKE 'TG%'
        """,

        # -------------------------------------------------
        # Non-reco base: latest customer_dim MSISDNs not in reco base
        # -------------------------------------------------
        f"DROP TABLE TBL_ATL_PREDICTION_{pm}_NON_RECO_BASE PURGE",
        f"""
        CREATE TABLE TBL_ATL_PREDICTION_{pm}_NON_RECO_BASE AS
        SELECT /*+ PARALLEL(8) */ DISTINCT MSISDN
        FROM (
            SELECT DISTINCT MSISDN
            FROM GPBI_MARKETING.gpbi_customer_dim
            WHERE LAST_UPDATE_DATE = '{latest_dim_date}'
        ) A
        WHERE MSISDN NOT IN (
            SELECT MSISDN FROM TBL_ATL_PREDICTION_{pm}_RECO_BASE
        )
        """,

        # -------------------------------------------------
        # Non-reco: cross join non-reco base with fixed denominations
        # -------------------------------------------------
        f"DROP TABLE TBL_ATL_PREDICTION_{pm}_NON_RECO PURGE",
        f"""
        CREATE TABLE TBL_ATL_PREDICTION_{pm}_NON_RECO AS
        WITH ServiceAmounts AS (
            -- DATA
            SELECT 118 AS DENO, 'DATA' AS SRVC, 1 AS "RANK" FROM dual UNION ALL
            SELECT 179, 'DATA', 2 FROM dual UNION ALL
            SELECT 98,  'DATA', 3 FROM dual UNION ALL
            SELECT 198, 'DATA', 4 FROM dual UNION ALL
            SELECT 219, 'DATA', 5 FROM dual UNION ALL

            -- MIXED
            SELECT 217, 'MIXED', 1 FROM dual UNION ALL
            SELECT 258, 'MIXED', 2 FROM dual UNION ALL
            SELECT 398, 'MIXED', 3 FROM dual UNION ALL
            SELECT 299, 'MIXED', 4 FROM dual UNION ALL

            -- VOICE
            SELECT 49,  'VOICE', 1 FROM dual UNION ALL
            SELECT 99,  'VOICE', 2 FROM dual UNION ALL
            SELECT 139, 'VOICE', 3 FROM dual UNION ALL
            SELECT 119, 'VOICE', 4 FROM dual UNION ALL
            SELECT 128, 'VOICE', 5 FROM dual
        )
        SELECT /*+ PARALLEL(8) */
            t.MSISDN,
            s.DENO,
            s.SRVC,
            s."RANK"
        FROM
            (SELECT DISTINCT MSISDN FROM TBL_ATL_PREDICTION_{pm}_NON_RECO_BASE) t
        CROSS JOIN
            ServiceAmounts s
        """,

        # -------------------------------------------------
        # Final MYGP table: reco (from prediction) UNION ALL non-reco
        # -------------------------------------------------
        f"DROP TABLE TBL_ATL_PREDICTION_{pm}_MYGP PURGE",
        f"""
        CREATE TABLE TBL_ATL_PREDICTION_{pm}_MYGP COMPRESS FOR ARCHIVE AS
        SELECT /*+ PARALLEL(16) */ *
        FROM (
            (SELECT TO_NUMBER(MSISDN) AS MSISDN, TO_NUMBER(DENO) AS DENO,
                    TO_NUMBER(RANK) AS "RANK", TO_CHAR(SRVC) AS SRVC,
                    'RECO' AS FLAG
             FROM TBL_ATL_PREDICTION_{pm}
             WHERE GROUP_TYPE NOT IN ('CG')
               AND UPPER(SRVC) IN ('VOICE', 'DATA', 'MIXED'))
            UNION ALL
            (SELECT TO_NUMBER(MSISDN) AS MSISDN, TO_NUMBER(DENO) AS DENO,
                    TO_NUMBER(RANK) AS "RANK", TO_CHAR(SRVC) AS SRVC,
                    'NON_RECO' AS FLAG
             FROM TBL_ATL_PREDICTION_{pm}_NON_RECO)
        )
        """,
    ]


# =========================
# Utilities
# =========================
def sanity_check_rows(cursor, table_name: str, min_rows: int = 1):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    (cnt,) = cursor.fetchone()
    print(f"      ↳ Row count for {table_name}: {cnt:,}")
    if cnt < min_rows:
        print(f"      ⚠ WARNING: {table_name} has {cnt} rows")


# =========================
# Sanity tests
# =========================
def run_sanity_tests(cursor, pm: str) -> None:
    print("\n" + "=" * 60)
    print("SANITY TESTS")
    print("=" * 60)

    non_reco = f"TBL_ATL_PREDICTION_{pm}_NON_RECO"
    reco_base = f"TBL_ATL_PREDICTION_{pm}_RECO_BASE"
    mygp = f"TBL_ATL_PREDICTION_{pm}_MYGP"

    # 1. Non-reco denomination spread per service (row vs distinct MSISDN)
    print(f"\n[1] {non_reco} — count per SRVC:")
    cursor.execute(
        f"SELECT SRVC, COUNT(MSISDN), COUNT(DISTINCT MSISDN) "
        f"FROM {non_reco} GROUP BY SRVC ORDER BY 1"
    )
    for srvc, total, distinct in cursor.fetchall():
        print(f"      {srvc:<8} rows={total:>12,}  distinct_msisdn={distinct:>12,}")

    # 2. Final MYGP breakdown per FLAG
    print(f"\n[2] {mygp} — count per FLAG:")
    cursor.execute(
        f"SELECT FLAG, COUNT(*), COUNT(DISTINCT MSISDN) "
        f"FROM {mygp} GROUP BY FLAG ORDER BY 1"
    )
    for flag, total, distinct in cursor.fetchall():
        print(f"      {flag:<10} rows={total:>12,}  distinct_msisdn={distinct:>12,}")

    # 3. No NULL MSISDN / DENO in the final table
    print(f"\n[3] {mygp} — NULL checks:")
    cursor.execute(
        f"SELECT COUNT(*) FROM {mygp} WHERE MSISDN IS NULL OR DENO IS NULL"
    )
    (nulls,) = cursor.fetchone()
    status = "OK" if nulls == 0 else "⚠ FAIL"
    print(f"      NULL MSISDN/DENO rows: {nulls:,}  [{status}]")

    # 4. Reco base and non-reco base must not overlap
    print(f"\n[4] RECO / NON_RECO base overlap:")
    cursor.execute(
        f"SELECT COUNT(*) FROM {reco_base} "
        f"WHERE MSISDN IN (SELECT MSISDN FROM TBL_ATL_PREDICTION_{pm}_NON_RECO_BASE)"
    )
    (overlap,) = cursor.fetchone()
    status = "OK" if overlap == 0 else "⚠ FAIL"
    print(f"      Overlapping MSISDNs: {overlap:,}  [{status}]")

    print("\n" + "=" * 60)


# =========================
# Main Runner
# =========================
def main() -> int:
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")

    if not all([user, password, dsn]):
        print("Missing ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN", file=sys.stderr)
        return 2

    if not re.fullmatch(r"\d{6}", PM):
        print(f"Invalid pred_month={PM}", file=sys.stderr)
        return 3

    with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
        with conn.cursor() as cur:
            print(f"Using gpbi_customer_dim date: {P.dim_date}")

            statements = build_statements(PM, P.dim_date)

            for i, stmt in enumerate(statements, start=1):
                stmt = stmt.strip()
                if not stmt:
                    continue

                first_line = stmt.splitlines()[0]
                create_match = re.match(r"CREATE\s+TABLE\s+(\w+)", stmt, re.I)
                table_created = create_match.group(1) if create_match else None

                print(f"[{i}/{len(statements)}] Executing: {first_line[:120]}")

                try:
                    cur.execute(stmt)
                    conn.commit()

                    if table_created:
                        sanity_check_rows(cur, table_created)

                except oracledb.Error as e:
                    if "ORA-00942" in str(e) and stmt.upper().startswith("DROP TABLE"):
                        print("      (ignored) ORA-00942")
                        conn.commit()
                        continue
                    raise

            run_sanity_tests(cur, PM)

    print("MYGP table creation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
