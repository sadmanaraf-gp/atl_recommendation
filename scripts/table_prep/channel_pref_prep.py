from __future__ import annotations

import os
import re
import sys
import datetime
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
import oracledb
from dateutil.relativedelta import relativedelta

load_dotenv()


# =========================
# Date Parameters
# =========================
@dataclass(frozen=True)
class DateParams:
    # Fixed run reference
    _ref_date = datetime.date.today()

    _fmt_oracle = "%d-%b-%y"

    # Activity window (Oct 1 – Dec 31)
    _start_day = (_ref_date - relativedelta(months=3)).replace(day=1)
    _end_day = (_start_day + relativedelta(months=3)) - relativedelta(days=1)

    # Load window with buffer
    load_dt_start: str = (_start_day - relativedelta(days=5)).strftime(_fmt_oracle).upper()
    load_dt_end: str = (_end_day + relativedelta(days=5)).strftime(_fmt_oracle).upper()

    day_key_start: str = _start_day.strftime(_fmt_oracle).upper()
    day_key_end: str = _end_day.strftime(_fmt_oracle).upper()


DP = DateParams()

print("=" * 60)
print("CHANNEL PREFERENCE DATE PARAMETERS")
print("-" * 60)
for k, v in vars(DP).items():
    print(f"{k:<25} : {v}")
print("=" * 60)


# =========================
# SQL Statements
# =========================
STATEMENTS: List[str] = [

    # -------------------------
    # Stage 1: Raw Channel Usage
    # -------------------------
    "DROP TABLE TBL_CHANNEL_PREFERENCE_01 PURGE",
    f"""
    CREATE TABLE TBL_CHANNEL_PREFERENCE_01
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           MSISDN,
           CASE
               WHEN Channel IN ('Scratch Card','EOCUSSD','SMS','EOCUMB121','CMPUMB',
                               'EOCCRM','EOCPAM','CMPUMB121','EOCNDNC')
               THEN 'USSD'
               WHEN Channel IN ('BKASHMFS','NAGAD','DBBLMFS')
               THEN 'MFS'
               WHEN Channel IN ('EOCMYGP','CMPMYGP','CRMMYGP','EOCFACEBOOK','PGWMYGP')
               THEN 'MyGP'
               WHEN Channel = 'IRIS'
               THEN 'TRIGGER'
           END AS PREFERRED_CHANNEL,
           TRUNC(SYSDATE) - TRUNC(DAY_KEY) AS DELTA_DAYS,
           REV AS PACK_REV
    FROM GPBI_MARKETING.GPBI_SUBSCRIPTION_REV_DAILY
    WHERE LOAD_DT BETWEEN '{DP.load_dt_start}' AND '{DP.load_dt_end}'
      AND DAY_KEY BETWEEN '{DP.day_key_start}' AND '{DP.day_key_end}'
      AND CHANNEL IN (
          'Scratch Card','EOCUSSD','BKASHMFS','SMS','IRIS','EOCMYGP',
          'CMPMYGP','EOCUMB121','CMPUMB','NAGAD','CRMMYGP','EOCCRM',
          'EOCFACEBOOK','EOCPAM','PGWMYGP','DBBLMFS','CMPUMB121','EOCNDNC'
      )
    """,

    # -------------------------
    # Stage 2: Channel Scoring
    # -------------------------
    "DROP TABLE TBL_CHANNEL_PREFERENCE PURGE",
    """
    CREATE TABLE TBL_CHANNEL_PREFERENCE
    COMPRESS FOR ARCHIVE HIGH AS
    WITH CHANNEL_SCORE AS (
        SELECT /*+ PARALLEL(8) */
               MSISDN,
               PREFERRED_CHANNEL,
               SUM(EXP(-(DELTA_DAYS / 30))) AS USAGE_SCORE,
               SUM(EXP(-(DELTA_DAYS / 30)) * LN(1 + PACK_REV)) AS REVENUE_SCORE,
               SUM(PACK_REV) AS PACK_REV
        FROM TBL_CHANNEL_PREFERENCE_01
        GROUP BY MSISDN, PREFERRED_CHANNEL
    )
    SELECT /*+ PARALLEL(8) */
           MSISDN,
           PREFERRED_CHANNEL,
           USAGE_SCORE,
           REVENUE_SCORE,
           USAGE_SCORE
             / NULLIF(SUM(USAGE_SCORE) OVER (PARTITION BY MSISDN), 0)
             AS PREFERENCE_SHARE,
           REVENUE_SCORE
             / NULLIF(SUM(REVENUE_SCORE) OVER (PARTITION BY MSISDN), 0)
             AS REVENUE_SHARE
    FROM CHANNEL_SCORE
    """,

    # -------------------------
    # Stage 3: Final Preference
    # -------------------------
    "DROP TABLE TBL_CHANNEL_PREFERENCE_FINAL_TEST PURGE",
    """
    CREATE TABLE TBL_CHANNEL_PREFERENCE_FINAL_TEST
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.*
    FROM (
        SELECT A.*,
               ROW_NUMBER() OVER (
                   PARTITION BY MSISDN
                   ORDER BY REVENUE_SHARE DESC
               ) AS CHANNEL_PREF_RANK,
               CASE
                   WHEN UPPER(PREFERRED_CHANNEL) = 'MYGP'    THEN 1
                   WHEN UPPER(PREFERRED_CHANNEL) = 'USSD'    THEN 2
                   WHEN UPPER(PREFERRED_CHANNEL) = 'MFS'     THEN 3
                   WHEN UPPER(PREFERRED_CHANNEL) = 'TRIGGER' THEN 4
               END AS CHANNEL_MAPPING
        FROM TBL_CHANNEL_PREFERENCE A
    ) A
    WHERE CHANNEL_PREF_RANK = 1
    """,
]


# =========================
# Utilities
# =========================
def table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM user_tables WHERE table_name = :tn",
        tn=table_name.upper(),
    )
    (cnt,) = cursor.fetchone()
    return cnt > 0


def sanity_check_rows(cursor, table_name: str, min_rows: int = 1):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    (cnt,) = cursor.fetchone()
    print(f"      ↳ Row count for {table_name}: {cnt:,}")
    if cnt < min_rows:
        print(f"      ⚠ WARNING: {table_name} has {cnt} rows")


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

    with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
        with conn.cursor() as cur:
            for i, stmt in enumerate(STATEMENTS, start=1):
                stmt = stmt.strip()
                if not stmt:
                    continue

                first_line = stmt.splitlines()[0]
                create_match = re.match(r"CREATE\s+TABLE\s+(\w+)", stmt, re.I)
                table_created = create_match.group(1) if create_match else None

                print(f"[{i}/{len(STATEMENTS)}] Executing: {first_line[:120]}")

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

    print("Channel preference pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
