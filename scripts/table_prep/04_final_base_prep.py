from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
import oracledb
from datetime import datetime, timedelta

load_dotenv()


# =========================
# Parameters
# =========================
@dataclass(frozen=True)
class Params:
    # Calculate current month details dynamically
    
    # Typically runs for the current month.
    # If today is Jan 15 2026, pred_month = 202601, mapping_day_key = 31-JAN-26
    date_ref = datetime.now()
    pred_month: str = date_ref.strftime("%Y%m")
    
    # Last day of the current month
    # Logic: First day of next month minus 1 day
    next_month = date_ref.replace(day=28) + timedelta(days=4)
    last_day = next_month - timedelta(days=next_month.day)
    mapping_day_key: str = last_day.strftime("%d-%b-%y").upper()


P = Params()

print("=" * 60)
print("FINAL PREDICTION BASE PARAMETERS")
print("-" * 60)
for k, v in vars(P).items():
    print(f"{k:<25} : {v}")
print("=" * 60)

PM = P.pred_month


# =========================
# SQL Statements
# =========================
STATEMENTS: List[str] = [

    # -------------------------------------------------
    # Unique MSISDN
    # -------------------------------------------------
    f"DROP TABLE SA_TBL_ATL_PRED_{PM}_UNIQUE PURGE",
    f"""
    CREATE TABLE SA_TBL_ATL_PRED_{PM}_UNIQUE AS
    SELECT DISTINCT MSISDN
    FROM SA_TBL_ATL_PRED_{PM}
    """,

    # -------------------------------------------------
    # Control / Test Group
    # -------------------------------------------------
    f"DROP TABLE TBL_ATL_PREDICTION_{PM}_1 PURGE",
    f"""
    CREATE TABLE TBL_ATL_PREDICTION_{PM}_1
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.*,
           CASE
               WHEN B.MSISDN IS NULL THEN 'TG'
               ELSE 'CG'
           END AS GROUP_TYPE
    FROM SA_TBL_ATL_PRED_{PM} A
    LEFT JOIN SA_TBL_ATL_PRED_{PM}_UNIQUE SAMPLE(5) B
           ON A.MSISDN = B.MSISDN
    WHERE A.MSISDN NOT IN (
        SELECT DISTINCT MSISDN
        FROM GPBI_MARKETING.GPBI_MSISDN_EXCLUSION_LIST
        WHERE TYPE IN ('SKITTO')
    )
    """,

    # -------------------------------------------------
    # Recommendation Segments
    # -------------------------------------------------
    "DROP TABLE TMP_RECO_SEGM_1 PURGE",
    f"""
    CREATE TABLE TMP_RECO_SEGM_1
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT DISTINCT MSISDN
    FROM TBL_ATL_PREDICTION_{PM}_1
    WHERE GROUP_TYPE <> 'CG'
    """,

    "DROP TABLE TMP_RECO_SEGM_2 PURGE",
    """
    CREATE TABLE TMP_RECO_SEGM_2
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT A.MSISDN,
           CASE
               WHEN B.MSISDN IS NULL THEN 'UPSELL'
               ELSE 'PROB'
           END AS SEGM_TAG
    FROM TMP_RECO_SEGM_1 A
    LEFT JOIN (
        SELECT MSISDN FROM TMP_RECO_SEGM_1 SAMPLE(70)
    ) B
      ON A.MSISDN = B.MSISDN
    """,

    # -------------------------------------------------
    # Upsell Mapping
    # -------------------------------------------------
    f"DROP TABLE TBL_ATL_PREDICTION_{PM}_UPSELL PURGE",
    f"""
    CREATE TABLE TBL_ATL_PREDICTION_{PM}_UPSELL
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT MSISDN,
    CASE	WHEN	DENO	=	98	THEN	118
            WHEN	DENO	=	118	THEN	148
            WHEN	DENO	=	148	THEN	179
            WHEN	DENO	=	179	THEN	198
            WHEN	DENO	=	198	THEN	219
            WHEN	DENO	=	219	THEN	249
            WHEN	DENO	=	249	THEN	308
            WHEN	DENO	=	308	THEN	399
            WHEN	DENO	=	399	THEN	499
            WHEN	DENO	=	499	THEN	599
            WHEN	DENO	=	599	THEN	698
            WHEN	DENO	=	698	THEN	798
            WHEN	DENO	=	798	THEN	898
            WHEN	DENO	=	898	THEN	899
                                
            WHEN	DENO	=	217	THEN	258
            WHEN	DENO	=	258	THEN	398
            WHEN	DENO	=	398	THEN	498
            WHEN	DENO	=	498	THEN	598
            WHEN	DENO	=	598	THEN	699
            WHEN	DENO	=	699	THEN	818
            WHEN	DENO	=	818	THEN	999
            WHEN	DENO	=	999	THEN	1099
            WHEN	DENO	=	1099	THEN	1199
                                
            WHEN	DENO	=	59	THEN	89
            WHEN	DENO	=	89	THEN	107
            WHEN	DENO	=	107	THEN	129
            WHEN	DENO	=	129	THEN	208
            WHEN	DENO	=	208	THEN	309
            WHEN	DENO	=	309	THEN	509
            WHEN	DENO	=	509	THEN	989
                                
            WHEN	DENO	=	9	THEN	11
            WHEN	DENO	=	11	THEN	15
            WHEN	DENO	=	15	THEN	19
            WHEN	DENO	=	19	THEN	29
            WHEN	DENO	=	29	THEN	39
            WHEN	DENO	=	39	THEN	49
            WHEN	DENO	=	49	THEN	68
            WHEN	DENO	=	68	THEN	99
            WHEN	DENO	=	99	THEN	119
            WHEN	DENO	=	119	THEN	139
            WHEN	DENO	=	139	THEN	158
            WHEN	DENO	=	158	THEN	197
            WHEN	DENO	=	197	THEN	228
            WHEN	DENO	=	228	THEN	268
            WHEN	DENO	=	268	THEN	279
            WHEN	DENO	=	279	THEN	319
            WHEN	DENO	=	319	THEN	339
            WHEN	DENO	=	339	THEN	419
            WHEN	DENO	=	419	THEN	519
            WHEN	DENO	=	519	THEN	547
            WHEN	DENO	=	547	THEN	639
            WHEN	DENO	=	639	THEN	729
            ELSE TO_NUMBER(DENO)
           END AS DENO,
           PROB,
           RANK,
           'TG_UPSELL' AS GROUP_TYPE
    FROM TBL_ATL_PREDICTION_{PM}_1
    WHERE MSISDN IN (
        SELECT MSISDN FROM TMP_RECO_SEGM_2 WHERE SEGM_TAG = 'UPSELL'
    )
    """,

    # -------------------------------------------------
    # Prob Group
    # -------------------------------------------------
    f"DROP TABLE TBL_ATL_PREDICTION_{PM}_PROB PURGE",
    f"""
    CREATE TABLE TBL_ATL_PREDICTION_{PM}_PROB
    COMPRESS FOR ARCHIVE AS
    SELECT MSISDN, DENO, PROB, RANK, 'TG_PROB' AS GROUP_TYPE
    FROM TBL_ATL_PREDICTION_{PM}_1
    WHERE MSISDN IN (
        SELECT MSISDN FROM TMP_RECO_SEGM_2 WHERE SEGM_TAG = 'PROB'
    )
    """,

    # -------------------------------------------------
    # Union All Groups
    # -------------------------------------------------
    f"DROP TABLE TBL_ATL_PREDICTION_{PM}_2 PURGE",
    f"""
    CREATE TABLE TBL_ATL_PREDICTION_{PM}_2
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT TO_NUMBER(MSISDN) MSISDN,
           TO_NUMBER(DENO) DENO,
           CAST(PROB AS FLOAT) PROB,
           TO_NUMBER(RANK) RANK,
           CAST(GROUP_TYPE AS VARCHAR2(32)) GROUP_TYPE
    FROM TBL_ATL_PREDICTION_{PM}_UPSELL
    UNION
    SELECT TO_NUMBER(MSISDN), TO_NUMBER(DENO),
           CAST(PROB AS FLOAT), TO_NUMBER(RANK),
           CAST(GROUP_TYPE AS VARCHAR2(32))
    FROM TBL_ATL_PREDICTION_{PM}_PROB
    UNION
    SELECT TO_NUMBER(MSISDN), TO_NUMBER(DENO),
           CAST(PROB AS FLOAT), TO_NUMBER(RANK),
           CAST(GROUP_TYPE AS VARCHAR2(32))
    FROM TBL_ATL_PREDICTION_{PM}_1
    WHERE GROUP_TYPE = 'CG'
    """,

    # -------------------------------------------------
    # Mapping Identifier
    # -------------------------------------------------
    f"DROP TABLE TBL_ATL_PREDICTION_{PM}_3 PURGE",
    f"""
    CREATE TABLE TBL_ATL_PREDICTION_{PM}_3
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.*, B.IDENTIFIER, B.SRVC
    FROM TBL_ATL_PREDICTION_{PM}_2 A
    INNER JOIN (
        SELECT DISTINCT AMOUNT, IDENTIFIER, SRVC
        FROM TBL_ATL_MAPPING
        WHERE DAY_KEY = '{P.mapping_day_key}'
    ) B
      ON A.DENO = B.AMOUNT
    """,

    # -------------------------------------------------
    # Final Prediction Table
    # -------------------------------------------------
    f"DROP TABLE TBL_ATL_PREDICTION_{PM} PURGE",
    f"""
    CREATE TABLE TBL_ATL_PREDICTION_{PM}
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.*,
           B.PREFERRED_CHANNEL,
           NVL(B.CHANNEL_MAPPING, 2) AS CHANNEL_MAPPING
    FROM TBL_ATL_PREDICTION_{PM}_3 A
    LEFT JOIN (
        SELECT DISTINCT MSISDN, PREFERRED_CHANNEL, CHANNEL_MAPPING
        FROM TBL_CHANNEL_PREFERENCE_FINAL
    ) B
      ON A.MSISDN = B.MSISDN
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

    print("Final prediction base creation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
