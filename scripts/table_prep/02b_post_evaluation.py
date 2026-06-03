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
class EvalParams:
    date_ref = datetime.now()
    
    # Prediction month (current month)
    pred_month: str = date_ref.strftime("%Y%m")
    
    # Mapping day key (last day of current month)
    next_month = date_ref.replace(day=28) + timedelta(days=4)
    last_day = next_month - timedelta(days=next_month.day)
    mapping_day_key: str = last_day.strftime("%d-%b-%y").upper()
    
    # Evaluation window: 1st of current month to today
    _fmt_oracle = "%d-%b-%y"
    eval_start: str = date_ref.replace(day=1).strftime(_fmt_oracle).upper()
    eval_end: str = date_ref.strftime(_fmt_oracle).upper()
    
    # Load date buffer
    eval_load_dt_start: str = (date_ref.replace(day=1) - timedelta(days=5)).strftime(_fmt_oracle).upper()
    eval_load_dt_end: str = (date_ref + timedelta(days=5)).strftime(_fmt_oracle).upper()
    
    eval_rc_load_dt_start: str = eval_start
    eval_rc_load_dt_end: str = eval_end


EP = EvalParams()
PM = EP.pred_month

print("=" * 60)
print(f"EVALUATION PARAMETERS (Ref: {EvalParams.date_ref})")
print("-" * 60)
for k, v in vars(EP).items():
    print(f"{k:<30} : {v}")
print("=" * 60)


# =========================
# SQL Statements
# =========================
STATEMENTS: List[str] = [

    # 1-2: Rental pack takers for current month (up to today)
#    f"DROP TABLE SA_TBL_EVAL_RENTAL_TAKERS_{PM} PURGE",
    f"""
    CREATE TABLE SA_TBL_EVAL_RENTAL_TAKERS_{PM} COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           B.MSISDN, A.AMOUNT, SUM(NVL(HIT,0)) AS HIT
    FROM GPBI_MARKETING.GPBI_SUBSCRIPTION_REV_DAILY B
    INNER JOIN (
        SELECT * FROM TBL_ATL_MAPPING
        WHERE DAY_KEY = '{EP.mapping_day_key}'
    ) A
      ON A.PACK = B.PACK
    WHERE LOAD_DT BETWEEN '{EP.eval_load_dt_start}' AND '{EP.eval_load_dt_end}'
      AND B.DAY_KEY BETWEEN '{EP.eval_start}' AND '{EP.eval_end}'
      AND A.PACK IS NOT NULL
    GROUP BY B.MSISDN, A.AMOUNT
    """,

    # 3-4: RC takers for current month
#    f"DROP TABLE SA_TBL_EVAL_RC_TAKERS_{PM} PURGE",
    f"""
    CREATE TABLE SA_TBL_EVAL_RC_TAKERS_{PM} COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.PYMT_RSLT_CD AS MSISDN,
           B.AMOUNT,
           COUNT(*) AS HIT
    FROM OCDM_SYS.DWB_PRPD_RCHRG A
    INNER JOIN (
        SELECT * FROM TBL_ATL_MAPPING
        WHERE DAY_KEY = '{EP.mapping_day_key}'
          AND UPPER(SRVC) = 'RC'
    ) B
      ON ROUND(A.PYMT_AMT) = B.PACK
    WHERE A.LOAD_DT BETWEEN '{EP.eval_rc_load_dt_start}' AND '{EP.eval_rc_load_dt_end}'
    GROUP BY A.PYMT_RSLT_CD, B.AMOUNT
    """,

    # 5-6: Union all takers
 #   f"DROP TABLE SA_TBL_EVAL_PACK_TAKERS_{PM} PURGE",
    f"""
    CREATE TABLE SA_TBL_EVAL_PACK_TAKERS_{PM} COMPRESS FOR ARCHIVE HIGH AS
    SELECT * FROM (
        SELECT MSISDN, AMOUNT, SUM(HIT) HIT
        FROM SA_TBL_EVAL_RENTAL_TAKERS_{PM}
        GROUP BY MSISDN, AMOUNT
        UNION
        SELECT MSISDN, AMOUNT, HIT
        FROM SA_TBL_EVAL_RC_TAKERS_{PM}
    )
    """,

    # 7-8: Evaluate — join predictions with actual takers
    f"DROP TABLE SA_TBL_EVAL_RESULTS_{PM} PURGE",
    f"""
    CREATE TABLE SA_TBL_EVAL_RESULTS_{PM} COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.MSISDN,
           A.DENO AS PRED_DENO,
           A.PROB,
           A.RANK AS PRED_RANK,
           A.TAKER_FLAG,
           B.AMOUNT AS ACTUAL_DENO,
           B.HIT AS ACTUAL_HIT,
           CASE WHEN B.AMOUNT IS NOT NULL THEN 1 ELSE 0 END AS IS_HIT
    FROM SA_TBL_ATL_PRED_{PM} A
    LEFT JOIN SA_TBL_EVAL_PACK_TAKERS_{PM} B
      ON A.MSISDN = B.MSISDN AND A.DENO = B.AMOUNT
    """,
]


# =========================
# Evaluation Metrics
# =========================
EVAL_QUERIES = {
    "total_predictions": f"""
        SELECT COUNT(*) FROM SA_TBL_ATL_PRED_{PM}
    """,
    "unique_subscribers_predicted": f"""
        SELECT COUNT(DISTINCT MSISDN) FROM SA_TBL_ATL_PRED_{PM}
    """,
    "actual_takers_this_month": f"""
        SELECT COUNT(DISTINCT MSISDN) FROM SA_TBL_EVAL_PACK_TAKERS_{PM}
    """,
    "predicted_subs_who_took_any_pack": f"""
        SELECT COUNT(DISTINCT MSISDN)
        FROM SA_TBL_EVAL_RESULTS_{PM}
        WHERE ACTUAL_DENO IS NOT NULL
    """,
    "hit_at_1": f"""
        SELECT ROUND(
            COUNT(DISTINCT CASE WHEN IS_HIT = 1 AND PRED_RANK = 1 THEN MSISDN END) * 100.0 /
            NULLIF(COUNT(DISTINCT MSISDN), 0), 2
        ) AS HIT_AT_1_PCT
        FROM SA_TBL_EVAL_RESULTS_{PM}
        WHERE MSISDN IN (SELECT DISTINCT MSISDN FROM SA_TBL_EVAL_PACK_TAKERS_{PM})
    """,
    "hit_at_3": f"""
        SELECT ROUND(
            COUNT(DISTINCT CASE WHEN IS_HIT = 1 AND PRED_RANK <= 3 THEN MSISDN END) * 100.0 /
            NULLIF(COUNT(DISTINCT MSISDN), 0), 2
        ) AS HIT_AT_3_PCT
        FROM SA_TBL_EVAL_RESULTS_{PM}
        WHERE MSISDN IN (SELECT DISTINCT MSISDN FROM SA_TBL_EVAL_PACK_TAKERS_{PM})
    """,
    "hit_at_5": f"""
        SELECT ROUND(
            COUNT(DISTINCT CASE WHEN IS_HIT = 1 AND PRED_RANK <= 5 THEN MSISDN END) * 100.0 /
            NULLIF(COUNT(DISTINCT MSISDN), 0), 2
        ) AS HIT_AT_5_PCT
        FROM SA_TBL_EVAL_RESULTS_{PM}
        WHERE MSISDN IN (SELECT DISTINCT MSISDN FROM SA_TBL_EVAL_PACK_TAKERS_{PM})
    """,
    "hit_at_1_taker": f"""
        SELECT ROUND(
            COUNT(DISTINCT CASE WHEN IS_HIT = 1 AND PRED_RANK = 1 THEN MSISDN END) * 100.0 /
            NULLIF(COUNT(DISTINCT MSISDN), 0), 2
        ) AS HIT_AT_1_TAKER_PCT
        FROM SA_TBL_EVAL_RESULTS_{PM}
        WHERE TAKER_FLAG = 1
          AND MSISDN IN (SELECT DISTINCT MSISDN FROM SA_TBL_EVAL_PACK_TAKERS_{PM})
    """,
    "hit_at_1_nontaker": f"""
        SELECT ROUND(
            COUNT(DISTINCT CASE WHEN IS_HIT = 1 AND PRED_RANK = 1 THEN MSISDN END) * 100.0 /
            NULLIF(COUNT(DISTINCT MSISDN), 0), 2
        ) AS HIT_AT_1_NONTAKER_PCT
        FROM SA_TBL_EVAL_RESULTS_{PM}
        WHERE TAKER_FLAG = 0
          AND MSISDN IN (SELECT DISTINCT MSISDN FROM SA_TBL_EVAL_PACK_TAKERS_{PM})
    """,
    "top_predicted_packs": f"""
        SELECT PRED_DENO, COUNT(*) AS PRED_COUNT,
               SUM(IS_HIT) AS HIT_COUNT,
               ROUND(SUM(IS_HIT)*100.0/NULLIF(COUNT(*),0), 2) AS HIT_RATE_PCT
        FROM SA_TBL_EVAL_RESULTS_{PM}
        WHERE PRED_RANK = 1
        GROUP BY PRED_DENO
        ORDER BY PRED_COUNT DESC
        FETCH FIRST 15 ROWS ONLY
    """,
}


# =========================
# Execution
# =========================
def table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM user_tables WHERE table_name = :tn",
        tn=table_name.upper(),
    )
    (cnt,) = cursor.fetchone()
    return cnt > 0


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
            # --- Phase 1: Create evaluation tables ---
            print("\n" + "=" * 60)
            print("PHASE 1: Building actual taker tables & evaluation join")
            print("=" * 60)

            for i, stmt in enumerate(STATEMENTS, start=1):
                stmt = stmt.strip()
                if not stmt:
                    continue

                first_line = stmt.splitlines()[0]
                m = re.match(r"CREATE\s+TABLE\s+(\w+)", stmt, re.I)
                if m and table_exists(cur, m.group(1)):
                    print(f"[{i}] Skipping existing table {m.group(1)}")
                    continue

                print(f"[{i}] Executing: {first_line[:120]}")
                try:
                    cur.execute(stmt)
                    conn.commit()
                except oracledb.Error as e:
                    if "ORA-00942" in str(e) and stmt.upper().startswith("DROP TABLE"):
                        print("      (ignored) ORA-00942")
                        conn.commit()
                        continue
                    raise

            # --- Phase 2: Compute & display metrics ---
            print("\n" + "=" * 60)
            print(f"PHASE 2: Evaluation Metrics ({EP.eval_start} to {EP.eval_end})")
            print("=" * 60)

            for metric_name, query in EVAL_QUERIES.items():
                try:
                    cur.execute(query.strip())
                    if metric_name == "top_predicted_packs":
                        rows = cur.fetchall()
                        print(f"\n  {metric_name}:")
                        print(f"  {'DENO':<8} {'PRED_CNT':<10} {'HIT_CNT':<10} {'HIT_RATE%':<10}")
                        print(f"  {'-'*38}")
                        for row in rows:
                            print(f"  {row[0]:<8} {row[1]:<10} {row[2]:<10} {row[3]:<10}")
                    else:
                        (val,) = cur.fetchone()
                        print(f"  {metric_name:<40}: {val}")
                except oracledb.Error as e:
                    print(f"  {metric_name:<40}: ERROR - {e}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())