from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from dotenv import load_dotenv
import oracledb

load_dotenv()

_FMT_ORACLE = "%d-%b-%y"


# =========================
# Parameters
# =========================
@dataclass(frozen=True)
class EvalParams:
    # Prediction month (YYYYMM)
    pred_month: str

    # Mapping day key (last day of the prediction month)
    mapping_day_key: str

    # Evaluation window (Oracle DD-MON-YY)
    eval_start: str
    eval_end: str

    # Load date buffer
    eval_load_dt_start: str
    eval_load_dt_end: str

    eval_rc_load_dt_start: str
    eval_rc_load_dt_end: str


def _last_day_of_month(d: datetime) -> datetime:
    next_month = d.replace(day=28) + timedelta(days=4)
    return next_month - timedelta(days=next_month.day)


def build_params(pred_month: str, eval_start: datetime, eval_end: datetime,
                 mapping_day_key: str | None = None) -> EvalParams:
    """Build evaluation parameters from a prediction month and an explicit
    evaluation window (start/end dates)."""
    pm_dt = datetime.strptime(pred_month, "%Y%m")

    if mapping_day_key is None:
        mapping_day_key = _last_day_of_month(pm_dt).strftime(_FMT_ORACLE).upper()

    eval_start_str = eval_start.strftime(_FMT_ORACLE).upper()
    eval_end_str = eval_end.strftime(_FMT_ORACLE).upper()

    return EvalParams(
        pred_month=pred_month,
        mapping_day_key=mapping_day_key,
        eval_start=eval_start_str,
        eval_end=eval_end_str,
        eval_load_dt_start=(eval_start - timedelta(days=5)).strftime(_FMT_ORACLE).upper(),
        eval_load_dt_end=(eval_end + timedelta(days=5)).strftime(_FMT_ORACLE).upper(),
        eval_rc_load_dt_start=eval_start_str,
        eval_rc_load_dt_end=eval_end_str,
    )


# =========================
# SQL Statements
# =========================
def build_statements(EP: EvalParams) -> List[str]:
    PM = EP.pred_month
    return [

        # # 1-2: Rental pack takers for the evaluation window
        # f"DROP TABLE SA_TBL_EVAL_RENTAL_TAKERS_{PM} PURGE",
        # f"""
        # CREATE TABLE SA_TBL_EVAL_RENTAL_TAKERS_{PM} COMPRESS FOR ARCHIVE HIGH AS
        # SELECT /*+ PARALLEL(8) */
        #        B.MSISDN, A.AMOUNT, SUM(NVL(HIT,0)) AS HIT
        # FROM GPBI_MARKETING.GPBI_SUBSCRIPTION_REV_DAILY B
        # INNER JOIN (
        #     SELECT * FROM TBL_ATL_MAPPING
        #     WHERE DAY_KEY = '{EP.mapping_day_key}'
        # ) A
        #   ON A.AMOUNT = B.AMOUNT
        # WHERE LOAD_DT BETWEEN '{EP.eval_load_dt_start}' AND '{EP.eval_load_dt_end}'
        #   AND B.DAY_KEY BETWEEN '{EP.eval_start}' AND '{EP.eval_end}'
        #   AND A.PACK IS NOT NULL
        # GROUP BY B.MSISDN, A.AMOUNT
        # """,

        # # 3-4: RC takers for the evaluation window
        # f"DROP TABLE SA_TBL_EVAL_RC_TAKERS_{PM} PURGE",
        # f"""
        # CREATE TABLE SA_TBL_EVAL_RC_TAKERS_{PM} COMPRESS FOR ARCHIVE HIGH AS
        # SELECT /*+ PARALLEL(8) */
        #        A.PYMT_RSLT_CD AS MSISDN,
        #        B.AMOUNT,
        #        COUNT(*) AS HIT
        # FROM OCDM_SYS.DWB_PRPD_RCHRG A
        # INNER JOIN (
        #     SELECT * FROM TBL_ATL_MAPPING
        #     WHERE DAY_KEY = '{EP.mapping_day_key}'
        #       AND UPPER(SRVC) = 'RC'
        # ) B
        #   ON ROUND(A.PYMT_AMT) = B.AMOUNT
        # WHERE A.LOAD_DT BETWEEN '{EP.eval_rc_load_dt_start}' AND '{EP.eval_rc_load_dt_end}'
        # GROUP BY A.PYMT_RSLT_CD, B.AMOUNT
        # """,

        # # 5-6: Union all takers
        # f"DROP TABLE SA_TBL_EVAL_PACK_TAKERS_{PM} PURGE",
        # f"""
        # CREATE TABLE SA_TBL_EVAL_PACK_TAKERS_{PM} COMPRESS FOR ARCHIVE HIGH AS
        # SELECT * FROM (
        #     SELECT MSISDN, AMOUNT, SUM(HIT) HIT
        #     FROM SA_TBL_EVAL_RENTAL_TAKERS_{PM}
        #     GROUP BY MSISDN, AMOUNT
        #     UNION
        #     SELECT MSISDN, AMOUNT, HIT
        #     FROM SA_TBL_EVAL_RC_TAKERS_{PM}
        # )
        # """,

        # 7-8: Evaluate — join predictions with actual takers
        # f"DROP TABLE SA_TBL_EVAL_RESULTS_{PM} PURGE",
        # f"""
        # CREATE TABLE SA_TBL_EVAL_RESULTS_{PM} COMPRESS FOR ARCHIVE HIGH AS
        # SELECT /*+ PARALLEL(8) */
        #        A.MSISDN,
        #        A.DENO AS PRED_DENO,
        #        A.PROB,
        #        A.RANK AS PRED_RANK,
        #        B.AMOUNT AS ACTUAL_DENO,
        #        B.HIT AS ACTUAL_HIT,
        #        CASE WHEN B.AMOUNT IS NOT NULL THEN 1 ELSE 0 END AS IS_HIT
        # FROM (SELECT * FROM TBL_ATL_PREDICTION_{PM} WHERE GROUP_TYPE LIKE 'TG%') A
        # LEFT JOIN SA_TBL_EVAL_PACK_TAKERS_{PM} B
        #   ON A.MSISDN = B.MSISDN AND A.DENO = B.AMOUNT
        # """,
    ]


# =========================
# Evaluation Metrics
# =========================
def build_eval_queries(PM: str) -> dict:
    return {
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
              AND MSISDN IN (SELECT DISTINCT MSISDN FROM SA_TBL_EVAL_PACK_TAKERS_{PM})
        """,
        # "hit_at_1_nontaker": f"""
        #     SELECT ROUND(
        #         COUNT(DISTINCT CASE WHEN IS_HIT = 1 AND PRED_RANK = 1 THEN MSISDN END) * 100.0 /
        #         NULLIF(COUNT(DISTINCT MSISDN), 0), 2
        #     ) AS HIT_AT_1_NONTAKER_PCT
        #     FROM SA_TBL_EVAL_RESULTS_{PM}
        #     WHERE TAKER_FLAG = 'NON_TAKER'
        #       AND MSISDN IN (SELECT DISTINCT MSISDN FROM SA_TBL_EVAL_PACK_TAKERS_{PM})
        # """,
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
# CLI
# =========================
def _parse_date(s: str) -> datetime:
    """Accept ISO (YYYY-MM-DD) or Oracle-style (DD-MON-YY) dates."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s.upper(), fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date {s!r}; use YYYY-MM-DD or DD-MON-YY (e.g. 2026-07-01 or 01-JUL-26)"
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dynamic post-evaluation for ATL recommendations. "
                    "Select the prediction month and evaluation window."
    )
    p.add_argument(
        "--pred-month", required=True, metavar="YYYYMM",
        help="Prediction month, e.g. 202607. Also names the SA_TBL_ATL_PRED_<PM> table.",
    )
    p.add_argument(
        "--eval-start", required=True, type=_parse_date, metavar="DATE",
        help="Evaluation window start date (YYYY-MM-DD or DD-MON-YY).",
    )
    p.add_argument(
        "--eval-end", required=True, type=_parse_date, metavar="DATE",
        help="Evaluation window end date (YYYY-MM-DD or DD-MON-YY).",
    )
    p.add_argument(
        "--mapping-day-key", default=None, metavar="DD-MON-YY",
        help="Override the TBL_ATL_MAPPING day key. "
             "Defaults to the last day of the prediction month.",
    )
    p.add_argument(
        "--rebuild", action="store_true",
        help="Drop and rebuild the evaluation tables even if they already exist.",
    )
    return p.parse_args(argv)


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


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    PM = args.pred_month
    if not re.fullmatch(r"\d{6}", PM):
        print(f"Invalid pred_month={PM}; expected YYYYMM", file=sys.stderr)
        return 3

    if args.eval_start > args.eval_end:
        print(f"eval_start ({args.eval_start.date()}) is after eval_end "
              f"({args.eval_end.date()})", file=sys.stderr)
        return 3

    EP = build_params(PM, args.eval_start, args.eval_end, args.mapping_day_key)

    print("=" * 60)
    print("EVALUATION PARAMETERS")
    print("-" * 60)
    for k, v in vars(EP).items():
        print(f"{k:<30} : {v}")
    print("=" * 60)

    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")

    if not all([user, password, dsn]):
        print("Missing ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN", file=sys.stderr)
        return 2

    statements = build_statements(EP)
    eval_queries = build_eval_queries(PM)

    with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
        with conn.cursor() as cur:
            # --- Phase 1: Create evaluation tables ---
            print("\n" + "=" * 60)
            print("PHASE 1: Building actual taker tables & evaluation join")
            print("=" * 60)

            for i, stmt in enumerate(statements, start=1):
                stmt = stmt.strip()
                if not stmt:
                    continue

                first_line = stmt.splitlines()[0]
                m = re.match(r"CREATE\s+TABLE\s+(\w+)", stmt, re.I)
                if m and not args.rebuild and table_exists(cur, m.group(1)):
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

            for metric_name, query in eval_queries.items():
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
