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
# Date Parameter Definition
# =========================
@dataclass(frozen=True)
class DateParams:
    # Fixed reference date for inference run
    _ref_date = datetime.date.today()

    # Label month = Dec 2025
    _label_date = _ref_date.replace(day=1) - relativedelta(months=1)

    # Formats
    _fmt_mon = "%Y%m"
    _fmt_oracle = "%d-%b-%y"

    # Mapping snapshot (Jan end)
    mapping_day_key: str = (_label_date + relativedelta(months=1, day=31)).strftime(_fmt_oracle).upper()

    # Inference month
    infer_month_key: str = _label_date.strftime(_fmt_mon)

    # Feature window: Oct–Dec 2025
    _infer_start = (_label_date - relativedelta(months=2)).replace(day=1)
    _infer_end = _label_date + relativedelta(day=31)

    infer_load_dt_start: str = (_infer_start - relativedelta(days=6)).strftime(_fmt_oracle).upper()
    infer_load_dt_end: str = (_infer_end + relativedelta(days=5)).strftime(_fmt_oracle).upper()

    infer_day_key_start: str = _infer_start.strftime(_fmt_oracle).upper()
    infer_day_key_end: str = _infer_end.strftime(_fmt_oracle).upper()

    infer_rc_load_dt_start: str = infer_day_key_start
    infer_rc_load_dt_end: str = infer_day_key_end

    joining_date_cutoff: str = infer_day_key_start

    # Partitions
    part_t: str = f"PART_{_infer_end.strftime('%d%m%Y')}"
    part_t_1: str = f"PART_{(_infer_end.replace(day=1) - relativedelta(days=1)).strftime('%d%m%Y')}"
    part_t_2: str = f"PART_{((_infer_end.replace(day=1) - relativedelta(days=1)).replace(day=1) - relativedelta(days=1)).strftime('%d%m%Y')}"


DP = DateParams()

print("=" * 60)
print(f"INFERENCE DATE PARAMETERS LOG (Ref: {DateParams._ref_date})")
print("-" * 60)
for k, v in vars(DP).items():
    print(f"{k:<30} : {v}")
print("=" * 60)

INFER = DP.infer_month_key


# =========================
# SQL Statements
# =========================
STATEMENTS: List[str] = [

    # 1–2 Rental Pack Aggregation
    f"DROP TABLE SA_TBL_AGG_RENTAL_PACK_TAKERS_{INFER} PURGE",
    f"""
    CREATE TABLE SA_TBL_AGG_RENTAL_PACK_TAKERS_{INFER} COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           B.MSISDN, A.PACK, A.AMOUNT,
           SUM(NVL(REV,0)) AS PACK_REV,
           SUM(NVL(HIT,0)) AS HIT
    FROM GPBI_MARKETING.GPBI_SUBSCRIPTION_REV_DAILY B
    LEFT JOIN (
        SELECT * FROM TBL_ATL_MAPPING
        WHERE DAY_KEY = '{DP.mapping_day_key}'
    ) A
      ON A.PACK = B.PACK
    WHERE LOAD_DT BETWEEN '{DP.infer_load_dt_start}' AND '{DP.infer_load_dt_end}'
      AND B.DAY_KEY BETWEEN '{DP.infer_day_key_start}' AND '{DP.infer_day_key_end}'
      AND A.PACK IS NOT NULL
    GROUP BY B.MSISDN, A.PACK, A.AMOUNT
    """,

    # 3–4 RC Aggregation
    f"DROP TABLE SA_TBL_AGG_RC_TAKER_{INFER} PURGE",
    f"""
    CREATE TABLE SA_TBL_AGG_RC_TAKER_{INFER} COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.PYMT_RSLT_CD AS MSISDN,
           B.AMOUNT,
           COUNT(*) AS HIT
    FROM OCDM_SYS.DWB_PRPD_RCHRG A
    INNER JOIN (
        SELECT * FROM TBL_ATL_MAPPING
        WHERE DAY_KEY = '{DP.mapping_day_key}'
          AND UPPER(SRVC) = 'RC'
    ) B
      ON ROUND(A.PYMT_AMT) = B.PACK
    WHERE A.LOAD_DT BETWEEN '{DP.infer_rc_load_dt_start}' AND '{DP.infer_rc_load_dt_end}'
    GROUP BY A.PYMT_RSLT_CD, B.AMOUNT
    """,

    # 5–6 Union Pack Takers
    f"DROP TABLE SA_TBL_AGG_PACK_TAKERS_{INFER} PURGE",
    f"""
    CREATE TABLE SA_TBL_AGG_PACK_TAKERS_{INFER} COMPRESS FOR ARCHIVE HIGH AS
    SELECT *
    FROM (
        SELECT MSISDN, AMOUNT, SUM(HIT) HIT
        FROM SA_TBL_AGG_RENTAL_PACK_TAKERS_{INFER}
        GROUP BY MSISDN, AMOUNT
        UNION
        SELECT MSISDN, AMOUNT, HIT
        FROM SA_TBL_AGG_RC_TAKER_{INFER}
    )
    """,

    # 7–8 Ranking
    f"DROP TABLE SA_TBL_AGG_PACK_TAKERS_{INFER}_1 PURGE",
    f"""
    CREATE TABLE SA_TBL_AGG_PACK_TAKERS_{INFER}_1 NOLOGGING COMPRESS FOR ARCHIVE HIGH AS
    SELECT A.*,
           ROW_NUMBER() OVER (PARTITION BY MSISDN ORDER BY HIT DESC, AMOUNT DESC) AS HIT_RANK,
           ROW_NUMBER() OVER (PARTITION BY MSISDN ORDER BY HIT ASC, AMOUNT DESC) AS HIT_RANK_MIN
    FROM SA_TBL_AGG_PACK_TAKERS_{INFER} A
    """,

    # 9–10 Pivot
    f"DROP TABLE SA_TBL_AGG_PACK_TAKERS_{INFER}_2 PURGE",
    f"""
    CREATE TABLE SA_TBL_AGG_PACK_TAKERS_{INFER}_2 NOLOGGING COMPRESS FOR ARCHIVE HIGH AS
    SELECT MSISDN,
           MAX(CASE WHEN AMOUNT1 = 1 THEN AMOUNT_MIN ELSE AMOUNT1 END) AMOUNT1,
           MAX(CASE WHEN AMOUNT2 = 1 THEN AMOUNT_MIN ELSE AMOUNT2 END) AMOUNT2,
           MAX(CASE WHEN AMOUNT3 = 1 THEN AMOUNT_MIN ELSE AMOUNT3 END) AMOUNT3,
           MAX(CASE WHEN AMOUNT4 = 1 THEN AMOUNT_MIN ELSE AMOUNT4 END) AMOUNT4,
           MAX(CASE WHEN AMOUNT5 = 1 THEN AMOUNT_MIN ELSE AMOUNT5 END) AMOUNT5,
           MAX(CASE WHEN AMOUNT6 = 1 THEN AMOUNT_MIN ELSE AMOUNT6 END) AMOUNT6,
           MAX(CASE WHEN AMOUNT7 = 1 THEN AMOUNT_MIN ELSE AMOUNT7 END) AMOUNT7,
           MAX(CASE WHEN AMOUNT8 = 1 THEN AMOUNT_MIN ELSE AMOUNT8 END) AMOUNT8
    FROM (
        SELECT MSISDN,
               NVL(MAX(CASE WHEN HIT_RANK = 1 THEN AMOUNT END),1) AMOUNT1,
               NVL(MAX(CASE WHEN HIT_RANK = 2 THEN AMOUNT END),1) AMOUNT2,
               NVL(MAX(CASE WHEN HIT_RANK = 3 THEN AMOUNT END),1) AMOUNT3,
               NVL(MAX(CASE WHEN HIT_RANK = 4 THEN AMOUNT END),1) AMOUNT4,
               NVL(MAX(CASE WHEN HIT_RANK = 5 THEN AMOUNT END),1) AMOUNT5,
               NVL(MAX(CASE WHEN HIT_RANK = 6 THEN AMOUNT END),1) AMOUNT6,
               NVL(MAX(CASE WHEN HIT_RANK = 7 THEN AMOUNT END),1) AMOUNT7,
               NVL(MAX(CASE WHEN HIT_RANK = 8 THEN AMOUNT END),1) AMOUNT8,
               NVL(MAX(CASE WHEN HIT_RANK_MIN = 1 THEN AMOUNT END),1) AMOUNT_MIN
        FROM SA_TBL_AGG_PACK_TAKERS_{INFER}_1
        GROUP BY MSISDN
    )
    GROUP BY MSISDN
    """,

    # 11–12 Encode Services
    f"DROP TABLE SA_TBL_AGG_PACK_TAKERS_{INFER}_TRAIN PURGE",
    f"""
    CREATE TABLE SA_TBL_AGG_PACK_TAKERS_{INFER}_TRAIN NOLOGGING COMPRESS FOR ARCHIVE HIGH AS
    SELECT A.*,
           B.SRVC_ENC SRVC1, C.SRVC_ENC SRVC2, D.SRVC_ENC SRVC3, E.SRVC_ENC SRVC4,
           F.SRVC_ENC SRVC5, G.SRVC_ENC SRVC6, H.SRVC_ENC SRVC7, I.SRVC_ENC SRVC8
    FROM SA_TBL_AGG_PACK_TAKERS_{INFER}_2 A
    LEFT JOIN (SELECT DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY='{DP.mapping_day_key}') B ON A.AMOUNT1=B.AMOUNT
    LEFT JOIN (SELECT DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY='{DP.mapping_day_key}') C ON A.AMOUNT2=C.AMOUNT
    LEFT JOIN (SELECT DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY='{DP.mapping_day_key}') D ON A.AMOUNT3=D.AMOUNT
    LEFT JOIN (SELECT DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY='{DP.mapping_day_key}') E ON A.AMOUNT4=E.AMOUNT
    LEFT JOIN (SELECT DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY='{DP.mapping_day_key}') F ON A.AMOUNT5=F.AMOUNT
    LEFT JOIN (SELECT DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY='{DP.mapping_day_key}') G ON A.AMOUNT6=G.AMOUNT
    LEFT JOIN (SELECT DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY='{DP.mapping_day_key}') H ON A.AMOUNT7=H.AMOUNT
    LEFT JOIN (SELECT DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY='{DP.mapping_day_key}') I ON A.AMOUNT8=I.AMOUNT
    """,
       "DROP TABLE SA_TBL_ATL_BASE_INFER_01 PURGE",
    f"""
    CREATE TABLE SA_TBL_ATL_BASE_INFER_01
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.MSISDN,
           TOTAL_DSTR, VOICEREV_TOTAL, DATAREV_TOTAL, MIXED_BUNDLE_REV,
           VOICEREV_TRIG, DATAREV_TRIG,
           VOL_KB/1024 AS VOL_MB,
           MO_MOU, RECHARGE_CNT, RECHARGE_AMOUNT, RECHARGE_MAX,
           DATA_RG_DAYS, VOICE_RG_DAYS,
           A.CIRCLE, A.REGION, A.AREA,
           AMOUNT1, AMOUNT2, AMOUNT3, AMOUNT4,
           AMOUNT5, AMOUNT6, AMOUNT7, AMOUNT8,
           SRVC1, SRVC2, SRVC3, SRVC4,
           SRVC5, SRVC6, SRVC7, SRVC8,
           CASE
               WHEN HS_TYPE IN ('SMARTPHONE','SMARTPHONE 4G','SMARTPHONE 5G')
               THEN 1 ELSE 0
           END AS SMARTPHONE,
           SUBS_ALL_90D,
           NVL((MONTH_KEY - RCHG_LAST_DT),30) AS DAYS_SINCE_LAST_RCRG,
           RCHG_CHNL, MYGP_M1_ACT, MYGP_M2_ACT,
           MYGP_ACTIVE_DAYS, MYGP_PACK_REV, MYGP_PACK_HITS,
           RCHG_AMT_MAX_MYGP, RCHG_AMT_MYGP, RCHG_HIT_MYGP,
           RCHG_AMT_MAX_DIGITAL, RCHG_AMT_DIGTAL, RCHG_CNT_DIGTAL,
           CASE
               WHEN B.MSISDN IS NOT NULL THEN 'TAKER'
               ELSE 'NON_TAKER'
           END AS PACK_FLAG
    FROM gpbi_marketing.gpbi_mstr_detail_info PARTITION({DP.part_t}) A
    LEFT JOIN SA_TBL_AGG_PACK_TAKERS_{INFER}_TRAIN B
           ON A.MSISDN = B.MSISDN
    WHERE A.JOINING_DATE < '{DP.joining_date_cutoff}'
      AND SUBS_ALL_90D = 1
    """,

    # =========================
    # Inference Base – T-1
    # =========================
    "DROP TABLE SA_TBL_ATL_BASE_INFER_02 PURGE",
    f"""
    CREATE TABLE SA_TBL_ATL_BASE_INFER_02
    COMPRESS FOR ARCHIVE HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.MSISDN AS MSISDN_1,
           TOTAL_DSTR AS TOTAL_DSTR_1,
           VOL_KB/1024 AS VOL_MB_1,
           MO_MOU AS MO_MOU_1,
           RECHARGE_CNT AS RECHARGE_CNT_1,
           RECHARGE_MAX AS RECHARGE_MAX_1,
           VOICEREV_TOTAL AS VOICEREV_TOTAL_1,
           DATAREV_TOTAL AS DATAREV_TOTAL_1,
           MIXED_BUNDLE_REV AS MIXED_BUNDLE_REV_1,
           VOICEREV_TRIG AS VOICEREV_TRIG_1,
           DATAREV_TRIG AS DATAREV_TRIG_1
    FROM gpbi_marketing.gpbi_mstr_detail_info PARTITION({DP.part_t_1}) A
    INNER JOIN (
        SELECT DISTINCT MSISDN FROM SA_TBL_ATL_BASE_INFER_01
    ) B
      ON A.MSISDN = B.MSISDN
    WHERE A.JOINING_DATE < '{DP.joining_date_cutoff}'
    GROUP BY A.MSISDN, TOTAL_DSTR, VOL_KB, MO_MOU,
             RECHARGE_CNT, RECHARGE_MAX,
             VOICEREV_TOTAL, DATAREV_TOTAL,
             MIXED_BUNDLE_REV, VOICEREV_TRIG, DATAREV_TRIG
    """,

    # =========================
    # Inference Base – T-2
    # =========================
    "DROP TABLE SA_TBL_ATL_BASE_INFER_03 PURGE",
    f"""
    CREATE TABLE SA_TBL_ATL_BASE_INFER_03
    COMPRESS FOR QUERY HIGH AS
    SELECT /*+ PARALLEL(8) */
           A.MSISDN AS MSISDN_2,
           TOTAL_DSTR AS TOTAL_DSTR_2,
           VOL_KB/1024 AS VOL_MB_2,
           MO_MOU AS MO_MOU_2,
           RECHARGE_CNT AS RECHARGE_CNT_2,
           RECHARGE_MAX AS RECHARGE_MAX_2,
           VOICEREV_TOTAL AS VOICEREV_TOTAL_2,
           DATAREV_TOTAL AS DATAREV_TOTAL_2,
           MIXED_BUNDLE_REV AS MIXED_BUNDLE_REV_2,
           VOICEREV_TRIG AS VOICEREV_TRIG_2,
           DATAREV_TRIG AS DATAREV_TRIG_2
    FROM gpbi_marketing.gpbi_mstr_detail_info PARTITION({DP.part_t_2}) A
    INNER JOIN (
        SELECT MSISDN FROM SA_TBL_ATL_BASE_INFER_01
    ) B
      ON A.MSISDN = B.MSISDN
    WHERE A.JOINING_DATE < '{DP.joining_date_cutoff}'
    GROUP BY A.MSISDN, TOTAL_DSTR, VOL_KB, MO_MOU,
             RECHARGE_CNT, RECHARGE_MAX,
             VOICEREV_TOTAL, DATAREV_TOTAL,
             MIXED_BUNDLE_REV, VOICEREV_TRIG, DATAREV_TRIG
    """,
]


# =========================
# Execution Utilities
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

    if not re.fullmatch(r"\d{6}", INFER):
        print(f"Invalid infer_month_key={INFER}", file=sys.stderr)
        return 3

    with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
        with conn.cursor() as cur:
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
                        print("  (ignored) ORA-00942")
                        conn.commit()
                        continue
                    raise

    print("Inference SQL execution completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
