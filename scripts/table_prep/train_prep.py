from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import oracledb
import datetime
from dateutil.relativedelta import relativedelta

load_dotenv()
@dataclass(frozen=True)
class DateParams:    
# ...existing code...
    _ref_date = datetime.date(2026, 1, 15) # Fixed for Jan Run

    _label_date = _ref_date.replace(day=1) - relativedelta(months=1)  # Dec 2025
    _train_date = _label_date - relativedelta(months=1) # Nov 2025
    
    # Formats
    _fmt_mon = "%Y%m"
    _fmt_oracle = "%d-%b-%y"
    
    mapping_day_key: str = (_label_date + relativedelta(months=1, day=31)).strftime(_fmt_oracle).upper()

    # Train window key (Nov 2025)
    train_month_key: str = _train_date.strftime(_fmt_mon)
    
    # Train Window Definition: Sep 01 to Nov 30
    _train_end = _train_date + relativedelta(day=31) # Nov 30
    _train_start = (_train_end - relativedelta(months=2)).replace(day=1) # Sep 01
    
    # Load dates usually have a buffer (e.g. -7 days to +5 days around the month window)
    train_load_dt_start: str = (_train_start - relativedelta(days=7)).strftime(_fmt_oracle).upper()
    train_load_dt_end: str = (_train_end + relativedelta(days=5)).strftime(_fmt_oracle).upper()
    
    train_day_key_start: str = _train_start.strftime(_fmt_oracle).upper()
    train_day_key_end: str = _train_end.strftime(_fmt_oracle).upper()

    train_rc_load_dt_start: str = train_day_key_start
    train_rc_load_dt_end: str = train_day_key_end

    # Label window (Month T): Dec 01 to Dec 31
    label_month_key: str = _label_date.strftime(_fmt_mon)
    
    _label_start = _label_date.replace(day=1)
    _label_end = _label_date + relativedelta(day=31)

    label_load_dt_start: str = (_label_start - relativedelta(days=5)).strftime(_fmt_oracle).upper()
    label_load_dt_end: str = (_label_end + relativedelta(days=5)).strftime(_fmt_oracle).upper()
    
    label_day_key_start: str = _label_start.strftime(_fmt_oracle).upper()
    label_day_key_end: str = _label_end.strftime(_fmt_oracle).upper()

    label_rc_load_dt_start: str = label_day_key_start
    label_rc_load_dt_end: str = label_day_key_end

    # Base filters / partitions
    # Joining date is usually start of training (Sep 01)
    joining_date_cutoff: str = train_day_key_start
    
    # Partitions 
    # Label Month End (Nov 30)
    part_a_01: str = f"PART_{_train_end.strftime('%d%m%Y')}"
    # Train Month End (Oct 31) - Snapshot T
    _prev_train_end = _train_end.replace(day=1) - relativedelta(days=1)
    part_a_02: str = f"PART_{_prev_train_end.strftime('%d%m%Y')}"
    
    # Train Month - 1 (Sep 30) - Snapshot T-1
    _prev_train_end = _prev_train_end.replace(day=1) - relativedelta(days=1)
    part_a_03: str = f"PART_{_prev_train_end.strftime('%d%m%Y')}"


DP = DateParams()

print("=" * 60)
print(f"DATE PARAMETERS LOG (Ref: {DateParams._ref_date})")
print("-" * 60)
for key, value in vars(DP).items():
    print(f"{key:<25} : {value}")
print("=" * 60)



TRAIN = DP.train_month_key
LABEL = DP.label_month_key

STATEMENTS: List[str] = [
    # 1
    f"DROP TABLE SA_TBL_AGG_RENTAL_PACK_TAKERS_{TRAIN} PURGE",
    # 2
    f"""
    CREATE TABLE SA_TBL_AGG_RENTAL_PACK_TAKERS_{TRAIN}  AS
    SELECT /*+ PARALLEL(8) */  B.MSISDN,A.PACK,A.AMOUNT, SUM(NVL(REV,0)) as PACK_REV, SUM(NVL(HIT,0)) as HIT
    FROM GPBI_MARKETING.GPBI_SUBSCRIPTION_REV_DAILY B
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  * FROM TBL_ATL_MAPPING WHERE DAY_KEY = '{DP.mapping_day_key}') A
      ON A.PACK = B.PACK
    WHERE LOAD_DT BETWEEN '{DP.train_load_dt_start}' and '{DP.train_load_dt_end}'
      and B.DAY_KEY BETWEEN '{DP.train_day_key_start}' and '{DP.train_day_key_end}'
      AND A.PACK IS NOT NULL
    GROUP BY B.MSISDN,A.PACK,A.AMOUNT
    """,
    # 3
    f"DROP TABLE SA_TBL_AGG_RC_TAKER_{TRAIN} PURGE",
    # 4
    f"""
    CREATE TABLE SA_TBL_AGG_RC_TAKER_{TRAIN}  AS
    SELECT /*+ PARALLEL(8) */  A.PYMT_RSLT_CD MSISDN,
           B.AMOUNT, COUNT(*) AS HIT
    FROM OCDM_SYS.DWB_PRPD_RCHRG A
    INNER JOIN (SELECT /*+ PARALLEL(8) */  * FROM TBL_ATL_MAPPING WHERE DAY_KEY = '{DP.mapping_day_key}') B
      ON ROUND(A.PYMT_AMT) = B.PACK
    WHERE A.LOAD_DT BETWEEN '{DP.train_rc_load_dt_start}' and '{DP.train_rc_load_dt_end}'
      AND UPPER(B.SRVC) = 'RC'
    GROUP BY PYMT_RSLT_CD, B.AMOUNT
    """,
    # 5
    f"DROP TABLE SA_TBL_AGG_PACK_TAKERS_{TRAIN} PURGE",
    # 6
    f"""
    CREATE TABLE SA_TBL_AGG_PACK_TAKERS_{TRAIN}  AS
    SELECT /*+ PARALLEL(8) */  *
    FROM (
        (SELECT /*+ PARALLEL(8) */  MSISDN, AMOUNT, SUM(HIT) HIT
         FROM SA_TBL_AGG_RENTAL_PACK_TAKERS_{TRAIN}
         GROUP BY MSISDN, AMOUNT)
        UNION
        (SELECT /*+ PARALLEL(8) */  MSISDN, AMOUNT, HIT
         FROM SA_TBL_AGG_RC_TAKER_{TRAIN})
    )
    """,
    # 7
    f"DROP TABLE SA_TBL_AGG_PACK_TAKERS_{TRAIN}_1 PURGE",
    # 8
    f"""
    CREATE TABLE SA_TBL_AGG_PACK_TAKERS_{TRAIN}_1 NOLOGGING  AS
    SELECT /*+ PARALLEL(8) */  A.*,
           ROW_NUMBER() OVER(PARTITION BY MSISDN ORDER BY HIT DESC, AMOUNT DESC) as HIT_RANK,
           ROW_NUMBER() OVER(PARTITION BY MSISDN ORDER BY HIT ASC, AMOUNT DESC) as HIT_RANK_MIN
    FROM SA_TBL_AGG_PACK_TAKERS_{TRAIN} A
    """,
    # 9
    f"DROP TABLE SA_TBL_AGG_PACK_TAKERS_{TRAIN}_2 PURGE",
    # 10
    f"""
    CREATE TABLE SA_TBL_AGG_PACK_TAKERS_{TRAIN}_2 NOLOGGING  AS
    SELECT /*+ PARALLEL(8) */  A.MSISDN,
           MAX(CASE WHEN AMOUNT1 = 1 THEN AMOUNT_MIN ELSE AMOUNT1 END) AMOUNT1,
           MAX(CASE WHEN AMOUNT2 = 1 THEN AMOUNT_MIN ELSE AMOUNT2 END) AMOUNT2,
           MAX(CASE WHEN AMOUNT3 = 1 THEN AMOUNT_MIN ELSE AMOUNT3 END) AMOUNT3,
           MAX(CASE WHEN AMOUNT4 = 1 THEN AMOUNT_MIN ELSE AMOUNT4 END) AMOUNT4,
           MAX(CASE WHEN AMOUNT5 = 1 THEN AMOUNT_MIN ELSE AMOUNT5 END) AMOUNT5,
           MAX(CASE WHEN AMOUNT6 = 1 THEN AMOUNT_MIN ELSE AMOUNT6 END) AMOUNT6,
           MAX(CASE WHEN AMOUNT7 = 1 THEN AMOUNT_MIN ELSE AMOUNT7 END) AMOUNT7,
           MAX(CASE WHEN AMOUNT8 = 1 THEN AMOUNT_MIN ELSE AMOUNT8 END) AMOUNT8
    FROM (
        SELECT /*+ PARALLEL(8) */  MSISDN,
               NVL(MAX(CASE WHEN HIT_RANK = 1 THEN NVL(AMOUNT,0) END),1) AMOUNT1,
               NVL(MAX(CASE WHEN HIT_RANK = 2 THEN NVL(AMOUNT,0) END),1) AMOUNT2,
               NVL(MAX(CASE WHEN HIT_RANK = 3 THEN NVL(AMOUNT,0) END),1) AMOUNT3,
               NVL(MAX(CASE WHEN HIT_RANK = 4 THEN NVL(AMOUNT,0) END),1) AMOUNT4,
               NVL(MAX(CASE WHEN HIT_RANK = 5 THEN NVL(AMOUNT,0) END),1) AMOUNT5,
               NVL(MAX(CASE WHEN HIT_RANK = 6 THEN NVL(AMOUNT,0) END),1) AMOUNT6,
               NVL(MAX(CASE WHEN HIT_RANK = 7 THEN NVL(AMOUNT,0) END),1) AMOUNT7,
               NVL(MAX(CASE WHEN HIT_RANK = 8 THEN NVL(AMOUNT,0) END),1) AMOUNT8,
               NVL(MAX(CASE WHEN HIT_RANK_MIN = 1 THEN NVL(AMOUNT,0) END),1) AMOUNT_MIN
        FROM SA_TBL_AGG_PACK_TAKERS_{TRAIN}_1
        GROUP BY MSISDN
    ) A
    GROUP BY A.MSISDN
    """,
    # 11
    f"DROP TABLE SA_TBL_AGG_PACK_TAKERS_{TRAIN}_TRAIN PURGE",
    # 12
    f"""
    CREATE TABLE SA_TBL_AGG_PACK_TAKERS_{TRAIN}_TRAIN NOLOGGING  AS
    SELECT /*+ PARALLEL(8) */  A.*, B.SRVC_ENC SRVC1, C.SRVC_ENC SRVC2,
           D.SRVC_ENC SRVC3, E.SRVC_ENC SRVC4, F.SRVC_ENC SRVC5,
           G.SRVC_ENC SRVC6, H.SRVC_ENC SRVC7, I.SRVC_ENC SRVC8
    FROM SA_TBL_AGG_PACK_TAKERS_{TRAIN}_2 A
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY = ('{DP.mapping_day_key}')) B
      ON A.AMOUNT1 = B.AMOUNT
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY = ('{DP.mapping_day_key}')) C
      ON A.AMOUNT2 = C.AMOUNT
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY = ('{DP.mapping_day_key}')) D
      ON A.AMOUNT3 = D.AMOUNT
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY = ('{DP.mapping_day_key}')) E
      ON A.AMOUNT4 = E.AMOUNT
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY = ('{DP.mapping_day_key}')) F
      ON A.AMOUNT5 = F.AMOUNT
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY = ('{DP.mapping_day_key}')) G
      ON A.AMOUNT6 = G.AMOUNT
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY = ('{DP.mapping_day_key}')) H
      ON A.AMOUNT7 = H.AMOUNT
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  DISTINCT AMOUNT, SRVC_ENC FROM TBL_ATL_MAPPING WHERE DAY_KEY = ('{DP.mapping_day_key}')) I
      ON A.AMOUNT8 = I.AMOUNT
    """,
    # 13
    f"DROP TABLE SA_TBL_RENTAL_PACK_TAKERS_{LABEL} PURGE",
    # 14
    f"""
    CREATE TABLE SA_TBL_RENTAL_PACK_TAKERS_{LABEL}  AS
    SELECT /*+ PARALLEL(8) */   B.MSISDN,A.PACK, A.AMOUNT, A.SRVC, SUM(NVL(REV,0)) as PACK_REV, SUM(NVL(HIT,0)) as HIT
    FROM GPBI_MARKETING.GPBI_SUBSCRIPTION_REV_DAILY B
    LEFT JOIN (SELECT /*+ PARALLEL(8) */  * FROM TBL_ATL_MAPPING WHERE DAY_KEY = '{DP.mapping_day_key}') A
      ON A.PACK = B.PACK
    WHERE LOAD_DT BETWEEN '{DP.label_load_dt_start}' and '{DP.label_load_dt_end}'
      and B.DAY_KEY BETWEEN '{DP.label_day_key_start}' and '{DP.label_day_key_end}'
      AND A.PACK IS NOT NULL
    GROUP BY B.MSISDN,A.PACK,A.AMOUNT, A.SRVC
    """,
    # 15
    f"DROP TABLE SA_TBL_RC_TAKER_{LABEL} PURGE",
    # 16
    f"""
    CREATE TABLE SA_TBL_RC_TAKER_{LABEL}  AS
    SELECT /*+ PARALLEL(8) */  A.PYMT_RSLT_CD MSISDN, B.AMOUNT, 'RC' AS SRVC, COUNT(*) AS HIT
    FROM OCDM_SYS.DWB_PRPD_RCHRG A
    INNER JOIN (SELECT /*+ PARALLEL(8) */  * FROM TBL_ATL_MAPPING WHERE DAY_KEY = '{DP.mapping_day_key}' AND SRVC = 'RC') B
      ON ROUND(A.PYMT_AMT) = B.PACK
    WHERE A.LOAD_DT BETWEEN '{DP.label_rc_load_dt_start}' and '{DP.label_rc_load_dt_end}'
    GROUP BY PYMT_RSLT_CD, B.AMOUNT, 'RC'
    """,
    # 17
    f"DROP TABLE SA_TBL_PACK_TAKERS_{LABEL} PURGE",
    # 18
    f"""
    CREATE TABLE SA_TBL_PACK_TAKERS_{LABEL} COMPRESS FOR ARCHIVE AS
    SELECT /*+ PARALLEL(8) */  *
    FROM (
        (SELECT /*+ PARALLEL(8) */  MSISDN, AMOUNT, SRVC, SUM(HIT) HIT
         FROM SA_TBL_RENTAL_PACK_TAKERS_{LABEL}
         GROUP BY MSISDN, AMOUNT, SRVC)
        UNION
        (SELECT /*+ PARALLEL(8) */  MSISDN, AMOUNT, SRVC, HIT
         FROM SA_TBL_RC_TAKER_{LABEL})
    )
    """,
    # 19
    "DROP TABLE SA_TBL_ATL_BASE_01_TEST PURGE",
    # 20
    f"""
    CREATE TABLE SA_TBL_ATL_BASE_01_TEST  AS
    SELECT /*+ PARALLEL(8) */  TOTAL_DSTR, VOICEREV_TOTAL, DATAREV_TOTAL, MIXED_BUNDLE_REV, VOICEREV_TRIG, DATAREV_TRIG,
           VOL_KB/1024 as VOL_MB, MO_MOU, RECHARGE_CNT, RECHARGE_AMOUNT, RECHARGE_MAX, DATA_RG_DAYS, VOICE_RG_DAYS,
           A.CIRCLE, A.REGION, A.AREA, B.*, C.AMOUNT, C.HIT, C.SRVC,
           CASE
               WHEN HS_TYPE IN ('SMARTPHONE','SMARTPHONE 4G','SMARTPHONE 5G') THEN 1
               ELSE 0
           END AS SMARTPHONE,
           SUBS_ALL_90D, NVL((MONTH_KEY - RCHG_LAST_DT),30) as DAYS_SINCE_LAST_RCRG, RCHG_CHNL, MYGP_M1_ACT, MYGP_M2_ACT,
           MYGP_ACTIVE_DAYS, MYGP_PACK_REV, MYGP_PACK_HITS, RCHG_AMT_MAX_MYGP, RCHG_AMT_MYGP, RCHG_HIT_MYGP,
           RCHG_AMT_MAX_DIGITAL, RCHG_AMT_DIGTAL, RCHG_CNT_DIGTAL
    FROM gpbi_marketing.gpbi_mstr_detail_info PARTITION({DP.part_a_01}) A
    INNER JOIN SA_TBL_AGG_PACK_TAKERS_{TRAIN}_TRAIN B
      ON A.MSISDN = B.MSISDN
    INNER JOIN (
        SELECT /*+ PARALLEL(8) */  MSISDN, AMOUNT, HIT, SRVC
        FROM SA_TBL_PACK_TAKERS_{LABEL}
    ) C
      ON A.MSISDN = C.MSISDN
    WHERE A.JOINING_DATE < '{DP.joining_date_cutoff}'
    """,
    # 22
    "DROP TABLE SA_TBL_ATL_BASE_02_TEST PURGE",
    # 23
    f"""
    CREATE TABLE SA_TBL_ATL_BASE_02_TEST  AS
    SELECT /*+ PARALLEL(8) */  A.MSISDN as MSISDN_1, TOTAL_DSTR as TOTAL_DSTR_1, VOL_KB/1024 as VOL_MB_1, MO_MOU as MO_MOU_1,
           RECHARGE_CNT as RECHARGE_CNT_1, RECHARGE_MAX AS RECHARGE_MAX_1, VOICEREV_TOTAL as VOICEREV_TOTAL_1,
           DATAREV_TOTAL as DATAREV_TOTAL_1, MIXED_BUNDLE_REV as MIXED_BUNDLE_REV_1, VOICEREV_TRIG as VOICEREV_TRIG_1,
           DATAREV_TRIG as DATAREV_TRIG_1
    FROM gpbi_marketing.gpbi_mstr_detail_info PARTITION({DP.part_a_02}) A
    INNER JOIN (SELECT  DISTINCT MSISDN FROM SA_TBL_ATL_BASE_01_TEST) B
      ON A.MSISDN = B.MSISDN
    WHERE A.JOINING_DATE < '{DP.joining_date_cutoff}'
    GROUP BY A.MSISDN, TOTAL_DSTR, VOL_KB, MO_MOU, RECHARGE_CNT, RECHARGE_MAX,
             VOICEREV_TOTAL, DATAREV_TOTAL, MIXED_BUNDLE_REV, VOICEREV_TRIG, DATAREV_TRIG
    """,
    # 24
    "DROP TABLE SA_TBL_ATL_BASE_03_TEST PURGE",
    # 25
    f"""
    CREATE TABLE SA_TBL_ATL_BASE_03_TEST AS
    SELECT /*+ PARALLEL(8) */  A.MSISDN as MSISDN_2, TOTAL_DSTR as TOTAL_DSTR_2, VOL_KB/1024 as VOL_MB_2, MO_MOU as MO_MOU_2,
           RECHARGE_CNT as RECHARGE_CNT_2, RECHARGE_MAX AS RECHARGE_MAX_2, VOICEREV_TOTAL as VOICEREV_TOTAL_2,
           DATAREV_TOTAL as DATAREV_TOTAL_2, MIXED_BUNDLE_REV as MIXED_BUNDLE_REV_2, VOICEREV_TRIG as VOICEREV_TRIG_2,
           DATAREV_TRIG as DATAREV_TRIG_2
    FROM gpbi_marketing.gpbi_mstr_detail_info PARTITION({DP.part_a_03}) A
    INNER JOIN (SELECT  MSISDN FROM SA_TBL_ATL_BASE_01_TEST) B
      ON A.MSISDN = B.MSISDN
    WHERE A.JOINING_DATE < '{DP.joining_date_cutoff}'
    GROUP BY A.MSISDN, TOTAL_DSTR, VOL_KB, MO_MOU, RECHARGE_CNT, RECHARGE_MAX,
             VOICEREV_TOTAL, DATAREV_TOTAL, MIXED_BUNDLE_REV, VOICEREV_TRIG, DATAREV_TRIG
    """,
]

def table_exists(cursor, table_name):
    check_sql = "SELECT count(*) FROM user_tables WHERE table_name = :tn"
    cursor.execute(check_sql, tn=table_name.upper())
    (count,) = cursor.fetchone()
    return count > 0

def main() -> int:
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")


    if not user or not password or not dsn:
        print("Missing ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN env vars.", file=sys.stderr)
        return 2

    # Basic validation to avoid accidental bad object names.
    for vname, v in [("train_month_key", TRAIN), ("label_month_key", LABEL)]:
        if not re.fullmatch(r"\d{6}", v):
            print(f"Invalid {vname}={v!r}. Expected YYYYMM (6 digits).", file=sys.stderr)
            return 4


    if not STATEMENTS:
        print("No SQL statements found.", file=sys.stderr)
        return 3

    with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
        with conn.cursor() as cur:
            for i, stmt in enumerate(STATEMENTS, start=1):
                stmt = stmt.strip()
                if not stmt:
                    continue

                first_line = stmt.splitlines()[0].strip()

                # Simple parsing to check if it's a CREATE TABLE statement
                create_match = re.match(r"CREATE\s+TABLE\s+(\w+)", stmt, re.IGNORECASE)
                if create_match:
                    tbl_name = create_match.group(1)
                    if table_exists(cur, tbl_name):
                        print(f"[{i}/{len(STATEMENTS)}] Table {tbl_name} already exists, skipping creation.")
                        continue

                print(f"[{i}/{len(STATEMENTS)}] Executing: {first_line[:120]}...")

                try:
                    cur.execute(stmt)

                    if stmt.upper().startswith("SELECT /*+ PARALLEL(8) */ "):
                        rows = cur.fetchmany(20)
                        print(f"  Returned {len(rows)} rows (showing up to 20): {rows}")

                    conn.commit()

                except oracledb.Error as e:
                    msg = str(e)
                    if "ORA-00942" in msg and stmt.upper().startswith("DROP TABLE"):
                        print("  (ignored) ORA-00942: table or view does not exist")
                        conn.commit()
                        continue
                    print(f"  FAILED: {e}", file=sys.stderr)
                    raise

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())