import warnings
warnings.filterwarnings(
    "ignore",
    message=".*np.object.*FutureWarning.*",
    category=FutureWarning,
)
import pandas as pd
from sklearn import preprocessing
from scripts.db_utils import read_oracle_data
from scripts import config as cfg
import os
import gzip
import pickle

def preprocess_data(base_df, infer=False):
    """
    When infer=True, label encoding is skipped (no 'label' column created).
    """
    df = base_df.copy()
    df.columns = df.columns.str.lower()
    
    # One-hot encode
    df = pd.get_dummies(df, columns=['circle','region','rchg_chnl'])
    df.columns = df.columns.str.lower()

    # Outlier handling
    if not infer:
        # During training: remove outliers and save thresholds
        rchg_cap = df["recharge_amount"].quantile(0.98)
        dstr_cap = df["total_dstr"].quantile(0.99)
        # Save thresholds for inference clipping
        import json
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/outlier_thresholds.json', 'w') as f:
            json.dump({"recharge_amount_cap": float(rchg_cap), "total_dstr_cap": float(dstr_cap)}, f)
        df = df[df["recharge_amount"] <= rchg_cap]
        df = df[df["total_dstr"] <= dstr_cap]
    else:
        # During inference: clip to training thresholds (don't drop subscribers)
        import json
        try:
            with open('artifacts/outlier_thresholds.json', 'r') as f:
                thresholds = json.load(f)
            df["recharge_amount"] = df["recharge_amount"].clip(upper=thresholds["recharge_amount_cap"])
            df["total_dstr"] = df["total_dstr"].clip(upper=thresholds["total_dstr_cap"])
        except FileNotFoundError:
            print("WARNING: outlier_thresholds.json not found. Skipping outlier clipping at inference.")




    # Feature engineering
    column_diff_01 = ['dstr_change_01','mou_change_01','vol_change_01']
    column_diff_12 = ['dstr_change_12','mou_change_12','vol_change_12']
    column_0 = ['total_dstr','mo_mou','vol_mb']
    column_1 = ['total_dstr_1','mo_mou_1','vol_mb_1']
    column_2 = ['total_dstr_2','mo_mou_2','vol_mb_2']

    for index, name in enumerate(column_diff_01):
        df[name] = (df[column_0[index]] - df[column_1[index]]).fillna(df[column_0[index]])

    for index, name in enumerate(column_diff_12):
        df[name] = (df[column_1[index]] - df[column_2[index]]).fillna(
            df[column_1[index]].fillna(0)
        )
    
    # Fill NaNs
    if df.isnull().values.any():
        print("NaNs found in data. Filling with 0 to prevent propogation to loss.")
        df.fillna(0, inplace=True)

    label_le = None
    if not infer:
        # Label encoding only during training
        label_le = preprocessing.LabelEncoder()
        df['label'] = label_le.fit_transform(df['amount'])
        
        # Persist label encoder for inference consistency
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_le, f)
        print(f"LabelEncoder saved ({len(label_le.classes_)} classes): {list(label_le.classes_)}")
        
        # Rank in terms of pack rev to make distinct transaction
        df['pack_rank'] = df.groupby(['msisdn'])['hit'].rank(method="first", ascending=False)
        df = df.astype({"pack_rank": int})
        
    return df

def load_and_process_data(db_config, infer=False):
    """Loads data from Oracle and processes it."""
    print("Loading and merging data from Oracle...")
    if infer:
        table1, table2, table3 = cfg.TABLE_INFER_01, cfg.TABLE_INFER_02, cfg.TABLE_INFER_03
        base_path = 'data/base_infer.gz'
    else:
        table1, table2, table3 = cfg.TABLE_BASE_01, cfg.TABLE_BASE_02, cfg.TABLE_BASE_03
        base_path = 'data/base_train.gz'

    # If saved base is available, load from gz
    if os.path.exists(base_path):
        print(f"Loading base from {base_path}...")
        with gzip.open(base_path, 'rb') as f:
            base = pickle.load(f)
    else:
        # Otherwise, import from database and save
        base1 = read_oracle_data(f"SELECT * FROM {table1}", **db_config)
        base2 = read_oracle_data(f"SELECT * FROM {table2}", **db_config)
        base3 = read_oracle_data(f"SELECT * FROM {table3}", **db_config)
        
        if base1 is None or base2 is None or base3 is None:
            print("Failed to load one or more tables. Aborting.")
            return None

        base1['MSISDN'] = base1['MSISDN'].astype(int)
        base2['MSISDN_1'] = base2['MSISDN_1'].astype(int)
        base3['MSISDN_2'] = base3['MSISDN_2'].astype(int)
        
        base = base1.merge(base2, 'left', left_on='MSISDN', right_on='MSISDN_1') \
                    .merge(base3, how='left', left_on='MSISDN', right_on='MSISDN_2')
        
        print("Saving base...")
        with gzip.open(base_path, 'wb') as f:
            pickle.dump(base, f, protocol=4)
    
    print("Preprocessing data...")
    processed_df = preprocess_data(base, infer=infer)
    
    # Data quality checks
    if processed_df is None or len(processed_df) == 0:
        print("ERROR: Preprocessing returned empty DataFrame.")
        return None
    
    print(f"  Processed rows: {len(processed_df):,}")
    
    # Validate expected features exist
    expected_features = cfg.TAKER_FEATURES if not infer else cfg.NONTAKER_FEATURES
    missing_cols = [c for c in expected_features if c not in processed_df.columns]
    if missing_cols:
        print(f"  WARNING: Missing expected columns: {missing_cols}")
    
    # Label distribution summary (training only)
    if not infer and 'label' in processed_df.columns:
        label_counts = processed_df['label'].value_counts()
        print(f"  Label classes: {label_counts.nunique()}")
        print(f"  Most common class: label={label_counts.index[0]} ({label_counts.iloc[0]:,} samples, {label_counts.iloc[0]/len(processed_df)*100:.1f}%)")
        print(f"  Least common class: label={label_counts.index[-1]} ({label_counts.iloc[-1]:,} samples, {label_counts.iloc[-1]/len(processed_df)*100:.1f}%)")
    
    return processed_df