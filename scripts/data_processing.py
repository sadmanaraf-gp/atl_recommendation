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
    df = pd.get_dummies(df, columns=['circle','region','rchg_chnl','segment_9box'])
    df.columns = df.columns.str.lower()

    # Remove outliers
    df = df[df["recharge_amount"] <= df["recharge_amount"].quantile(0.98)]
    df = df[df["total_dstr"] <= df["total_dstr"].quantile(0.99)]



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

    label_le = None
    if not infer:
        # Label encoding only during training
        label_le = preprocessing.LabelEncoder()
        df['label'] = label_le.fit_transform(df['amount'])
        # Rank in terms of pack rev to make distinct transaction
        df['pack_rank'] = df.groupby(['msisdn'])['hit'].rank(method="first", ascending=False)
        df = df.astype({"pack_rank": int})

    # Fill NaNs
    for i in df.columns[df.isnull().any(axis=0)]:
        df[i].fillna(0, inplace=True)
        
    return df, label_le

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
            return None, None

        base1['MSISDN'] = base1['MSISDN'].astype(int)
        base2['MSISDN_1'] = base2['MSISDN_1'].astype(int)
        base3['MSISDN_2'] = base3['MSISDN_2'].astype(int)
        
        base = base1.merge(base2, 'left', left_on='MSISDN', right_on='MSISDN_1') \
                    .merge(base3, how='left', left_on='MSISDN', right_on='MSISDN_2')
        
        print("Saving base...")
        with gzip.open(base_path, 'wb') as f:
            pickle.dump(base, f, protocol=4)
    
    print("Preprocessing data...")
    processed_df, label_encoder = preprocess_data(base, infer=infer)
    
    return processed_df, label_encoder
