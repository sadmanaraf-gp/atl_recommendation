import warnings
warnings.filterwarnings(
    "ignore",
    message=".*np.object.*FutureWarning.*",
    category=FutureWarning,
)
import pandas as pd
import numpy as np
import pickle
import gzip
import os
from dotenv import load_dotenv
from scripts.model import build_model
from scripts.config import *
from scripts.data_processing import load_and_process_data
from scripts.db_utils import export_to_oracle 
import oracledb



load_dotenv()

def run_inference(model, scaler, data, features, batch_size: int = 20000000):
    """Scales data and runs model prediction in batches."""
    data_scaled = scaler.transform(data[features])

    start_pos = 0
    max_size = len(data)
    prediction_batches = []

    print('Prediction generation started...')
    while start_pos < max_size:
        print(f'Generating from {start_pos} to {start_pos + batch_size}')
        batch_data = data_scaled[start_pos:start_pos + batch_size]
        pred_batch = model.predict(batch_data)
        prediction_batches.append(pred_batch)
        start_pos += batch_size

    predictions = np.concatenate(prediction_batches, axis=0)

    for index, name in enumerate(class_names):
        data[name] = predictions[:, index]

    return data


def main():
    """Main function to run the inference pipeline."""
    db_config = {
        "user": os.getenv("ORACLE_USER"),
        "password": os.getenv("ORACLE_PASSWORD"),
        "dsn": os.getenv("ORACLE_DSN")
    }

    # 1. Load inference data
    try:
        with gzip.open('data/processed_base_infer.gz', 'rb') as f:
            base_infer = pickle.load(f)
        print("Loaded inference data from cache.")
    except FileNotFoundError:
        print("Inference data cache not found. Loading from source...")
        base_infer = load_and_process_data(db_config, infer=True)
        if base_infer is None:
            return
        os.makedirs('data', exist_ok=True)
        with gzip.open('data/processed_base_infer.gz', 'wb') as f:
            pickle.dump(base_infer, f, protocol=4)
        print("Saved processed inference data to data/processed_base_infer.gz")

    # 2. Load models and scalers
    print("Loading models and scalers...")
    model_taker = build_model(input_shape=len(TAKER_FEATURES))
    model_taker.load_weights('artifacts/atl_reco_taker.h5')

    model_non_taker = build_model(input_shape=len(NONTAKER_FEATURES))
    model_non_taker.load_weights('artifacts/atl_reco_non_taker.h5')

    with open('artifacts/atl_scaler_taker.pkl', 'rb') as f:
        sc_taker = pickle.load(f)
    with open('artifacts/atl_scaler_non_taker.pkl', 'rb') as f:
        sc_non_taker = pickle.load(f)

    # 3. Check if base predictions are already stored
    base_pred_path = 'data/base_pred.gz'
    if os.path.exists(base_pred_path):
        print(f"Base predictions already exist at {base_pred_path}. Skipping prediction flow.")
        with gzip.open(base_pred_path, 'rb') as f:
            base_pred = pickle.load(f)
    else:
        print("Base predictions not found. Running prediction flow...")
        # Split base data

        taker_pred_path = 'data/taker_predictions.gz'
        nontaker_pred_path = 'data/nontaker_predictions.gz'

        taker_predictions = None
        nontaker_predictions = None

        # Check if taker predictions exist
        if os.path.exists(taker_pred_path):
            print(f"Taker predictions already exist at {taker_pred_path}. Skipping taker prediction flow.")
            with gzip.open(taker_pred_path, 'rb') as f:
                taker_predictions = pickle.load(f)
        else:
            print("Taker predictions not found. Running taker prediction flow...")
            taker_infer_df = base_infer[base_infer['pack_flag'] == 'TAKER'].copy()
            taker_predictions = run_inference(model_taker, sc_taker, taker_infer_df, TAKER_FEATURES)
            os.makedirs('data', exist_ok=True)
            with gzip.open(taker_pred_path, 'wb') as f:
                pickle.dump(taker_predictions, f, protocol=4)
                print(f"Taker predictions saved to '{taker_pred_path}'.")

        # Check if non-taker predictions exist
        if os.path.exists(nontaker_pred_path):
            print(f"Non-taker predictions already exist at {nontaker_pred_path}. Skipping non-taker prediction flow.")
            with gzip.open(nontaker_pred_path, 'rb') as f:
                nontaker_predictions = pickle.load(f)
        else:
            print("Non-taker predictions not found. Running non-taker prediction flow...")
            nontaker_infer_df = base_infer[base_infer['pack_flag'] == 'NON_TAKER'].copy()
            nontaker_predictions = run_inference(model_non_taker, sc_non_taker, nontaker_infer_df, NONTAKER_FEATURES)
            os.makedirs('data', exist_ok=True)
            with gzip.open(nontaker_pred_path, 'wb') as f:
                pickle.dump(nontaker_predictions, f, protocol=4)
                print(f"Non-taker predictions saved to '{nontaker_pred_path}'.")

        print("Inference complete. Predictions available in 'data/'.")


        taker_predictions[class_names] = taker_predictions[class_names].astype('float32')
        nontaker_predictions[class_names] = nontaker_predictions[class_names].astype('float32')

        mpb_taker = taker_predictions.melt(id_vars = 'msisdn', value_vars = class_names, var_name = 'deno', value_name = 'prob')
        mpb_non_taker = nontaker_predictions.melt(id_vars = 'msisdn', value_vars = class_names, var_name = 'deno', value_name = 'prob')

        mpb_taker = mpb_taker[mpb_taker['prob']>TAKER_PROB_CUTOFF]
        mpb_non_taker = mpb_non_taker[mpb_non_taker['prob']>NONTAKER_PROB_CUTOFF]

        mpb_taker['rank'] = (mpb_taker.groupby('msisdn')['prob'].rank(method="first",ascending = False))
        mpb_taker = mpb_taker.astype({"rank": int})

        mpb_non_taker['rank'] = (mpb_non_taker.groupby('msisdn')['prob'].rank(method="first",ascending = False))
        mpb_non_taker = mpb_non_taker.astype({"rank": int})

        taker_base_path = 'data/base_taker_pred.gz'
        nontaker_base_path = 'data/base_nontaker_pred.gz'

        # Load if exists, otherwise compute and save
        if os.path.exists(taker_base_path):
            print(f"Base taker pred already exist at {taker_base_path}. Loading...")
            with gzip.open(taker_base_path, 'rb') as f:
                base_taker_pred = pickle.load(f)
        else:
            print("Base taker pred not found. Computing...")
            base_taker_pred = mpb_taker[mpb_taker['rank'] <= PACK_NO]
            base_taker_pred['taker_flag'] = 1       
            os.makedirs('data', exist_ok=True)
            with gzip.open(taker_base_path, 'wb', compresslevel=4) as f:
                pickle.dump(base_taker_pred, f, protocol=4)
                print(f"Saved base taker pred to {taker_base_path}.")

        if os.path.exists(nontaker_base_path):
            print(f"Base non-taker pred already exist at {nontaker_base_path}. Loading...")
            with gzip.open(nontaker_base_path, 'rb') as f:
                base_nontaker_pred = pickle.load(f)
        else:
            print("Base non-taker pred not found. Computing...")
            base_nontaker_pred = mpb_non_taker[mpb_non_taker['rank'] <= 5]
            base_nontaker_pred['taker_flag'] = 0
            os.makedirs('data', exist_ok=True)
            with gzip.open(nontaker_base_path, 'wb', compresslevel=4) as f:
                pickle.dump(base_nontaker_pred, f, protocol=4)
                print(f"Saved base non-taker pred to {nontaker_base_path}.")



        # Combine predictions and save base_pred (load if exists, otherwise compute & save)
        if os.path.exists(base_pred_path):
            print(f"Base predictions already exist at {base_pred_path}. Loading...")
            with gzip.open(base_pred_path, 'rb') as f:
                base_pred = pickle.load(f)
        else:
            print("Base predictions not found. Combining and saving...")
            base_pred = pd.concat([base_taker_pred, base_nontaker_pred], ignore_index=True)
            base_pred = base_pred.astype({"msisdn": int, "deno": int})

            os.makedirs('data', exist_ok=True)
            with gzip.open(base_pred_path, 'wb', compresslevel=4) as f:
                pickle.dump(base_pred, f, protocol=4)

    # Use the shared utility to export to Oracle
    export_to_oracle(
        df=base_pred,
        table_name=FINAL_TABLE,
        user=db_config["user"],
        password=db_config["password"],
        dsn=db_config["dsn"],
        batch_size=EXPORT_BATCH_SIZE,
    )


if __name__ == '__main__':
    main()

