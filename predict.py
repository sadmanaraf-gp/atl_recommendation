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
from scripts.config import FEATURES, N_CLASSES
from scripts.data_processing import load_and_process_data
from scripts.db_utils import export_to_oracle  # <- use this
import oracledb



load_dotenv()

def run_inference(model, scaler, data, features, batch_size: int = 20000000):
    """Scales data and runs model prediction in batches."""
    data_scaled = scaler.transform(data[features])

    start_pos = 0
    batch_size = batch_size
    max_size = len(data)
    predictions = np.empty((0, N_CLASSES), int)

    print('Prediction generation started...')
    while start_pos < max_size:
        print(f'Generating from {start_pos} to {start_pos + batch_size}')
        batch_data = data_scaled[start_pos:start_pos + batch_size]
        pred_batch = model.predict(batch_data)
        predictions = np.vstack([predictions, pred_batch])
        start_pos += batch_size

    return predictions


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
        base_infer, _ = load_and_process_data(db_config, infer=True)
        if base_infer is None:
            return
        os.makedirs('data', exist_ok=True)
        with gzip.open('data/processed_base_infer.gz', 'wb') as f:
            pickle.dump(base_infer, f, protocol=4)
        print("Saved processed inference data to data/processed_base_infer.gz")

    # 2. Load models and scalers
    print("Loading models and scalers...")
    model_taker = build_model(input_shape=len(FEATURES))
    model_taker.load_weights('artifacts/pl_reco_taker.h5')

    model_non_taker = build_model(input_shape=len(FEATURES))
    model_non_taker.load_weights('artifacts/pl_reco_non_taker.h5')

    with open('artifacts/pl_scaler_taker.pkl', 'rb') as f:
        sc_taker = pickle.load(f)
    with open('artifacts/pl_scaler_non_taker.pkl', 'rb') as f:
        sc_non_taker = pickle.load(f)

    # 3. Run predictions
    taker_infer_df = base_infer[base_infer['pack_flag'] == 'TAKER'].copy()
    nontaker_infer_df = base_infer[base_infer['pack_flag'] == 'NON_TAKER'].copy()

    # Predict Takers
    taker_predictions = run_inference(model_taker, sc_taker, taker_infer_df, FEATURES)

    # Predict Non-Takers
    nontaker_predictions = run_inference(model_non_taker, sc_non_taker, nontaker_infer_df, FEATURES)

    # 4. Post-process and save results
    print("Saving prediction results...")
    with gzip.open('data/taker_predictions.gz', 'wb') as f:
        pickle.dump(taker_predictions, f, protocol=4)

    with gzip.open('data/nontaker_predictions.gz', 'wb') as f:
        pickle.dump(nontaker_predictions, f, protocol=4)

    print("Inference complete. Predictions saved to 'data/'.")

    base_pred = pd.concat([taker_predictions, nontaker_predictions])
    base_pred = base_pred.astype({"msisdn": int, "deno": int})

    with gzip.open('artifacts/base_pred.gz', 'wb', compresslevel=4) as f:
        pickle.dump(base_pred, f, protocol=4)

    # Use the shared utility to export to Oracle
    export_to_oracle(
        df=base_pred,
        table_name="TBL_PL_PRED_2025",
        db_config=db_config,
        batch_size=1559752,
    )


if __name__ == '__main__':
    main()
