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


# LabelEncoder.classes_ is sorted and matches model output order
with open('artifacts/label_encoder.pkl', 'rb') as f:
    _label_encoder = pickle.load(f)
class_names = list(_label_encoder.classes_)


def run_inference_top_k(model, scaler, data, features, prob_cutoff, top_k, taker_flag, batch_size=2_000_000):
    """Predict in chunks and extract top-K immediately.
    Never stores the full N×53 probability matrix — extracts top-K per batch and discards."""
    data_scaled = scaler.transform(data[features]).astype(np.float32)
    msisdns = data['msisdn'].values
    class_arr = np.array(class_names)
    k = min(top_k, len(class_names))

    results = []
    n = len(data_scaled)

    print(f'  Prediction started: {n:,} subscribers, top-{k}, cutoff={prob_cutoff}')
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        print(f'  Batch {start:,} – {end:,} / {n:,}')

        pred_batch = model.predict(data_scaled[start:end], verbose=0)
        batch_msisdns = msisdns[start:end]

        # Extract top-K immediately per batch
        top_k_idx = np.argpartition(pred_batch, -k, axis=1)[:, -k:]
        row_idx = np.arange(len(batch_msisdns))[:, None]
        top_k_probs = pred_batch[row_idx, top_k_idx]

        # Sort within top-K (descending)
        sort_order = np.argsort(-top_k_probs, axis=1)
        top_k_idx = np.take_along_axis(top_k_idx, sort_order, axis=1)
        top_k_probs = np.take_along_axis(top_k_probs, sort_order, axis=1)

        # Flatten
        n_batch = len(batch_msisdns)
        msisdn_rep = np.repeat(batch_msisdns, k)
        deno_flat = class_arr[top_k_idx.ravel()]
        prob_flat = top_k_probs.ravel()
        rank_flat = np.tile(np.arange(1, k + 1), n_batch)

        # Apply probability cutoff before appending
        mask = prob_flat > prob_cutoff
        if mask.any():
            results.append(pd.DataFrame({
                'msisdn': msisdn_rep[mask],
                'deno': deno_flat[mask],
                'prob': prob_flat[mask],
                'rank': rank_flat[mask],
            }))

        del pred_batch, top_k_probs, top_k_idx  # free memory immediately

    result = pd.concat(results, ignore_index=True)
    result['taker_flag'] = taker_flag
    print(f'  Done. {len(result):,} predictions after cutoff.')
    return result


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
        with gzip.open('data/processed_base_infer.gz', 'wb', compresslevel=1) as f:
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

    # 3. Run predictions (fused predict + top-K extraction)
    base_pred_path = 'data/base_pred.gz'
    if os.path.exists(base_pred_path):
        print(f"Base predictions found at {base_pred_path}. Loading...")
        with gzip.open(base_pred_path, 'rb') as f:
            base_pred = pickle.load(f)
    else:
        print("Running prediction flow...")

        taker_df = base_infer[base_infer['pack_flag'] == 'TAKER']
        nontaker_df = base_infer[base_infer['pack_flag'] == 'NON_TAKER']

        print(f"\nTaker inference ({len(taker_df):,} subscribers)...")
        base_taker_pred = run_inference_top_k(
            model_taker, sc_taker, taker_df, TAKER_FEATURES,
            TAKER_PROB_CUTOFF, PACK_NO, taker_flag=1
        )

        print(f"\nNon-Taker inference ({len(nontaker_df):,} subscribers)...")
        base_nontaker_pred = run_inference_top_k(
            model_non_taker, sc_non_taker, nontaker_df, NONTAKER_FEATURES,
            NONTAKER_PROB_CUTOFF, PACK_NO, taker_flag=0
        )

        base_pred = pd.concat([base_taker_pred, base_nontaker_pred], ignore_index=True)
        base_pred = base_pred.astype({"msisdn": int, "deno": int})

        print(f"\nTotal predictions: {len(base_pred):,}")
        with gzip.open(base_pred_path, 'wb', compresslevel=1) as f:
            pickle.dump(base_pred, f, protocol=4)
        print(f"Saved to {base_pred_path}")

    # 4. Export to Oracle
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