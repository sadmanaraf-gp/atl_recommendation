import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

import os
from dotenv import load_dotenv

load_dotenv()

from scripts.data_processing import load_and_process_data
from scripts.model import build_model
from scripts.config import FEATURES, TARGET, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*np.object.*FutureWarning.*",
    category=FutureWarning,
)


def main():
    """Main function to run the training pipeline."""
    db_config = {
        "user": os.getenv("ORACLE_USER"),
        "password": os.getenv("ORACLE_PASSWORD"),
        "dsn": os.getenv("ORACLE_DSN")
    }

    # 1. Load and process data
    # This step can be slow. For faster iteration, we save the processed file.
    try:
        base = pd.read_pickle('data/processed_base.pkl')
        print("Loaded processed data from cache.")
    except FileNotFoundError:
        print("Processed data cache not found. Loading from source...")
        base, _ = load_and_process_data(db_config, infer=False)
        if base is None:
            return
        base.to_pickle('data/processed_base.pkl')
        print("Saved processed data to data/processed_base.pkl")

    # 2. Prepare data for Taker model
    print("Preparing data for 'Taker' model...")
    base['label_one_hot'] = list(to_categorical(base['label']))
    
    # Sampling and splitting
    base_sample = base[base.pack_rank==1].sample(frac=0.7, random_state=111)
    train_ind, test_ind = train_test_split(base_sample.index, test_size=0.3, random_state=123)
    
    train_x = base_sample.loc[train_ind, FEATURES].astype(np.float32)
    train_y = base_sample.loc[train_ind, 'label_one_hot']
    
    # Scaling
    sc = StandardScaler().fit(train_x)
    train_x_scaled = sc.transform(train_x)
    
    # Save the scaler
    with open('artifacts/pl_scaler_taker.pkl', 'wb') as f:
        pickle.dump(sc, f)
    print("Taker scaler saved to artifacts/pl_scaler_taker.pkl")

    # 3. Train Taker model
    print("Training 'Taker' model...")
    model_taker = build_model(input_shape=len(FEATURES))
    model_taker.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpointer = ModelCheckpoint(
        filepath='artifacts/pl_reco_taker.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )

    train_y_tensor = tf.convert_to_tensor(np.array(train_y.values.tolist()), dtype=tf.float16)
    
    model_taker.fit(
        train_x_scaled,
        train_y_tensor,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpointer]
    )
    print("Taker model training complete. Best model saved to artifacts/pl_reco_taker.h5")

    # 4. Prepare and Train Non-Taker model (logic is similar)
    print("\nPreparing and training 'Non-Taker' model...")
    # The notebook re-samples the entire base for the non-taker model.
    base_sample_non_taker = base.sample(frac=1, random_state=111)
    train_ind_nt, _ = train_test_split(base_sample_non_taker.index, test_size=0.3, random_state=123)
    
    train_x_nt = base_sample_non_taker.loc[train_ind_nt, FEATURES].astype(np.float32)
    train_y_nt = base_sample_non_taker.loc[train_ind_nt, 'label_one_hot']

    sc_non_taker = StandardScaler().fit(train_x_nt)
    train_x_scaled_nt = sc_non_taker.transform(train_x_nt)

    with open('artifacts/pl_scaler_non_taker.pkl', 'wb') as f:
        pickle.dump(sc_non_taker, f)
    print("Non-Taker scaler saved to artifacts/pl_scaler_non_taker.pkl")

    model_non_taker = build_model(input_shape=len(FEATURES))
    model_non_taker.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpointer_nt = ModelCheckpoint(
        filepath='artifacts/pl_reco_non_taker.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    
    train_y_tensor_nt = tf.convert_to_tensor(np.array(train_y_nt.values.tolist()), dtype=tf.float16)

    model_non_taker.fit(
        train_x_scaled_nt,
        train_y_tensor_nt,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpointer_nt]
    )
    print("Non-Taker model training complete. Best model saved to artifacts/pl_reco_non_taker.h5")

if __name__ == '__main__':
    main()
