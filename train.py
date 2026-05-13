import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import gzip

import os
from dotenv import load_dotenv

load_dotenv()

from scripts.data_processing import load_and_process_data
from scripts.model import build_model, focal_loss
from scripts.evaluate import evaluate_model, save_evaluation
from scripts.config import (TAKER_FEATURES, NONTAKER_FEATURES, TARGET, EPOCHS, BATCH_SIZE,
                            VALIDATION_SPLIT, N_CLASSES, USE_FOCAL_LOSS, FOCAL_LOSS_GAMMA,
                            FOCAL_LOSS_ALPHA, LABEL_SMOOTHING, NONTAKER_HIT_QUANTILE)
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*np.object.*FutureWarning.*",
    category=FutureWarning,
)

# Select loss function based on config
if USE_FOCAL_LOSS:
    LOSS_FN = focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
    print(f"Using FOCAL LOSS (gamma={FOCAL_LOSS_GAMMA}, alpha={FOCAL_LOSS_ALPHA})")
else:
    LOSS_FN = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    print(f"Using CATEGORICAL CROSSENTROPY with label_smoothing={LABEL_SMOOTHING}")


def main():
    """Main function to run the training pipeline."""
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    db_config = {
        "user": os.getenv("ORACLE_USER"),
        "password": os.getenv("ORACLE_PASSWORD"),
        "dsn": os.getenv("ORACLE_DSN")
    }

    # 1. Load and process data
    # This step can be slow. For faster iteration, we save the processed file.
    try:
        with gzip.open('data/processed_base.gz', 'rb') as f:
            base = pickle.load(f)
        print("Loaded processed data")
    except FileNotFoundError:
        print("Processed data cache not found. Loading from source...")
        base = load_and_process_data(db_config, infer=False)
        if base is None:
            return

        with gzip.open('data/processed_base.gz', 'wb') as f:
            pickle.dump(base, f, protocol=4)
        print("Saved processed data to data/processed_base.gz")

    # 2. Prepare data for Taker model
    print("Preparing data for 'Taker' model...")
    
    # Validate N_CLASSES matches actual label count
    n_unique_labels = base['label'].nunique()
    assert n_unique_labels == N_CLASSES, (
        f"N_CLASSES mismatch: config says {N_CLASSES} but data has {n_unique_labels} unique labels. "
        f"Update config.py (N_CLASSES and class_names) to match current pack universe."
    )
    print(f"Label validation passed: {n_unique_labels} classes confirmed.")
    
    base['label_one_hot'] = list(to_categorical(base['label']))
    
    # Sampling and splitting
    base_sample = base[base.pack_rank==1].sample(frac=0.7, random_state=111)
    train_ind, test_ind = train_test_split(base_sample.index, test_size=0.3, random_state=123)
    
    train_x = base_sample.loc[train_ind, TAKER_FEATURES].astype(np.float32)
    train_y = base_sample.loc[train_ind, 'label_one_hot']
    
    # Scaling
    sc = StandardScaler().fit(train_x)
    train_x_scaled = sc.transform(train_x)
    
    # Save the scaler
    with open('artifacts/atl_scaler_taker.pkl', 'wb') as f:
        pickle.dump(sc, f)
    print("Taker scaler saved to artifacts/atl_scaler_taker.pkl")

    # 3. Train Taker model
    print("Training 'Taker' model...")
    model_taker = build_model(input_shape=len(TAKER_FEATURES))
    model_taker.compile(optimizer='adam', loss=LOSS_FN, metrics=['accuracy'])
    
    # Compute class weights to counter imbalance
    train_labels = base_sample.loc[train_ind, 'label'].values
    cw = compute_class_weight('balanced', classes=np.arange(N_CLASSES), y=train_labels)
    class_weight_dict = dict(enumerate(cw))
    print(f"  Class weight range: {cw.min():.4f} – {cw.max():.4f}")
    
    checkpointer = ModelCheckpoint(
        filepath='artifacts/atl_reco_taker.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_y_tensor = tf.convert_to_tensor(np.array(train_y.values.tolist()), dtype=tf.float32)
    
    model_taker.fit(
        train_x_scaled,
        train_y_tensor,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpointer, lr_scheduler, early_stop],
        class_weight=class_weight_dict
    )
    print("Taker model training complete. Best model saved to artifacts/atl_reco_taker.h5")

    # 4. Prepare and Train Non-Taker model
    print("\nPreparing and training 'Non-Taker' model...")
    # Fix: Use low-engagement subscribers (bottom 30% by total hit count) as non-taker proxy.
    # The original split was broken because every msisdn in the training data has pack_rank==1
    # (the SQL pipeline only includes pack takers). Using low-engagement takers as a proxy
    # gives the non-taker model a population whose behavior is closer to actual non-takers.
    msisdn_hit_counts = base.groupby('msisdn')['hit'].sum()
    hit_threshold = msisdn_hit_counts.quantile(NONTAKER_HIT_QUANTILE)
    low_engagement_msisdns = msisdn_hit_counts[msisdn_hit_counts <= hit_threshold].index
    base_non_taker = base[(base['msisdn'].isin(low_engagement_msisdns)) & (base.pack_rank == 1)].copy()
    print(f"  Non-Taker proxy: {base_non_taker['msisdn'].nunique():,} low-engagement subscribers "
          f"(hit <= {hit_threshold:.0f}, bottom {NONTAKER_HIT_QUANTILE*100:.0f}%)")
    
    base_sample_non_taker = base_non_taker.sample(frac=1, random_state=111)
    train_ind_nt, test_ind_nt = train_test_split(base_sample_non_taker.index, test_size=0.3, random_state=123)
    
    train_x_nt = base_sample_non_taker.loc[train_ind_nt, NONTAKER_FEATURES].astype(np.float32)
    train_y_nt = base_sample_non_taker.loc[train_ind_nt, 'label_one_hot']

    sc_non_taker = StandardScaler().fit(train_x_nt)
    train_x_scaled_nt = sc_non_taker.transform(train_x_nt)

    with open('artifacts/atl_scaler_non_taker.pkl', 'wb') as f:
        pickle.dump(sc_non_taker, f)
    print("Non-Taker scaler saved to artifacts/atl_scaler_non_taker.pkl")

    model_non_taker = build_model(input_shape=len(NONTAKER_FEATURES))
    model_non_taker.compile(optimizer='adam', loss=LOSS_FN, metrics=['accuracy'])
    
    # Compute class weights for non-taker population
    train_labels_nt = base_sample_non_taker.loc[train_ind_nt, 'label'].values
    cw_nt = compute_class_weight('balanced', classes=np.arange(N_CLASSES), y=train_labels_nt)
    class_weight_dict_nt = dict(enumerate(cw_nt))
    print(f"  Non-Taker class weight range: {cw_nt.min():.4f} – {cw_nt.max():.4f}")
    
    checkpointer_nt = ModelCheckpoint(
        filepath='artifacts/atl_reco_non_taker.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    lr_scheduler_nt = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stop_nt = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    train_y_tensor_nt = tf.convert_to_tensor(np.array(train_y_nt.values.tolist()), dtype=tf.float32)

    model_non_taker.fit(
        train_x_scaled_nt,
        train_y_tensor_nt,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpointer_nt, lr_scheduler_nt, early_stop_nt],
        class_weight=class_weight_dict_nt
    )
    print("Non-Taker model training complete. Best model saved to artifacts/atl_reco_non_taker.h5")

    # 5. Evaluate both models on held-out test sets
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    # Evaluate Taker model
    model_taker.load_weights('artifacts/atl_reco_taker.h5')  # Load best checkpoint
    test_x = base_sample.loc[test_ind, TAKER_FEATURES].astype(np.float32)
    test_x_scaled = sc.transform(test_x)
    test_y_true = base_sample.loc[test_ind, 'label'].values
    
    test_pred_proba = model_taker.predict(test_x_scaled)
    taker_metrics = evaluate_model(test_y_true, test_pred_proba, model_name="Taker")
    
    # Evaluate Non-Taker model
    model_non_taker.load_weights('artifacts/atl_reco_non_taker.h5')  # Load best checkpoint
    test_x_nt = base_sample_non_taker.loc[test_ind_nt, NONTAKER_FEATURES].astype(np.float32)
    test_x_scaled_nt = sc_non_taker.transform(test_x_nt)
    test_y_true_nt = base_sample_non_taker.loc[test_ind_nt, 'label'].values
    
    test_pred_proba_nt = model_non_taker.predict(test_x_scaled_nt)
    non_taker_metrics = evaluate_model(test_y_true_nt, test_pred_proba_nt, model_name="Non-Taker")
    
    # Save all metrics
    save_evaluation([taker_metrics, non_taker_metrics])
    print("Training and evaluation pipeline complete.")

if __name__ == '__main__':
    main()
