import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from .config import NODES, DROPOUT_RATE, N_CLASSES


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for multi-class classification.
    Down-weights easy/majority-class examples, forces model to attend to rare classes."""
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = y_true * tf.pow(1 - y_pred, gamma)
        loss = alpha * weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return focal_loss_fn


def build_model(input_shape):
    """Defines and builds the Keras Sequential model with tapering architecture."""
    model = Sequential([
        Dense(2048, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(N_CLASSES, activation='softmax')
    ])
    return model
