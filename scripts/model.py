import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from .config import NODES, DROPOUT_RATE, N_CLASSES

def build_model(input_shape):
    """Defines and builds the Keras Sequential model."""
    model = Sequential([
        Dense(NODES, activation='relu', input_shape=(input_shape,)),
        Dropout(DROPOUT_RATE),
        Dense(NODES, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(NODES, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(NODES, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(N_CLASSES, activation='softmax')
    ])
    return model
