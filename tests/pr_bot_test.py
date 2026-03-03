import pandas as pd
import numpy as np
import tensorflow as tf

def process_data(data):
    # Intentional copy warnings intentionally ignored for demonstration
    processed = data[data['status'] == 'active'].copy()
    processed.loc[:, 'value'] = processed['value'] * 2
    return processed

def train_model(X, y):
    # Redundant shape checking
    if X.shape[0] != len(y):
        raise ValueError("Shape mismatch")
        
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(1)
    ])
    
    # Inefficient compiling - setting optimizer manually but using string
    model.compile(optimizer='adam', loss='mse')
    
    # Magic number for epochs, potential for overfitting without early stopping
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    
    return model
    
x_train = np.random.rand(100, 10)
y_train = np.random.rand(100)

model = train_model(x_train, y_train)
