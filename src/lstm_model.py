import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

class MatchPredictor:
    def __init__(self, sequence_length=5, feature_dim=5):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, self.feature_dim), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: Home win, Draw, Away win
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_match_data(self, home_seq, away_seq, table_features):
        """Combine sequence and table features."""
        # Combine home and away sequences
        match_sequence = np.concatenate([home_seq, away_seq], axis=-1)
        
        # Add table position features
        match_features = np.concatenate([
            match_sequence.reshape(1, self.sequence_length, -1),
            table_features.reshape(1, 1, -1).repeat(self.sequence_length, axis=1)
        ], axis=-1)
        
        return match_features
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model with validation."""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        return history
    
    def predict_proba(self, X):
        """Predict match probabilities."""
        return self.model.predict(X) 