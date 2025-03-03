import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def create_lstm(sequence_length: int, n_features: int, n_classes: int) -> Sequential:
    # Determine the output layer configuration based on number of classes
    if n_classes == 2:
        # Binary classification
        output_units = 1
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        # Multi-class classification
        output_units = n_classes
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    
    model = Sequential([
        LSTM(32, input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(output_units, activation=activation)
    ])

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    return model

class Models:
    def __init__(self, model_name: str, 
                 X_train: np.array, y_train: np.array) -> None:

        if model_name not in ['lstm', 'catch22', 'rocket']:
            raise ValueError("Invalid model name. Choose from ['lstm', 'catch22', 'rocket']")
        self.model_name = model_name
        self.X_train = X_train
        
        # Reshape y_train to ensure it's 2D for the encoder
        self.y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        
        # Store original class labels for later mapping
        self.original_classes = np.unique(y_train)
        self.n_classes = len(self.original_classes)
        
        # One-hot encode the labels
        self.encoder = OneHotEncoder(sparse=False)
        self.encoded_y = self.encoder.fit_transform(self.y_train)
        
        self.model = None
        self.catch22_train = None
        self.rocket_kernels = None
        return

    # LSTM model ==============================================================================
    def train_lstm(self, epochs=30, batch_size=8, validation_split=0.2, verbose=False) -> None:

        n_features = self.X_train.shape[2]
        sequence_length = self.X_train.shape[1]
        
        # Create an appropriate model based on number of classes
        model = create_lstm(sequence_length, n_features, self.n_classes)
        
        # Use the appropriate target based on number of classes
        target = self.encoded_y if self.n_classes > 2 else self.y_train
        
        # Train the model
        history = model.fit(
            self.X_train, 
            target, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            verbose=1 if verbose else 0
        )
        self.model = model
        
        # Plot training history if verbose
        if verbose:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('LSTM Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('LSTM Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.tight_layout()
            plt.show()
        return
    
    # Predictions ==============================================================================
    def predict(self, X_test: np.array) -> np.array:

        if self.model is None:
            raise ValueError(f"Model {self.model_name} has not been trained yet")
            
        if self.model_name == 'lstm':
            # Get model raw outputs
            y_proba = self.model.predict(X_test, verbose=0)
            
            # Process outputs based on number of classes
            if self.n_classes == 2:
                # Binary classification
                y_pred_indices = np.where(y_proba > 0.5, 1, 0).flatten()
            else:
                # Multi-class classification
                y_pred_indices = np.argmax(y_proba, axis=1)
            
            # Map indices back to original class labels
            return self.index_to_original_label(y_pred_indices)
            
        elif self.model_name == 'catch22':
            # Placeholder for catch22 prediction logic
            pass
            
        elif self.model_name == 'rocket':
            # Placeholder for rocket prediction logic
            pass
    
    def predict_proba(self, X_test: np.array) -> np.array:
        if self.model is None:
            raise ValueError(f"Model {self.model_name} has not been trained yet")
            
        if self.model_name == 'lstm':
            probs = self.model.predict(X_test, verbose=0)
            
            # For binary classification, ensure output is in format [P(0), P(1)]
            if self.n_classes == 2 and probs.shape[1] == 1:
                return np.hstack([1-probs, probs])
            return probs
        
        # Add implementations for other model types as needed
        return None
    
