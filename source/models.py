import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pycatch22

#from sktime.classification.kernel_based import RocketClassifier
#from rocket.code.rocket_functions import generate_kernels, apply_kernels
from sktime.transformations.panel.rocket import Rocket



def create_lstm(sequence_length: int, n_features: int, n_classes: int) -> Sequential:
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

def compute_catch22_features(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        X = X.values

    # Ensure X is a 2D numpy array
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Get feature names from first valid time series
    feature_names = pycatch22.catch22_all(X[0].flatten())['names']

    # Initialize array to store all features
    all_features = np.zeros((len(X), len(feature_names)))

    # Compute features for each time series
    for i, series in enumerate(X):
        try:
            # Ensure the series is 1D
            series = series.flatten()
            features = pycatch22.catch22_all(series)['values']
            all_features[i] = features
        except Exception as e:
            print(f"Error processing series {i}: {str(e)}")
            all_features[i] = np.nan

    # Create DataFrame with feature names
    df_features = pd.DataFrame(all_features, columns=feature_names)
    return df_features


class Models:
    def __init__(self, model_name : str, 
                 X_train : np.array, y_train : np.array) -> None:

        if model_name not in ['lstm', 'catch22', 'rocket']:
            raise ValueError("Invalid model name. Choose from ['lstm', 'catch22', 'rocket']")
        self.model_name = model_name
        self.X_train = X_train
        #self.y_train = y_trains

        self.original_classes = np.unique(y_train)
        self.n_classes = len(self.original_classes)

        self.y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoded_y = None

        self.model = None
        self.catch22_train = None
        self.rocket_kernels = None
        return

    # LSTM model ==============================================================================
    def train_lstm(self, epochs=30, batch_size=8, validation_split=0.2, verbose=False) -> None:
        n_features = self.X_train.shape[2]
        sequence_length = self.X_train.shape[1]
        #print(self.X_train.shape[0], n_features, sequence_length)

        self.encoded_y = self.encoder.fit_transform(self.y_train)

        model = create_lstm(sequence_length, n_features, self.n_classes)
        target = self.encoded_y if self.n_classes > 2 else self.y_train

        history = model.fit(
            self.X_train, 
            target, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            verbose=1 if verbose else 0
        )
        self.model = model
        
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
    
    def index_to_original_label(self, indices: np.array) -> np.array:
        return np.array([self.original_classes[idx] for idx in indices])
    
    # Random Forest w/ Catch22 Features =======================================================
    def train_catch22(self) -> None:
        self.catch22_train = compute_catch22_features(self.X_train)

        self.model = RandomForestClassifier()
        self.model.fit(self.catch22_train, self.y_train)
        return

    # Rocket w/ Ridge Classifier ==============================================================
    def train_rocket(self, n_kernels=10000) -> None: 
        # X_train = np.squeeze(self.X_train, axis=1) if len(self.X_train.shape) == 3 and self.X_train.shape[1] == 1 else self.X_train
    
        # self.rocket_kernels = generate_kernels(X_train.shape[-1], n_kernels)
        # X_train_transform = apply_kernels(X_train, self.rocket_kernels)
        
        # self.model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        # self.model.fit(X_train_transform, self.y_train)

        #self.model = RocketClassifier(num_kernels=n_kernels)
        #self.model.fit(self.X_train, self.y_train)

        self.rocket_kernels = Rocket(num_kernels=n_kernels) 
        self.rocket_kernels.fit(self.X_train) 

        X_train_trasform = self.rocket_kernels.transform(self.X_train)
        self.model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.model.fit(X_train_trasform, self.y_train)
        return

    # Predictions ==============================================================================
    def predict(self, X_test : np.array) -> tuple:
        if(self.model_name == 'lstm'):  
            y_proba = self.model.predict(X_test, verbose=0)

            if self.n_classes == 2:
                    y_pred_indices = np.where(y_proba > 0.5, 1, 0).flatten()
            else:
                y_pred_indices = np.argmax(y_proba, axis=1)
            y_pred = self.index_to_original_label(y_pred_indices)
    
            if self.n_classes == 2 and y_proba.shape[1] == 1:
                y_proba = np.hstack([1-y_proba, y_proba])
            
        elif(self.model_name == 'catch22'):
            catch22_test = compute_catch22_features(X_test)
            y_pred = self.model.predict(catch22_test)
            y_proba = self.model.predict_proba(catch22_test)
        
        elif(self.model_name == 'rocket'): 
            # X_test = np.squeeze(X_test, axis=1) if len(X_test.shape) == 3 and X_test.shape[1] == 1 else X_test
            X_test_transform  = self.rocket_kernels.transform(X_test)
            y_pred = self.model.predict(X_test_transform)
            y_proba = self.model.decision_function(X_test_transform)

        return y_pred, y_proba