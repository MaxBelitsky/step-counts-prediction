
from sklearn.preprocessing import MinMaxScaler
from utils.preprocessing import to_supervised
import numpy as np
import pandas as pd


class ModelContainer:
    def __init__(self, name, model_type, hyperparams, data, lag, future):
        self.hyperparams = hyperparams
        self.data = data
        self.lag = lag
        self.future = future
        self.name = name
        self.model_type = model_type
        self.history = None
        self.predictions = None
        self.prepare_data()

    def prepare_data(self):

        # Split to training, validation and test sets
        n = len(self.data)
        train_data = self.data[0:int(n*0.8)]
        val_data = self.data[int(n*0.8):int(n*0.9)]
        test_data = self.data[int(n*0.9):]

        # Normalize the data with MinMax normalization
        self.scaler = MinMaxScaler()
        if len(self.data.shape) == 1:
            n_features = 1
            train_data = self.scaler.fit_transform(
                train_data.to_numpy().reshape(-1, 1))
            val_data = self.scaler.transform(
                val_data.to_numpy().reshape(-1, 1))
            test_data = self.scaler.transform(
                test_data.to_numpy().reshape(-1, 1))
        else:
            # TODO: df["Steps"] instead of .iloc
            n_features = train_data.shape[1]
            train_data.iloc[:, 0] = self.scaler.fit_transform(
                train_data.iloc[:, 0].to_numpy().reshape(-1, 1))
            val_data.iloc[:, 0] = self.scaler.transform(
                val_data.iloc[:, 0].to_numpy().reshape(-1, 1))
            test_data.iloc[:, 0] = self.scaler.transform(
                test_data.iloc[:, 0].to_numpy().reshape(-1, 1))

        if self.model_type == "Baseline":
            pass
        elif self.model_type == "ConvLSTM":
            # TODO: complete the function
            print(self.model_type)
        else:
            # Convert sequence to supervised sequence
            self.X_train, self.y_train = to_supervised(
                np.concatenate((train_data, val_data[:self.future])),
                self.lag,
                self.future
                )
            self.X_val, self.y_val = to_supervised(
                np.concatenate((val_data, test_data[:self.future])),
                self.lag,
                self.future
                )
            self.X_test, self.y_test = to_supervised(test_data,
                self.lag,
                self.future
                )
            
            if len(self.data.shape) > 1:
                self.y_train = self.y_train[:, 0]
                self.y_val = self.y_val[:, 0]
                self.y_test = self.y_test[:, 0]

            # Reshape to (samples, timesteps, features)
            self.X_train = self.X_train.reshape(
                self.X_train.shape[0],
                self.X_train.shape[1],
                n_features
                )
            self.X_val = self.X_val.reshape(
                self.X_val.shape[0],
                self.X_val.shape[1],
                n_features
                )
            
            self.X_test = self.X_test.reshape(
                self.X_test.shape[0],
                self.X_test.shape[1],
                n_features
                )

    def __repr__(self):
        return "{}".format(self.name)
    
    def plot_predictions(self):
        if self.future == 1 and self.predictions is not None:
            plt.figure(figsize=(20, 5))
            plt.plot(self.y_test)
            plt.plot(self.predictions)
        else:
            print("k > 1")

    def plot_history(self):
        if history is not None:
            plt.figure(figsize=(20, 5))
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.plot(self.history.history['mean_absolute_error'])
            plt.plot(self.history.history['val_mean_absolute_error'])
            plt.legend()