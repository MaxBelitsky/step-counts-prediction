
from sklearn.preprocessing import MinMaxScaler
from utils.preprocessing import to_supervised
import numpy as np
import pandas as pd


class ModelContainer:
    def __init__(self, name, model, lag, future, X_train, y_train, X_val, y_val, X_test,
               y_test, predictions, scaler, history=None, hyperparams=None):
        self.name = name
        self.model = model
        self.lag = lag
        self.future = future
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = predictions
        self.scaler = scaler
        self.hyperparams = hyperparams

    def __repr__(self):
        return "{}, {}, {}, {}, {}, {}, {}".format(
            self.name, self.lag, self.future, self.X_train.shape,
            self.y_train.shape, self.X_test.shape, self.y_test.shape)
    
    def plot_predictions(self):
        if self.k == 1 and self.predictions is not None:
            plt.figure(figsize=(20, 5))
            plt.plot(self.y_test.squeeze())
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


def create_containers(config, data, containers):
    models = config['models']
    
    # Split to training, validation and test sets
    n = len(data)
    train_data = data[0:int(n*0.8)]
    val_data = data[int(n*0.8):int(n*0.9)]
    test_data = data[int(n*0.9):]
    # Normalize the data with MinMax normalization
    scaler = MinMaxScaler()
    if data.shape[1] == 1:
        train_data = scaler.fit_transform(train_data.to_numpy().reshape(-1, 1))
        val_data = scaler.transform(val_data.to_numpy().reshape(-1, 1))
        test_data = scaler.transform(test_data.to_numpy().reshape(-1, 1))
    else:
        # TODO: df["Steps"] instead of .iloc
        train_data.iloc[:, 0] = scaler.fit_transform(train_data.iloc[:, 0].to_numpy().reshape(-1, 1))
        val_data.iloc[:, 0] = scaler.transform(val_data.iloc[:, 0].to_numpy().reshape(-1, 1))
        test_data.iloc[:, 0] = scaler.transform(test_data.iloc[:, 0].to_numpy().reshape(-1, 1))

    for i in models:
        for lag in config['lag']:
            if i == "Baseline":
                pass
            elif i == "ConvLSTM":
                # TODO: complete the function
                print(i)
            else:
                # Convert sequence to supervised sequence
                X_train, y_train = to_supervised(
                    np.concatenate((train_data, val_data[:config['future']])),
                    lag,
                    config['future']
                    )
                X_val, y_val = to_supervised(
                    np.concatenate((val_data, test_data[:config['future']])),
                    lag,
                    config['future']
                    )
                X_test, y_test = to_supervised(test_data,
                    lag,
                    config['future']
                    )
                
                if data.shape[1] != 1:
                    y_train = y_train[:, 0]
                    y_val = y_val[:, 0]
                    y_test = y_test[:, 0]

                # Reshape to (samples, timesteps, features)
                n_features = train_data.shape[1]
                X_train = X_train.reshape(
                    X_train.shape[0],
                    X_train.shape[1],
                    n_features
                    )
                X_val = X_val.reshape(
                    X_val.shape[0],
                    X_val.shape[1],
                    n_features
                    )
                
                X_test = X_test.reshape(
                    X_test.shape[0],
                    X_test.shape[1],
                    n_features
                    )
                #print(X_train.shape, y_train.shape)
                #print(X_val.shape, y_val.shape)

                # Add to containers
                container = ModelContainer(
                    name=i,
                    model=None,
                    lag=lag,
                    future=config['future'],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test, 
                    predictions=None,
                    scaler=scaler,
                    hyperparams=models[i]
                )
                if data.shape[1] != 1:
                    containers[i+f"_{lag}_{config['future']}_aug"] = container
                else:
                    containers[i+f"_{lag}_{config['future']}"] = container