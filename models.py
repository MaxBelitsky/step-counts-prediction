from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, SimpleRNN, Flatten, TimeDistributed, ConvLSTM2D, GRU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from container import ModelContainer
import numpy as np
import pandas as pd



def naive(X_test, k):
        if k == 1:
            return X_test[:, -1].flatten()
        return X_test[:, -k:, -1]


def average_by_dow(history, additional, lag, data_type='steps_date'):
    if lag == 1:
        index = len(history)-1
        additional = additional.loc[:index]
        combined = pd.concat([pd.DataFrame(history, columns=['value']), additional], axis=1)
        dow = combined.iloc[-1, :].dow
        return combined.value[combined.dow == dow].mean()


def preceding_average(history, lag):
    return history[-lag:].mean()


def simple_RNN(units, n_hidden, n_timestamps, n_features, predict_next=1, lr=0.001):
    opt = Adam(learning_rate=lr)
    model = Sequential()
    for i in range(n_hidden):
        model.add(
            SimpleRNN(
            units[i],
            activation='tanh',
            input_shape=(n_timestamps, n_features),
            return_sequences=True)
            )
    model.add(SimpleRNN(units[-1],activation='tanh'))
    model.add(Dense(predict_next))
    model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])

    return model


def LSTM_model(units, n_hidden, n_timestamps,
                n_features, predict_next=1, optimizer='adam', lr=0.001):
    """ Builds and compiles an LSTM model """
    opt = Adam(learning_rate=lr)
    model = Sequential()
    for i in range(n_hidden):
        model.add(
            LSTM(units[i],
            input_shape=(n_timestamps, n_features),
            activation='tanh',
            return_sequences=True)
            )
    model.add(LSTM(units[-1], activation='tanh', input_shape=(n_timestamps, n_features)))
    model.add(Dense(predict_next))
    model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])
    #model.build()
    return model


def GRU_model(units, n_hidden, n_timestamps,
                n_features, predict_next=1, optimizer='adam', lr=0.001):
    """ Builds and compiles an LSTM model """
    opt = Adam(learning_rate=lr)
    model = Sequential()
    for i in range(n_hidden):
        model.add(
            GRU(units[i],
            input_shape=(n_timestamps, n_features),
            activation='tanh',
            return_sequences=True)
            )
    model.add(GRU(units[-1], activation='tanh'))
    model.add(Dense(predict_next))
    model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])
    return model


def BLSTM_model(units, n_hidden, n_timestamps, n_features, predict_next=1, lr=0.001):
    opt = Adam(learning_rate=lr)
    model = Sequential()
    for i in range(n_hidden):
        model.add(
            Bidirectional(
            LSTM(units[i], activation='tanh', return_sequences=True),
            input_shape=(n_timestamps, n_features))
            )
    model.add(
        Bidirectional(
            LSTM(units[-1], activation='tanh')
            )
        )
    model.add(Dense(predict_next))
    model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])
    return model


def Conv_LSTM(n_seq, n_steps, n_features, predict_next=1,
            lr=0.001, filters=64, kernel_size=(1, 3)):
    opt = Adam(learning_rate=lr)
    model = Sequential()
    model.add(
        ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='tanh',
            input_shape=(n_seq, 1, n_steps, n_features)
            )
        )
    model.add(Flatten())
    model.add(Dense(predict_next))
    model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])
    return model


def create_models(config, dataset, data, types='all'):
    models = {}
    # Fill the containers with data
    lags = config[dataset]["lag"]
    future = config[dataset]["future"]

    if types == 'all':
        to_create = config[dataset]["models"]
    else:
        to_create = types

    if to_create == ['Baseline']:
        if dataset != "augmented_steps_date":
            lag = config[dataset]["models"]['Baseline']['lag']
            future = config[dataset]["models"]['Baseline']['future']
            models['Baseline_'+dataset] = ModelContainer('Baseline', 
                    "Baseline", config[dataset]["models"]["Baseline"], 
                    data, lag, future)
    else:
        for lag in lags:
            for model in to_create:
                if dataset == "augmented_steps_date":
                    name = model + f"_{lag}_{future}_aug"
                else:
                    name = model + f"_{lag}_{future}"

                models[name] = ModelContainer(
                    name, model, config[dataset]["models"][model], 
                    data, lag, future)
    return models