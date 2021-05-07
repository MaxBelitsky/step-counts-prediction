from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, SimpleRNN, Flatten, TimeDistributed, ConvLSTM2D, Dropout, GRU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from container import ModelContainer



def naive(X_test, n, k):
    if k == 1:
        return X_test[:, -1].flatten()
    
    predictions = []
    for i in range(len(X_test)):
        t = k
        curr_window = X_test[i, :]
        curr_predictions = []
        while t != 0:
            prediction = curr_window[-1]
            curr_predictions.append(prediction)
            curr_window = np.append(curr_predictions, prediction)
            t -= 1
        predictions.append(curr_predictions)
    return predictions


def average(history, n, k):
  if k == 1:
    return np.mean(history[-n:])
  else:
    predictions = []
    while k != 0:
      prediction = np.mean(history[-n:])
      predictions.append(prediction)
      history = np.append(history, prediction)
      k -= 1
    return predictions


def simple_RNN(units, n_hidden, n_timestamps, n_features, next_predicted=1, lr=0.001):
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
    model.add(Dense(next_predicted))
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
    model.add(LSTM(units[-1], activation='tanh'))
    model.add(Dense(predict_next))
    model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])
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


def BLSTM_model(units, n_hidden, n_timestamps, n_features, next_predicted=1, lr=0.001):
    opt = Adam(learning_rate=lr)
    model = Sequential()
    for i in range(n_hidden):
        model.add(
            Bidirectional(
            LSTM(units[i], activation='tanh'),
            input_shape=(n_timestamps, n_features),
            return_sequences=True)
            )
    model.add(
        Bidirectional(
            LSTM(units[-1], activation='tanh')
            )
        )
    model.add(Dense(next_predicted))
    model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])
    return model


def Conv_LSTM(n_seq, n_steps, n_features, next_predicted=1,
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
    model.add(Dense(next_predicted))
    model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])
    return model


def create_models(config, steps_date, steps_hour, augmented_steps_date, types='all'):
    models = {}
    # Fill the containers with data
    for dataset in config:
        lags = config[dataset]["lag"]
        future = config[dataset]["future"]

        for lag in lags:
            
            if types == 'all':
                to_create = config[dataset]["models"]
            else:
                to_create = types

            for model in to_create:
                if dataset == "augmented_steps_date":
                    name = model + f"_{lag}_{future}_aug"
                else:
                    name = model + f"_{lag}_{future}"

                models[name] = ModelContainer(
                    name, model, config[dataset]["models"][model], 
                    eval(dataset), lag, future)
    return models