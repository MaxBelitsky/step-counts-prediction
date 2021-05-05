from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, SimpleRNN, Flatten, TimeDistributed, ConvLSTM2D, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical



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


def simple_RNN(output_size, n_timestamps, n_features, next_predicted=1):
    model = Sequential()
    model.add(
        SimpleRNN(
            output_size,
            activation='tanh',
            input_shape=(n_timestamps, n_features)))
    model.add(Dense(next_predicted))
    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

    return model


def vanilla_LSTM(output_sizes, n_hidden, n_timestamps,
                n_features, predict_next=1, optimizer='adam'):
    """ Builds and compiles an LSTM model """
    model = Sequential()
    for i in range(n_hidden):
        model.add(
            LSTM(output_sizes[i],
            input_shape=(n_timestamps, n_features),
            activation='tanh',
            return_sequences=True)
            )
    model.add(LSTM(output_size[-1], activation='tanh'))
    model.add(Dense(predict_next))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])
    return model


def BLSTM(output_size, n_timestamps, n_features, next_predicted=1):
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(output_size, activation='tanh'),
            input_shape=(n_timestamps, n_features)
            )
        )
    model.add(Dense(next_predicted))
    model.compile(optimizer='adam', loss='mse')
    return model


def Conv_LSTM(n_seq, n_steps, n_features, next_predicted=1):
    model = Sequential()
    model.add(
        ConvLSTM2D(
            filters=64,
            kernel_size=(1,3),
            activation='tanh',
            input_shape=(n_seq, 1, n_steps, n_features)
            )
        )
    model.add(Flatten())
    model.add(Dense(next_predicted))
    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])
    return model
