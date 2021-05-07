import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

from utils.preprocessing import aggregate_steps, augment, to_supervised
from models import naive, average
from models import simple_RNN, LSTM_model, GRU_model, BLSTM_model, Conv_LSTM
from models import create_models


# Load the data
PATH = 'data/StepCount.csv'
data = pd.read_csv(PATH)

# Load config file
with open("config.yml", 'r') as handle:
    config = yaml.safe_load(handle)

# Create 3 datasets
steps_date = aggregate_steps(data, ['date']).Steps
steps_hour = aggregate_steps(data, ['date', 'hour']).Steps
augmented_steps_date = augment(data)

print(steps_date.shape)
print(steps_hour.shape)
print(augmented_steps_date.shape)

"""steps_date = pd.DataFrame(np.arange(0, 1000))
steps_hour = pd.DataFrame(np.arange(0, 1000))
augmented_steps_date = pd.DataFrame(np.arange(0, 10000).reshape(1000, 10))"""

# Create RNN models
rnns = create_models(config, steps_date, steps_hour, augmented_steps_date, ["BLSTM"])

# Create two results DataFrames
results_date = pd.DataFrame(columns=["Model", "Lag", "MAE", "RMSE", "Error_Steps"])
results_hour = pd.DataFrame(columns=["Model", "Lag", "MAE", "RMSE", "Error_Steps"])

#EPOCHS = 1000
row = 0


for i in rnns:
    current = rnns[i]

    units = current.hyperparams['units']
    n_hidden = current.hyperparams['n_hidden']
    lr = current.hyperparams['lr']
    m_batch_size = current.hyperparams['m_batch_size']
    X_train, y_train, X_val, y_val, X_test, y_test = current.get_data()
    n_timestamps = X_train.shape[1]
    n_features = X_train.shape[2]
    next_predicted = current.future

    model = simple_RNN(units, n_hidden, n_timestamps, n_features, next_predicted, lr)
    print(model.summary())
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        shuffle=False,
        epochs=1,
        batch_size=m_batch_size,
        verbose=0
        )

    #model.save("models/"+i+".h5")

    # TODO: add predictions to model containers
    y_pred = model.predict(X_test).squeeze()

    mae = mean_absolute_error(y_test, y_pred)
    error_steps = int(
        current.scaler.inverse_transform(np.array(mae).reshape(1, -1))[0][0]
        )
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    if current.future == 1:
        results_date.loc[row] = [
                        current.name, current.lag,
                        round(mae, 4),
                        round(rmse, 4),
                        error_steps]
    else:
        results_hour.loc[row] = [
                        current.name, current.lag,
                        round(mae, 4),
                        round(rmse, 4),
                        error_steps]
    
    row += 1


print(results_date)
print(results_hour)