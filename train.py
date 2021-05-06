import pandas as pd
import numpy as np
import yaml

from utils.preprocessing import aggregate_steps, augment, to_supervised
from models import naive, average, simple_RNN, vanilla_LSTM, BLSTM, Conv_LSTM
from container import ModelContainer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import warnings
warnings.filterwarnings("ignore")


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

all_models = {}

# Fill the containers with data
for dataset in config:
    lags = config[dataset]["lag"]
    future = config[dataset]["future"]

    for lag in lags:

        for model in config[dataset]["models"]:
            if dataset == "augmented_steps_date":
                name = model + f"_{lag}_{future}_aug"
            else:
                name = model + f"_{lag}_{future}"

            all_models[name] = ModelContainer(
                name, model, config[dataset]["models"][model], 
                eval(dataset), lag, future)


# Create two results DataFrames
results_date = pd.DataFrame(columns=["Model", "Lag", "MAE", "RMSE", "Error_Steps"])
results_hour = pd.DataFrame(columns=["Model", "Lag", "MAE", "RMSE", "Error_Steps"])

#EPOCHS = 1000
row = 0

for i in all_models:
    if i.startswith("RNN"):
        current = all_models[i]

        units = current.hyperparams['units']
        lr = current.hyperparams['lr']
        m_batch_size = current.hyperparams['m_batch_size']
        #n_timesteps

        model = simple_RNN(units, current.X_train.shape[1],
                        current.X_train.shape[2], current.future)
        
        history = model.fit(
            current.X_train,
            current.y_train,
            validation_data=(current.X_val, current.y_val),
            epochs=1,
            batch_size=m_batch_size,
            verbose=0
            )

        #model.save("models/"+i+".h5")

        y_pred = model.predict(current.X_test).squeeze()

        mae = mean_absolute_error(current.y_test, y_pred)
        error_steps = int(
            current.scaler.inverse_transform(np.array(mae).reshape(1, -1))[0][0]
            )
        rmse = sqrt(mean_squared_error(current.y_test, y_pred))

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