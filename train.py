import pandas as pd
import numpy as np
import yaml

from utils.preprocessing import aggregate_steps, augment, to_supervised
from models import naive, average, simple_RNN, vanilla_LSTM, BLSTM, Conv_LSTM
from container import ModelContainer, create_containers


# Load the data
PATH = 'data/StepCount.csv'
data = pd.read_csv(PATH)

# Load config file
with open("config.yml", 'r') as handle:
    config = yaml.safe_load(handle)

# Create 3 datasets
"""steps_date = aggregate_steps(data, ['date']).Steps
steps_hour = aggregate_steps(data, ['date', 'hour']).Steps
augmented_steps_date = augment(data)"""

steps_date = pd.DataFrame(np.arange(0, 1000))
steps_hour = pd.DataFrame(np.arange(0, 1000))
augmented_steps_date = pd.DataFrame(np.arange(0, 10000).reshape(1000, 10))

all_models = {}

# Fill the containers with data
for dataset in config:
    create_containers(config[dataset], eval(dataset), all_models)

# Create two results DataFrames
results_date = pd.DataFrame(columns=["Model", "Lag", "Future", "MAE", "RMSE", "Error_Steps"])
results_hour = pd.DataFrame(columns=["Model", "Lag", "Future", "MAE", "RMSE", "Error_Steps"])

# Train LSTM models
"""for i in all_models:
    if i.startswith("LSTM"):
        print(i)
        current = all_models[i]
        print(current.lag, current.future, current.hyperparams)"""


print("Done")

