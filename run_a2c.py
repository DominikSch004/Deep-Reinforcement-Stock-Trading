import os
import warnings

import numpy as np
import pandas as pd
from stable_baselines3 import A2C

from env.EnvMultipleStock import StockEnvTrade
from preprocessing.preprocessors import *

warnings.filterwarnings('ignore')


def get_data(train=False):
    preprocessed_path = "done_data.csv"
    
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = data.sort_values(['datadate','tic'],ignore_index=True)
        data.index = data.datadate.factorize()[0]
        data.to_csv(preprocessed_path)
        
    if train:
        return data_split(data, start=20090000, end=20160000)
    else:
        return data_split(data, start=20160000, end=20201908)

def train_model(timesteps):
    
    data = get_data(train=True)
    env_train = StockEnvTrade(data, model_name="A2C", train=True)
    
    model = A2C("MlpPolicy", env_train)
    model.learn(timesteps)
    model.save("a2c{}".format(timesteps))

    return None

def test_model(timesteps):
    
    data = get_data(train=False)
    env_test = StockEnvTrade(data, model_name="A2C", train=False)
    
    model = A2C.load("a2c{}".format(timesteps))
    
    obs = env_test.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        
if __name__ == "__main__":
    timesteps = 100000
    train_model(timesteps)
    test_model(timesteps)