# common library
import pandas as pd
import numpy as np
import time

from EnvMultipleStock import StockEnv

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

# preprocessor
from preprocessors import *
# model
import os
import warnings
warnings.filterwarnings('ignore')

def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data=data.sort_values(['datadate','tic'],ignore_index=True)
        data.index = data.datadate.factorize()[0]
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)
    
    env = DummyVecEnv([lambda: StockEnv(data)])
    
    model = PPO("MlpPolicy", env)
    model.learn(50000)
    model.save("ppo")
    
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
        
        
if __name__ == "__main__":
    run_model()
