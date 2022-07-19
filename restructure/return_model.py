# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3

from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from restructure.trade_env import StockEnvTrade


def train_A2C(env_train, model_name, timesteps=25000):
	"""A2C model"""

	start = time.time()
	model = A2C('MlpPolicy', env_train, verbose=0, n_steps=32)
	model.learn(total_timesteps=timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training time (A2C): ', (end - start) / 60, ' minutes')
	return model

def train_ACER(env_train, model_name, timesteps=25000):
	start = time.time()
	model = ACER('MlpPolicy', env_train, verbose=0)
	model.learn(total_timesteps=timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training time (A2C): ', (end - start) / 60, ' minutes')
	return model


def train_DDPG(env_train, model_name, timesteps=10000):
	"""DDPG model"""

	# add the noise objects for DDPG
	n_actions = env_train.action_space.shape[-1]
	param_noise = None
	action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

	start = time.time()
	model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
	model.learn(total_timesteps=timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training time (DDPG): ', (end-start)/60,' minutes')
	return model

def train_PPO(env_train, model_name, timesteps=50000):
	"""PPO model"""

	start = time.time()
	model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 32)
	#model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

	model.learn(total_timesteps=timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training time (PPO): ', (end - start) / 60, ' minutes')
	return model

def train_GAIL(env_train, model_name, timesteps=1000):
	"""GAIL Model"""
	#from stable_baselines.gail import ExportDataset, generate_expert_traj
	start = time.time()
	# generate expert trajectories
	model = SAC('MLpPolicy', env_train, verbose=1)
	generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)

	# Load dataset
	dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
	model = GAIL('MLpPolicy', env_train, dataset, verbose=1)

	model.learn(total_timesteps=1000)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training time (PPO): ', (end - start) / 60, ' minutes')
	return model

def DRL_validation(model, test_data, test_env, test_obs):
		###validation process###
	for i in range(len(test_data.index.unique())):
		action, _states = model.predict(test_obs)
		test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
	###Calculate Sharpe ratio based on validation results###
	df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
	df_total_value.columns = ['account_value_train']
	df_total_value['daily_return'] = df_total_value.pct_change(1)
	sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
			 df_total_value['daily_return'].std()
	return sharpe


def validate(df):
	ppo_sharpe_list = []
	a2c_sharpe_list = []	
	ddpg_sharpe_list = []
	
	insample_turbulence = df.drop_duplicates(subset=['datadate'])
	insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

	historical_turbulence = df.drop_duplicates(subset=['datadate'])

	historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)


	if historical_turbulence_mean > insample_turbulence_threshold:
		# if the mean of the historical data is greater than the 90% quantile of insample turbulence data
		# then we assume that the current market is volatile,
		# therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
		# meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
		turbulence_threshold = insample_turbulence_threshold
	else:
		# if the mean of the historical data is less than the 90% quantile of insample turbulence data
		# then we tune up the turbulence_threshold, meaning we lower the risk
		turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)

	env_val = DummyVecEnv([lambda: StockEnvValidation(df,
														turbulence_threshold=turbulence_threshold,
														iteration=1)])

	obs_val = env_val.reset()
	# model_a2c = train_A2C(env_train, model_name="A2C_{}k_dow".format(n), timesteps=n)
	# model_ddpg = train_DDPG(env_train, model_name="DDPG_{}k_dow".format(n), timesteps=n)
	# model_ppo = train_PPO(env_train, model_name="PPO_{}k_dow".format(n), timesteps=n)
	# print("====Finished trainig with {} timesteps====".format(n))

	model_a2c = A2C.load("trained_model/Third/A2C_600000k_dow.zip")
	model_ddpg = DDPG.load("trained_models/Third/DDPG_40000k_dow.zip")
	model_ppo = PPO2.load("trained_models/Third/PPO_40000k_dow.zip")

	print("====Starting the Validation process")
	DRL_validation(model=model_a2c, test_data=df, test_env=env_val, test_obs=obs_val)
	sharpe_a2c = get_validation_sharpe(1)
	DRL_validation(model=model_ddpg, test_data=df, test_env=env_val, test_obs=obs_val)
	sharpe_ddpg = get_validation_sharpe(1)
	DRL_validation(model=model_ppo, test_data=df, test_env=env_val, test_obs=obs_val)
	sharpe_ppo = get_validation_sharpe(1)

	print("Sharpe Ratio A2C: "+str(sharpe_a2c))
	print("Sharpe Ratio DDPG: "+str(sharpe_ddpg))
	print("Sharpe Ratio PPO: "+str(sharpe_ppo))

	ppo_sharpe_list.append(sharpe_ppo)
	a2c_sharpe_list.append(sharpe_a2c)
	ddpg_sharpe_list.append(sharpe_ddpg)

	if sharpe_a2c >= sharpe_ddpg and sharpe_a2c >= sharpe_ppo:
		return sharpe_a2c
	elif sharpe_ddpg >= sharpe_a2c and sharpe_ddpg >= sharpe_ppo:
		return sharpe_ddpg
	else:
		return sharpe_ppo