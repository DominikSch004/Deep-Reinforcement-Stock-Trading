import alpaca_trade_api as tradeapi
import gym
import time
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use('Agg')
import pickle

import matplotlib.pyplot as plt
from config.config import *

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 50
# initial amount of money we have in our account
# INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio
STOCK_DIM = 4
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

# turbulence index: 90-150 reasonable threshold
#TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4


api = tradeapi.REST(base_url=API_BASE_URL, key_id=API_KEY, secret_key=SECRET_KEY)


class StockEnvTrade(gym.Env):
	"""A stock trading environment for OpenAI gym"""
	metadata = {'render.modes': ['human']}

	def __init__(self, df, turbulence_threshold=140, previous_state=[], model_name=''):
		#super(StockEnv, self).__init__()
		#money = 10 , scope = 1
		self.data = df
		self.account = api.get_account()
		# available money
		self.acc_balance = float(self.account.equity)
		self.previous_state = previous_state
		# action_space normalization and shape is STOCK_DIM
		self.action_space = spaces.Box(low = -1, high = 1,shape = (1,))

		# Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
		# +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
		self.observation_space = spaces.Box(low=0, high=np.inf, shape = (8,))

		'''
		make the env for a single stock,
		use alpaca to get info like owned shares and current balance,
		save the state for every stock or for each stock?
		'''

		# load data from a pandas dataframe
		self.turbulence_threshold = turbulence_threshold

		#print("Money: {}".format(self.account.equity))

		# initalize state
		if self.previous_state == []:
			self.asset_memory = [self.acc_balance]
			self.turbulence = 0
			self.cost = 0
			self.trades = 0
			#self.iteration=self.iteration
			self.rewards_memory = []
			#initiate state
			print("Empty state")
			self.state = [self.acc_balance] + \
						self.data.close.values.tolist() + \
						[0]*STOCK_DIM + \
						self.data.macd.values.tolist() + \
						self.data.rsi.values.tolist() + \
						self.data.cci.values.tolist() + \
						self.data.adx.values.tolist()

		else:
			previous_total_asset = self.previous_state[0]+ \
				sum(np.array(self.previous_state[1:(STOCK_DIM+1)])*np.array(self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
			self.asset_memory = [previous_total_asset]
			#self.asset_memory = [self.previous_state[0]]
			self.turbulence = 0
			self.cost = 0
			self.trades = 0
			#self.iteration=iteration
			self.rewards_memory = []
			print("used state")
			print(self.previous_state[0])

			self.state = [self.previous_state[0]] + \
						  self.data.close.values.tolist() + \
						  self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]+ \
						  self.data.macd.values.tolist() + \
					  	  self.data.rsi.values.tolist() + \
					  	  self.data.cci.values.tolist() + \
					  	  self.data.adx.values.tolist()
		#self.reset()
		self._seed()
		self.model_name=model_name        


	def _sell_stock(self, index, action):
		# perform sell action based on the sign of the action
		if self.turbulence<self.turbulence_threshold:
			if self.state[index+STOCK_DIM+1] > 0:
				#update balance

				api.submit_order(
					symbol=self.data.tic.values.tolist()[index],
					qty=HMAX_NORMALIZE,
					side='sell',
					type='market',
					time_in_force='gtc'
				)

				time.sleep(1)
				self.state[0] += \
				self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * (1- TRANSACTION_FEE_PERCENT)
				print(index)
				print("Stock: {}".format(self.data.tic.values.tolist()[index]))
				print("selling State: {}".format(self.state[0]))
				self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
				print("selling State + index: {}".format(self.state[index+STOCK_DIM+1]))
				self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
				 TRANSACTION_FEE_PERCENT
				self.trades+=1
				print("Sold")
				#self.flag = 0
			else:
				pass
		else:
			# if turbulence goes over threshold, just clear out all positions 
			if self.state[index+STOCK_DIM+1] > 0:
				print("Cleared all positions")
				#update balance
				self.state[0] += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
							  (1- TRANSACTION_FEE_PERCENT)
				self.state[index+STOCK_DIM+1] =0
				self.cost += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
							  TRANSACTION_FEE_PERCENT
				self.trades+=1
				#self.flag = 0
			else:
				pass
	
	def _buy_stock(self, index, action):
		# perform buy action based on the sign of the action
		if self.turbulence < self.turbulence_threshold:
			print("Bought")


			api.submit_order(
				symbol=self.data.tic.values.tolist()[index],
				qty=HMAX_NORMALIZE,
				side='buy',
				type='market',
				time_in_force='gtc'
			)

			time.sleep(1)
			print(self.state)
			available_amount = self.state[0] // self.state[index+1]
			print('available_amount:{}'.format(available_amount))

			print("Stock: {}".format(self.data.tic.values.tolist()[index]))
			#update balance
			
			self.state[0] -= self.state[index+1]*min(available_amount, action)* \
							  (1+ TRANSACTION_FEE_PERCENT)

			print("buying State: {}".format(self.state[0]))

			self.state[index+STOCK_DIM+1] += min(available_amount, action)
			
			print("buying State + index: {}".format(self.state[index+STOCK_DIM+1]))
			
			self.cost+=self.state[index+1]*min(available_amount, action)* \
							  TRANSACTION_FEE_PERCENT
			self.trades+=1
			#self.flag = 1
		else:
			# if turbulence goes over threshold, just stop buying
			pass
		
	def step(self, actions):
		# print(actions)

		print("trying step")
		try:
			# print(np.array(self.state[1:29]))

			actions = actions * HMAX_NORMALIZE
			#actions = (actions.astype(int))
			if self.turbulence>=self.turbulence_threshold:
				actions=np.array([-HMAX_NORMALIZE])
			
			begin_total_asset = float(self.state[0])+ \
			float(sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)])))
			print("begin_total_asset:{}".format(begin_total_asset))

			
			argsort_actions = np.argsort(actions)
			
			sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
			buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
			# print(sell_index)
			# print(buy_index)

			for index in sell_index:
				print('take sell action'.format(actions[index]))
				self._sell_stock(index, actions[index])

			for index in buy_index:
				print('take buy action: {}'.format(actions[index]))
				self._buy_stock(index, actions[index])

			self.turbulence = self.data['turbulence'].values
			#print(self.turbulence)
			#load next state
			print("stock_shares:{}".format(self.state[29:]))
			self.state =  [self.state[0]] + \
					self.data.close.values.tolist() + \
					list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
					self.data.macd.values.tolist() + \
					self.data.rsi.values.tolist() + \
					self.data.cci.values.tolist() + \
					self.data.adx.values.tolist()

			
			np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


			end_total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
			self.asset_memory.append(end_total_asset)

			print("end_total_asset:{}".format(end_total_asset))
			
			self.reward = end_total_asset - begin_total_asset            
			# print("step_reward:{}".format(self.reward))
			self.rewards_memory.append(self.reward)
			
			self.reward = self.reward*REWARD_SCALING

			plt.plot(self.asset_memory,'r')
			plt.savefig('results/account_value_trade_{}.png'.format(self.model_name))
			plt.close()
			df_total_value = pd.DataFrame(self.asset_memory)
			df_total_value.to_csv('results/account_value_trade_{}.csv'.format(self.model_name))

			print("total_reward:{}".format(float(self.state[0])+sum(np.array(self.state[1:(STOCK_DIM+1)], dtype=object)*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)], dtype=object)) - float(self.asset_memory[0] )))
			print("total_cost: ", self.cost)
			print("total trades: ", self.trades)

			df_total_value.columns = ['account_value']
			df_total_value['daily_return']=df_total_value.pct_change(1)
			sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ \
					df_total_value['daily_return'].std()
			print("Sharpe: ",sharpe)
			
			df_rewards = pd.DataFrame(self.rewards_memory)
			df_rewards.to_csv('results/account_rewards_trade_{}.csv'.format(self.model_name))
			
			# print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))

		except Exception as e:
			print("Step didnt work")
			print(e)

		with open('obs.pkl', 'wb') as f:  
			   pickle.dump(self.state, f)

		return self.state, self.reward, {}

	def reset(self):  
		if self.previous_state != []:
			previous_total_asset = self.previous_state[0]+ \
			sum(np.array(self.previous_state[1:(STOCK_DIM+1)])*np.array(self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
			self.asset_memory = [previous_total_asset]
			#self.asset_memory = [self.previous_state[0]]
			self.turbulence = 0
			self.cost = 0
			self.trades = 0
			#self.iteration=iteration
			self.rewards_memory = []
			#initiate state
			#self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]
			#[0]*STOCK_DIM + \

			self.state = [self.previous_state[0]] + \
						  self.data.close.values.tolist() + \
						  self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]+ \
						  self.data.macd.values.tolist() + \
					  	  self.data.rsi.values.tolist() + \
					  	  self.data.cci.values.tolist() + \
					  	  self.data.adx.values.tolist()
		else:
			self.asset_memory = [self.acc_balance]
			self.turbulence = 0
			self.cost = 0
			self.trades = 0
			#self.iteration=self.iteration
			self.rewards_memory = []
			#initiate state
			self.state = [self.acc_balance] + \
						  self.data.close.values.tolist() + \
						  [0] + \
						  self.data.macd.values.tolist() + \
					  	  self.data.rsi.values.tolist() + \
					  	  self.data.cci.values.tolist() + \
					  	  self.data.adx.values.tolist()
			
		return self.state
	
	def render(self, mode='human',close=False):
		return self.state
	

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
