import pandas as pd
import numpy as np
import os

from datetime import datetime
from run_trading import run_model

def create_done_data():
	#rel_path = "data/historical/"
	rel_path = "data/live/"
	done_data_path = "data/done_data2.csv"

	for file in os.listdir(rel_path):
		filename = os.fsdecode(file)
		#fileh = open(str(path)+str(filename), mode='r')
		df = pd.read_csv(str(rel_path)+str(filename))
		df = df.drop(axis=1, labels="Unnamed: 0", errors="ignore")
		if os.path.exists(done_data_path):
			try:
				df.to_csv(str(done_data_path), mode='a', header=False, index=False)
			except Exception as e:
				print(e)
				#f.close()
		else:
			#df = df.drop(axis=1, columns="Unnamed: 0", errors="ignore")
			df.to_csv(done_data_path, index=False)

	data = pd.read_csv("data/done_data2.csv")
	run_model(data)
	os.remove(done_data_path)
# df = pd.read_csv(str(done_data_path))
# df = df.reset_index(inplace=True)
# print(df)

# if __name__ == '__main__':
# 		try:
# 			create_done_data()
# 			df = pd.read_csv("data/done_data2.csv")
# 			df = df.reset_index(drop=True)
# 			run_model(df)
# 			os.remove("data/done_data2.csv")
# 		except Exception:
# 			os.remove("data/done_data2.csv")
