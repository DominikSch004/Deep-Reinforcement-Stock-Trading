
import pandas as pd
from datetime import datetime
from pandas.core.frame import DataFrame
import websocket
import json
from config.config import *
from create_done_data import *
import threading


def get_live_data():

	# tickers = ["AM.AXP", "AM.AAPL", "AM.VZ", "AM.BA", "AM.CAT", "AM.JPM", "AM.CVX", "AM.KO", "AM.DIS", "AM.DD", "AM.XOM", "AM.HD", "AM.INTC",
	# 			"AM.IBM", "AM.JNJ", "AM.MCD", "AM.MRK", "AM.MMM", "AM.NKE", "AM.PFE", "AM.PG", "AM.UNH", "AM.RTX", "AM.WMT", "AM.WBA", "AM.MSFT",
	# 			"AM.CSCO", "AM.TRV", "AM.GS", "AM.V"]


	tickers = ["AM.AAPL"]

	#df = pd.DataFrame(columns=['datadate', 'tic', 'close', 'open', 'high', 'low', 'volume'])
	
	def on_open(ws):
				
			auth_data = {
			"action": "authenticate",
			"data": {
				"key_id": API_KEY,
				"secret_key": SECRET_KEY
				}
			}

			ws.send(json.dumps(auth_data))

			channel_data = {
				"action": "listen",
				"data": {
					"streams": tickers
					}
					
				}	

			ws.send(json.dumps(channel_data))


	def on_message(ws, message):
			
			message = json.loads(message)
			#print(pd.to_datetime(message["data"]["s"], unit='ms').to_pydatetime())
			print(message)
			#try:
			d = {'datadate': [pd.to_datetime(message["data"]["s"], unit='ms').to_pydatetime()], 'tic': [message["data"]["T"]], 'close': [message["data"]["c"]], 'open': [message["data"]["o"]], 'high': [message["data"]["h"]], 'low': [message["data"]["l"]], 'volume': [message["data"]["v"]]}

			# df = pd.DataFrame(columns=['datadate', 'tic', 'close', 'open', 'high', 'low', 'volume'], data=pd.Series([str(datetime.fromtimestamp(message["data"]["streams"]["s"])), message["data"]["streams"]["T"],
			# 						message["data"]["streams"]["c"], message["data"]["streams"]["o"], message["data"]["streams"]["h"], message["data"]["streams"]["l"], message["data"]["streams"]["v"]]))
		
			df = pd.DataFrame(columns=['datadate', 'tic', 'close', 'open', 'high', 'low', 'volume'], data=d)
			#print(df)
			#print(message["data"]["T"])
			preprocessed_path = "data/live/{}.txt".format(str(message["data"]["T"]))
			if os.path.exists(preprocessed_path):
				try:
					df.to_csv("data/live/{}.txt".format(str(message["data"]["T"])), mode='a', header=False, index=False)
				except Exception as e:
					print(e)
			else:
				df.to_csv("data/live/{}.txt".format(str(message["data"]["T"])), index=False)

			run_model()

	#socket = "wss://alpaca.socket.polygon.io/stocks"
	socket = "wss://data.alpaca.markets/stream"

	ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message)
	ws.run_forever()

if __name__ == '__main__':
		t = threading.Thread(target=get_live_data(), daemon=True)
		t.start()