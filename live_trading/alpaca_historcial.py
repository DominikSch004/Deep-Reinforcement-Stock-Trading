import requests
import json
import pandas as pd
import numpy as np

from config.config import *

from datetime import datetime

import alpaca_trade_api as tradeapi

api = tradeapi.REST(API_KEY, SECRET_KEY)


def get_data(tickers):
    for i in tickers:
        barset = api.get_barset(i, '1Min', limit=1000)
        bars = barset[i]
        df = pd.DataFrame(columns=["datadate", "tic", "close", "open", "high", "low", "volume"])
        # timestamp = str(bars[0].t)
        # timestamp = timestamp.split(" ")[0]
        for n in range(len(bars)):
            df = df.append({"datadate": pd.to_datetime(bars[n].t).to_pydatetime(), "tic": i, "close": bars[n].c, "open": bars[n].o, "high": bars[n].h, "low": bars[n].l, "volume": bars[n].v}, ignore_index=True)
        df.to_csv("data/historical/{}.txt".format(i))

get_data(["AXP", "AAPL", "VZ", "BA", "CAT", "JPM", "CVX", "KO", "DIS", "DD", "XOM", "HD", "INTC",
		"IBM", "JNJ", "MCD", "MRK", "MMM", "NKE", "PFE", "PG", "UNH", "RTX", "WMT", "WBA", "MSFT",
		"CSCO", "TRV", "GS", "V"])