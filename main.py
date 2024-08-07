import requests as rq
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

key = ""


def get_closing_prices(ticker: str, interval=14):
    """
    Get closing prices of recent 14 days.
    """
    date_today = datetime.today().date() # end date of time series
    start_date = date_today - timedelta(np.ceil(interval * 7/5))

    response = rq.get(
            f"https://api.twelvedata.com/time_series",
            params={
                "symbol": ticker,
                "interval": "1day",
                "apikey": key,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": date_today.strftime("%Y-%m-%d"),
            }
        )
        
    response = response.json()

    closing_prices, dates = ([], []) 
    for value in response["values"]:
        dates.append(value["datetime"])
        closing_prices.append(float(value["close"]))

    df = pd.DataFrame(closing_prices, dates).iloc[::-1]
    df.rename(columns={0: "Last 14 days closing"}, inplace=True)

    return df


def plot_values(ticker: str):
    df = get_closing_prices(ticker)
    plt.plot(df)
    plt.show()



def rsi(ticker: str):
    df = get_closing_prices(ticker)

    for value in df["Last 14 days closing"]:
        u, d = ([], [])
        u.append(value)


    return u

print(rsi("NVDA"))