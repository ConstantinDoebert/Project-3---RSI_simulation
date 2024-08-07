import requests as rq
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

key = ""


def get_closing_prices(ticker: str, interval=30):
    """
    Get closing prices of recent 30 days.
    """
    date_today = datetime.today().date() # end date of time series
    start_date = date_today - timedelta(np.ceil(interval * 7/5)) # adjusting delta for trading days; 5 out of 7 days

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
    df.rename(columns={0: f"Last {interval} days closing"}, inplace=True)

    return df, interval


def plot_values(ticker: str):
    df = get_closing_prices(ticker)
    plt.plot(df)
    plt.show()


def rsi(ticker: str):
    closing_prices = get_closing_prices(ticker)
    df = closing_prices[0]
    interval = closing_prices[1]

    df["U"] = 0
    df["D"] = 0

    for i in range(1, len(df)):
        # # see: hhttps://en.wikipedia.org/wiki/Relative_strength_index#Calculation
        diff = df.iloc[i][f"Last {interval} days closing"] - df.iloc[i-1][f"Last {interval} days closing"]
        if diff > 0:
            df.at[df.index[i], "U"] = diff
        else:
            df.at[df.index[i], "D"] = abs(diff)

    df["EMA_U"] = df["U"].ewm(span=14, adjust=False).mean() # calculate exponential moving average
    df["EMA_D"] = df["D"].ewm(span=14, adjust=False).mean()
    df["RS"] = df["EMA_U"] / df["EMA_D"]
    df["RSI"] = 100 - 100 / (1 + df["RS"]) # see: hhttps://en.wikipedia.org/wiki/Relative_strength_index#Calculation

    return df

print(rsi("NVDA"))