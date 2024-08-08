import requests as rq
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def get_price(ticker: str):
    """Pulls live price."""
    price_cache = {}
    price = rq.get(f"https://api.twelvedata.com/price?symbol={ticker}&apikey={key}")
    price_cache[f'price of {ticker}'] = float(price.json()["price"])
    price_cache["datetime"] = datetime.today()

    return price_cache

key = ""
ticker = "NVDA"
cash = 5000 # amount of cash avaiable at t=0
depot = 0 # number of shares at t=0
total_value = cash + depot * float(get_price(ticker)[f"price of {ticker}"])

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

    df["U"] = 0.0
    df["D"] = 0.0

    for i in range(1, len(df)):
        # # see: https://en.wikipedia.org/wiki/Relative_strength_index#Calculation
        diff = df.iloc[i][f"Last {interval} days closing"] - df.iloc[i-1][f"Last {interval} days closing"]
        if diff > 0:
            df.at[df.index[i], "U"] = diff
        else:
            df.at[df.index[i], "D"] = abs(diff)

    df["EMA_U"] = df["U"].ewm(span=14, adjust=False).mean() # calculate exponential moving average
    df["EMA_D"] = df["D"].ewm(span=14, adjust=False).mean()
    df["RS"] = df["EMA_U"] / df["EMA_D"]
    df["RSI"] = 100 - 100 / (1 + df["RS"]) # see: https://en.wikipedia.org/wiki/Relative_strength_index#Calculation


    return df


class PositionException(Exception):
    "Raised when an order causes the position to be too much of total portfolio"
    pass


def simulator(ticker: str, cash, depot, buy_cap=0.05, position_cap=0.1):
    """
    1. Checks if today is weekday, because trades are (currently) only on weekdays possible between 9.30 and 16.00 New York Time.
    2. Checks if its between 9.30 and 16.00 New York Time. Not done!!!
    3. Checks if (yesterday's) RSI is below 30 for buy or above 70 for sell. Not complete!!!
    4. Checks if there's enough cash.
    5. Buys as much full shares possible up to maximum 5% of cash avaiable, if the total position is not more than 10% of entire portfolio <- maintaining diversification.
    """
    df = rsi(ticker)
    price = get_price(ticker)
    date_yesterday= datetime.today() - timedelta(1)
    is_weekday = False
    buy = False

    if date_yesterday.weekday() != 5 and date_yesterday.weekday() != 6:
        is_weekday = True

    if df.at[date_yesterday.date().strftime('%Y-%m-%d'), "RSI"] <= 30 and is_weekday:
        if price[f"price of {ticker}"] / cash <= 0.05:
            buy = True
        else:
            buy = False
    
    order_cap = np.floor(cash * buy_cap)
    temp_cash = cash
    temp_depot = depot
    if buy and order_cap > price[f"price of {ticker}"]:
        temp_cash -= order_cap / price[f"price of {ticker}"] * price[f"price of {ticker}"]
        temp_depot += np.floor(order_cap / price[f"price of {ticker}"])

    '''Overrides cash and depot values, if new position value is not more than 10% of total portfolio'''
    try:
        if temp_depot * price[f"price of {ticker}"] <= position_cap * (temp_cash + temp_depot * price[f"price of {ticker}"]):
            cash = temp_cash
            depot = temp_depot
        else:
            raise PositionException

    except PositionException:
        print(f"Exception occured: {ticker} would be +{position_cap * 100}% of portfolio. \n Risk of underdiversification!")
    

    return cash, depot

print(simulator(ticker, cash, depot))