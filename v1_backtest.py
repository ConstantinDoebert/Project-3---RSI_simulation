import requests as rq
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import json

# Enter desired ticker.
ticker = input("Enter your ticker: ")

# Loads api key from file for safety reasons.
with open('key.json') as f:
    key = json.load(f)["key"]


def get_ny_time():
    "Returns current New York date and time/UTC-4."
    ny_date = (datetime.now(timezone.utc) - timedelta(hours=4))
    ny_time = ny_date.time()
    return ny_date, ny_time


def get_price(ticker: str):
    """Pulls live price."""
    price_cache = {}

    price = rq.get(f"https://api.twelvedata.com/price?symbol={ticker}&apikey={key}")
    price_cache[f'price of {ticker}'] = float(price.json()["price"])
    price_cache["datetime"] = datetime.today()

    # Reutrns dict with price and time.
    return price_cache


# Backtests for approximately 100 trading days (7/5 correction).
len_backtest = 100
# Initial depost at t0. Currently 5000 USD.
'''Add EUR/USD exchange rate functionality?'''
cash = {(get_ny_time()[0] - timedelta(len_backtest * 7/5)).strftime("%Y-%m-%d %H:%M:%S"): 5000} 
"""Problem!: set global time, so its uniform for each iteration!"""
# Initial shares at t0.
depot = {(get_ny_time()[0] - timedelta(len_backtest * 7/5)).strftime("%Y-%m-%d %H:%M:%S"): 0}
total_value = {}
for key in cash:
    total_value[key] = cash[key] + depot[key] * 


def get_closing_prices(ticker: str, interval=len_backtest):
    """
    Get closing prices of recent trading days. Specified in var: len_backtests
    """
    date_today = datetime.today().date() # End date of time series.
    start_date = date_today - timedelta(np.ceil(interval * 7/5)) # Adjusting delta for trading days; 5 out of 7 days.

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

    # Pulls closing prices and corresponding date.
    closing_prices, dates = ([], []) 
    for value in response["values"]:
        dates.append(value["datetime"])
        closing_prices.append(float(value["close"]))


    # Fixes order to logical order: from top-to-bottom.
    df = pd.DataFrame(closing_prices, dates).iloc[::-1]
    df.rename(columns={0: f"Last {interval} days closing"}, inplace=True)


    return df, interval


def plot_values(ticker: str):
    '''Plots closing prices.'''
    df = get_closing_prices(ticker)
    plt.plot(df)
    plt.show()


def rsi(ticker: str):
    '''
    Calculates Relative-Strength-Indicator for given set of closing prices. See Wikipedia for detailed calculation.
    '''
    closing_prices = get_closing_prices(ticker)
    df = closing_prices[0]
    interval = closing_prices[1]

    df["U"] = 0.0
    df["D"] = 0.0


    for i in range(1, len(df)):
        # See: https://en.wikipedia.org/wiki/Relative_strength_index#Calculation.
        diff = df.iloc[i][f"Last {interval} days closing"] - df.iloc[i-1][f"Last {interval} days closing"]
        if diff > 0:
            df.at[df.index[i], "U"] = diff
        else:
            df.at[df.index[i], "D"] = abs(diff)


    # Calculates expontential moving average of sets "U" and "D".
    df["EMA_U"] = df["U"].ewm(span=14, adjust=False).mean() # Calculates exponential moving average.
    df["EMA_D"] = df["D"].ewm(span=14, adjust=False).mean()
    df["RS"] = df["EMA_U"] / df["EMA_D"]
    # Calculates normalized RSI.
    df["RSI"] = 100 - 100 / (1 + df["RS"]) # See: https://en.wikipedia.org/wiki/Relative_strength_index#Calculation.

    return df


def simulator(ticker: str, cash, depot, buy_cap=0.05, position_cap=0.1):
    """
    1. Checks if today is weekday, because trades are (currently) only on weekdays possible between 9.30 and 16.00 New York Time.
    2. Checks if its between 9.30 and 16.00 New York Time.
    3. Checks if (yesterday's) RSI is below 30 for buy or above 70 for sell. Not complete!!!
    4. Checks if there's enough cash.
    5. Buys as much full shares possible up to maximum 5% of cash avaiable, if the total position is not more than 10% of entire portfolio <- maintaining diversification.

    6. Add safety measure in case of high volatility. +/- 2 standard deviations from CBOE:VIX?
    """
    df = rsi(ticker)
    price = get_price(ticker)

    ny_datetime = get_ny_time()
    ny_date = ny_datetime[0].strftime("%Y-%m-%d")
    ny_time = ny_datetime[1]
    date_yesterday = datetime.today() - timedelta(1)
    trading_hours = ["09:30:00", "16:00:00"]

    is_weekday, is_intraday, buy = False, False, False
    date_bought, date_sold = ({}, {})

    # Checks if today is weekday.
    if date_yesterday.weekday() != 5 and date_yesterday.weekday() != 6:
        is_weekday = True
    else:
        is_weekday = False


    # Checks if its within NYSE trading hours.
    if ny_time > datetime.strptime(trading_hours[0], "%H:%M:%S").time() and ny_time < datetime.strptime(trading_hours[1], "%H:%M:%S").time():
        is_intraday = True
    else:
        is_intraday = False


    # Returns buy signal.
    if  is_weekday and is_intraday and df.at[date_yesterday.date().strftime('%Y-%m-%d'), "RSI"] < 30:
        if price[f"price of {ticker}"] / cash <= 0.05: # First buy cap safety measure.
            buy = True
        else:
            buy = False
    

    order_cap = np.floor(cash * buy_cap) # Initiates second buy cap safety measure.
    temp_cash = cash
    temp_depot = depot
    # Executes buy order.
    if buy and order_cap > price[f"price of {ticker}"]:
        temp_cash -= order_cap / price[f"price of {ticker}"] * price[f"price of {ticker}"]
        temp_depot += np.floor(order_cap / price[f"price of {ticker}"])

    # Checks third safety measure.
    '''Overrides cash and depot values, if new position value is not more than 10% of total portfolio'''
    try:
        if temp_depot * price[f"price of {ticker}"] <= position_cap * (temp_cash + temp_depot * price[f"price of {ticker}"]):
            cash = temp_cash
            depot = temp_depot
            date_bought[ny_date] = price[f"price of {ticker}"] # add count variables for buy, sell and general orders
        else:
            # Throws position cap warning.
            raise OrderPositionException

    except OrderPositionException:
        print(f"Exception occured: {ticker} would be +{position_cap * 100}% of portfolio.\nRisk of underdiversification!")
    

    # Checks and executes sell order.
    # All shares will be dumped.
    # Future adjustment possible for average down effect.
    if not buy and is_weekday and is_intraday and df.at[date_yesterday.date().strftime('%Y-%m-%d'), "RSI"] > 70:
        try:
            if depot <= 0:
                # Can't sell what you don't have :) (yet). Safety measure. Should never be thrown in 1st place (yet).
                raise CountError
            else:
                pass

            temp_depot -= depot
            temp_cash += depot * price[f"price of {ticker}"]
            date_sold[ny_date] = price[f"price of {ticker}"] # add count variables for buy, sell and general orders

            if temp_depot != 0:
                raise CountError
            else:
                pass

            depot = temp_depot
            cash = temp_cash
            
        
        except CountError:
            print("Expection occured: Didn't sell all shares or enters short position. No shorting possible.")
        
    
    return round(df.at[date_yesterday.date().strftime('%Y-%m-%d'), "RSI"], 6), price[f"price of {ticker}"], cash, depot, temp_cash, temp_depot


class OrderPositionException(Exception):
    "Raised when an order causes the position to be too much of total portfolio."
    pass


class CountError(Exception):
    "Raised when position enters short."
    pass



print(simulator(ticker, cash, depot))