import pandas as pd
import numpy as np
import requests as rq
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import exchange_rate_converter as conv




# Enter desired ticker.
ticker = "NVDA"

# Loads api key from file for safety reasons.
with open('key.json') as f:
    key = json.load(f)["key"]


def get_ny_time():
    "Returns current New York date and time/UTC-4."
    ny_date = (datetime.now(timezone.utc) - timedelta(hours=4))
    ny_time = ny_date.time()
    return ny_date, ny_time


ny_datetime = get_ny_time()


def get_price(ticker: str):
    '''Pulls live price. Assumption: programm executes fast enough, so live price is very close to ny_datetime called above.'''
    price_cache = {}

    price = rq.get(f"https://api.twelvedata.com/price?symbol={ticker}&apikey={key}")
    price_cache[f'price of {ticker}'] = float(price.json()["price"])
    price_cache["datetime"] = ny_datetime[0].strftime("%Y-%m-%d %H:%M:%S")

    # Reutrns dict with price and time.
    return price_cache


current_price = get_price(ticker)


# Backtests for approximately 100 trading days (7/5 correction).
len_backtest = 100



# Converts total value of portfolio to desired currency, USD to EUR by default.
def get_converted_portfolio(total_value):
    total_value_converted = {}
    for key in total_value:
        total_value_converted[key] = conv.convert_currency(total_value[key])

    return total_value_converted



def get_closing_prices(ticker: str, interval=len_backtest):
    """
    Get closing prices of recent trading days. Specified in var: len_backtests
    """
    date_today = ny_datetime[0].date() # End date of time series.
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
    df.rename(columns={0: f"Last {len(df.index)} days closing"}, inplace=True)


    return df, interval



def get_intraday_prices(ticker: str, interval=len_backtest):
    """
    Get intraday prices of recent trading days. Specified in var: len_backtests.\n
    Average
    """
    date_today = ny_datetime[0].date() # End date of time series.
    start_date = date_today - timedelta(np.ceil(interval * 7/5)) # Adjusting delta for trading days; 5 out of 7 days.

    response = rq.get(
            f"https://api.twelvedata.com/time_series",
            params={
                "symbol": ticker,
                "interval": "1h",
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


    # Fixes order to logical order: from top-to-bottom.
    df = pd.DataFrame(closing_prices, dates).iloc[::-1]
    df.rename(columns={0: f"Last {len(df.index)} full hour prices."}, inplace=True)


    return df



def calc_intra_average(ticker: str):
    df = get_intraday_prices(ticker)

    df.index = pd.to_datetime(df.index)
    df_daily_mean = df.groupby(df.index.date).mean()
    df_daily_mean.index = pd.to_datetime(df_daily_mean.index)
    df_daily_mean.columns = ['Mean of trading days']

    return df_daily_mean



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
        diff = df.iloc[i][f"Last {len(df.index)} days closing"] - df.iloc[i-1][f"Last {len(df.index)} days closing"]
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



def simulator(ticker: str, buy_cap=0.05, position_cap=0.1):
    """
    - Add safety measure in case of high volatility. +/- 2 standard deviations from CBOE:VIX?
    """
    df_rsi = rsi(ticker)
    df_rsi.index = pd.to_datetime(df_rsi.index)
    df_intraday = calc_intra_average(ticker)
    df_merged = pd.DataFrame()

    
    try:
        # Check if indices are equal
        if df_rsi.index.equals(df_intraday.index):
            # Merge dataframes
            df_merged = pd.concat([df_rsi, df_intraday], axis=1)
        else:
            raise UnequalIndexError
    
    except UnequalIndexError:
        print("Please adjust indices.")


    df_merged['Buy'] = (df_merged['RSI'] >= 10) & (df_merged['RSI'] <= 30)
    df_merged['Sell'] = df_merged['RSI'] > 70

    buy_order_counter = 0
    sell_order_counter = 0
    general_order_counter = 0

    # Create empty lists to hold the order values
    buy_orders = []
    sell_orders = []
    general_orders = []

    # Iterate through each row in the DataFrame
    for i, row in df_merged.iterrows():
        # Handle Buy order
        if row['Buy']:
            buy_order_counter += 1
            buy_orders.append(buy_order_counter)
        else:
            buy_orders.append(np.nan)
        
        # Handle Sell order
        if row['Sell']:
            sell_order_counter += 1
            sell_orders.append(sell_order_counter)
        else:
            sell_orders.append(np.nan)
        
        # Handle General order
        if row['Buy'] or row['Sell']:
            general_order_counter += 1
            general_orders.append(general_order_counter)
        else:
            general_orders.append(np.nan)

    # Add the new columns to the DataFrame
    df_merged['Buy order'] = buy_orders
    df_merged['Sell order'] = sell_orders
    df_merged['General order'] = general_orders

    df_merged['cash'] = np.nan
    df_merged['depot'] = np.nan
    df_merged['portfolio'] = np.nan

    # Set initial values for cash and depot starting from the 11th row (index 10)
    df_merged.at[df_merged.index[10], 'cash'] = 100000
    df_merged.at[df_merged.index[10], 'depot'] = 0

    # Iterate over the rows starting from the 12th row
    for i in range(11, len(df_merged)):
        previous_row = df_merged.iloc[i - 1]
        current_row = df_merged.iloc[i]

        if current_row['Buy']:
            # Calculate how many times Mean of trading days fits into the cash value
            units_to_buy = np.floor(previous_row['cash'] / current_row['Mean of trading days'])

            # Update depot with the number of units bought
            df_merged.at[df_merged.index[i], 'depot'] = previous_row['depot'] + units_to_buy

            # Update cash after buying
            df_merged.at[df_merged.index[i], 'cash'] = previous_row['cash'] - (units_to_buy * current_row['Mean of trading days'])

        elif current_row['Sell']:
            # Calculate cash after selling all depot units
            cash_after_sell = previous_row['cash'] + (previous_row['depot'] * current_row['Mean of trading days'])

            # Update cash
            df_merged.at[df_merged.index[i], 'cash'] = cash_after_sell

            # Update depot to 0 after selling
            df_merged.at[df_merged.index[i], 'depot'] = 0

        else:
            # If neither Buy nor Sell, carry forward the previous cash and depot values
            df_merged.at[df_merged.index[i], 'cash'] = previous_row['cash']
            df_merged.at[df_merged.index[i], 'depot'] = previous_row['depot']

        # Update the portfolio value (cash + depot * Mean of trading days)
        df_merged.at[df_merged.index[i], 'portfolio'] = df_merged.at[df_merged.index[i], 'cash'] + (df_merged.at[df_merged.index[i], 'depot'] * current_row['Mean of trading days'])


    return df_merged



def plot_mean_and_portfolio(ticker: str):
    df = simulator(ticker)
    # Create a plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot "Portfolio" on the first y-axis
    ax1.plot(df.index, df['portfolio'], color='blue', label='Portfolio')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis to plot "Mean of trading days"
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Mean of trading days'], color='green', label='Mean of trading days')
    ax2.set_ylabel('Mean of trading days', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Adding title and grid
    plt.title('Portfolio and Mean of Trading Days Over Time')
    ax1.grid(True)

    # Show plot
    plt.show()



plot_mean_and_portfolio(ticker)



class UnequalIndexError(Exception):
    "Raised when to dataframe indices don't match."
    pass



class OrderPositionException(Exception):
    "Raised when an order causes the position to be too much of total portfolio."
    pass



class CountError(Exception):
    "Raised when position enters short."
    pass

