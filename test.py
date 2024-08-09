from datetime import datetime, timedelta, timezone
from IPython.display import display, HTML
import pandas as pd
import json

ny_time = (datetime.now(timezone.utc) - timedelta(hours=4)).time()
trading_hours = ["09:30:00", "16:00:00"]

if ny_time > datetime.strptime(trading_hours[1], "%H:%M:%S").time():
    print(True)


with open('key.json') as f:
    key = json.load(f)["key"]
print(key)


def get_ny_time():
    ny_date = (datetime.now(timezone.utc) - timedelta(hours=4))
    ny_time = ny_date.time()
    return ny_date, ny_time

cash = {(get_ny_time()[0] - timedelta(100 * 7/5)).strftime("%Y-%m-%d %H:%M:%S"): 5000}
print(cash)