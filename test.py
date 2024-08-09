from datetime import datetime, timedelta, timezone

ny_time = (datetime.now(timezone.utc) - timedelta(hours=4)).time()
trading_hours = ["09:30:00", "16:00:00"]

if ny_time > datetime.strptime(trading_hours[1], "%H:%M:%S").time():
    print(True)