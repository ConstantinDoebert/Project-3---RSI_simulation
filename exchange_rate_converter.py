import requests as rq
import json

# Loads api key from file for safety reasons.
with open('key.json') as f:
    key = json.load(f)["key"]


def convert_currency(amount: float, currency_from="USD", currency_to="EUR") -> float:
    '''
    Converts monetary value to desired currency value. By default USD to EUR.\n
    Import currency codes:\n
        USD - U.S. Dollar\n
        EUR - Euro\n
        GBP - Great Britain pound (sterling)\n
        JPY - apanese yen\n
        CHF - Swiss Franc\n
        AUD - Australian dollar\n
        CAD - Canadian dollar\n
        CNY - China yuan renminbi\n
        NZD - New Zealand dollar\n
        INR - Indian rupee\n
        BZR - Brazilian real\n
        SEK - Swedish krona\n
        ZAR - South African rand\n
        HKD - Hong Kong dollar\n
    '''
    info = rq.get(f"https://api.twelvedata.com/currency_conversion?symbol={currency_from}/{currency_to}&amount={amount}&apikey={key}").json()
    converted_amount = info['amount']

    return converted_amount