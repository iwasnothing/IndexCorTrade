import os
from google.cloud import secretmanager_v1
from google.cloud import storage
from google.cloud import pubsub_v1
from google.cloud import bigquery
import alpaca_trade_api as tradeapi

PRJID="139391369285"
def init_vars():
    client = secretmanager_v1.SecretManagerServiceClient()
    secrets = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "NEWS_API_KEY"]
    result = {"APCA_API_BASE_URL": "https://paper-api.alpaca.markets" }
    for s in secrets:
        name = f"projects/{PRJID}/secrets/{s}/versions/latest"
        response = client.access_secret_version(request={'name': name})
        print(response)
        os.environ[s] = response.payload.data.decode('UTF-8')
        result[s] = response.payload.data.decode('UTF-8')
    print(result)
    return result

def place_order(ticker,spread):
    key = init_vars()
    api = tradeapi.REST(key['APCA_API_KEY_ID'], key['APCA_API_SECRET_KEY'], key['APCA_API_BASE_URL'], 'v2')
    q=api.get_last_quote(ticker)
    ticker_price = q.bidprice
    print("place order for ",ticker,ticker_price)

    # We could buy a position and add a stop-loss and a take-profit of 5 %
    try:
        r = api.submit_order(
            symbol=ticker,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            stop_loss={'stop_price': ticker_price * (1 - spread),
                        'limit_price': ticker_price * (1 - spread) * 0.95},
            take_profit={'limit_price': ticker_price * (1 + spread)}
        )
        print("place order returned ",r)
    except Exception as e:
        print(e)

#
# MAIN
#
place_order('QLD',0.1)