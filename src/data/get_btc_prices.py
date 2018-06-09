import ccxt
import numpy as np
import math
import time
import datetime
import sys
import atexit

# HELPERS region
epoch = datetime.datetime.utcfromtimestamp(0)

def date_to_unix(dt):
  return int((dt - epoch).total_seconds())

def secs_to_min(seconds):
  return math.ceil(seconds / 60)

def string_to_date(string):
  return datetime.datetime.strptime(string, '%d/%m/%Y')

def timestamp_to_date(timestamp):
  return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

# ---------------------

current_time = int(round(time.time()))
exchange = ccxt.bitfinex()
symbol = 'BTC/USD'
from_timestamp = date_to_unix(string_to_date(sys.argv[1] if len(sys.argv) > 1 else '01/01/2014'))
to_timestamp = date_to_unix(string_to_date(sys.argv[2])) if len(sys.argv) > 2 else current_time
output_file = '../../data/raw/ohlcv.csv'
timeframe = '1m'
batch_limit = 1000

def batch_count():
  diff = secs_to_min(to_timestamp - from_timestamp)
  batch_count = math.ceil(diff / batch_limit)
  return batch_count

def get_historical_prices():

  count = batch_count()
  prices = []
  seconds_left = from_timestamp
  print('batch count: ', count)

  # Save before keyboard kill
  atexit.register(lambda: np.savetxt(output_file, prices, delimiter=','))
  
  for x in range(count):
    try:
      print('#', x, ' | ', x / count * 100, '%')
      print('Batch from ' + timestamp_to_date(seconds_left))
      batch = exchange.fetch_ohlcv(symbol, timeframe, seconds_left * 1000, batch_limit)
      prices = prices + batch
      print('Received date: ' + timestamp_to_date(batch[0][0] / 1000) + '\n')
      seconds_left += batch_limit * 60
      time.sleep(1)
    except Exception as ex:
      print(ex)
      print('WAITING PENALTY\n\n\n')
      time.sleep(120)
  
  return np.array(prices)

historical_prices = get_historical_prices()
np.savetxt(output_file, historical_prices, delimiter=',')
# np.savetxt('btc_ohlcv_' + str(timestamp_to_date(from_timestamp)) + ' - ' + str(timestamp_to_date(to_timestamp)) + '.csv', historical_prices, delimiter=',')
