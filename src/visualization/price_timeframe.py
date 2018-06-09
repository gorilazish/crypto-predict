import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# Load dataset

data_filename = 'data/interim/tweets_aggregated.csv'
print('Loading prep file: ', data_filename)
df = pd.read_csv(data_filename, usecols=['timestamp', 'close'])
print('File has been loaded \n')

fig, ax = plt.subplots()
timeframe = '60min'
start_date = '2018-04-15'
end_date = '2018-04-30'
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.index = pd.DatetimeIndex(df['timestamp'])
df = df.resample(timeframe).agg({'close': 'last'})
mask = (df.index > start_date) & (df.index <= end_date)
df = df.loc[mask]
plt.plot(df.index, df['close'])
plt.ylabel('Price')
plt.xlabel('Date')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator())

plt.title('BTC price over 15/04 - 30/04')
plt.savefig('../outputs/btc_price_' + timeframe + '.png', bbox_inches='tight', dpi='figure')
plt.show()
plt.clf()
