import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# Plot Tweet count volume over period
data1_filename = 'data/interim/tweets_aggregated.csv'
print('Loading prep file: ', data1_filename)
df = pd.read_csv(data1_filename, usecols=['datetime'])
print('File has been loaded \n')

fig, ax = plt.subplots()
timeframe = '60min'
df.index = pd.DatetimeIndex(df['datetime'])
df = df.resample(timeframe).agg({'datetime': 'count'})
plt.plot(df.index, df['datetime'])
plt.ylabel('Tweet count')
plt.xlabel('Date')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator())

plt.title('Tweets volume over 15/04 - 30/04')
plt.savefig('../outputs/tweets_volume_' + timeframe + '.png', bbox_inches='tight')
plt.show()
plt.clf()
