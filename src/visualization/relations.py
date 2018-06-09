import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset

data_filename = 'data/interim/tweets_aggregated.csv'
print('Loading prep file: ', data_filename)
df = pd.read_csv(data_filename)
print('File has been loaded \n')

fig, ax = plt.subplots()

feature = 'volume'
ax.scatter(df['price_change'], df[feature])
plt.xlabel('Price change')
plt.ylabel('Followers * Compound [60min]')

plt.show()
plt.clf()
