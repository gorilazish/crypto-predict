import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

timeframe = '15min'

# Load dataset
data_filename = 'data/processed/tweets_prices_'+ timeframe +'.csv'
print('Loading prep file: ', data_filename)
df = pd.read_csv(data_filename, index_col=0)
print('File has been loaded \n')
print(df.shape)
print(df.describe())
