import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
import os

input_filename = '../../data/processed/aggregated.csv'
output_filename = '../../reports/exploration.png'
df = pd.read_csv(input_filename, index_col=0)
print(df.head())

plot = sns.pairplot(df, vars=['compound', 'positive', 'negative', 'volume'], hue='price_change')
plot.savefig(output_filename)
