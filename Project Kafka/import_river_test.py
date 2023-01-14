import pandas as pd

df = pd.read_csv('tweets-data.csv',delimiter=',',nrows=20)
print(df.iloc[:,0])
print(df.head())