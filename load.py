import pandas as pd
import sqlite3

df = pd.read_csv('dataset/Ridership_v1.csv')
with sqlite3.connect('dataset/Ridership_v1.sqlite') as conn:
    df.to_sql('MBTA_Ridership', conn, if_exists='replace', index=False)
print('Done, rows:', len(df))
