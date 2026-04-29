import os
import pandas as pd
import sqlite3

db_path = 'dataset/Ridership_v1.sqlite'
if os.path.exists(db_path):
    os.remove(db_path)

df = pd.read_csv('dataset/Ridership_v1.csv')
with sqlite3.connect(db_path) as conn:
    df.to_sql('MBTA_Ridership', conn, if_exists='replace', index=False)
print('Done, rows:', len(df))
