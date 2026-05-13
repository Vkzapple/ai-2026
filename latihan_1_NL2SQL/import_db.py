import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv(r'student-performance_cleanv2.csv')

engine = create_engine(
    'mysql+pymysql://root:@localhost/studentlks_db'
)

df.to_sql(
    'students',
    con=engine,
    if_exists='replace',
    index=False
)

print(f'Berhasil import {len(df)} data ke MySQL')