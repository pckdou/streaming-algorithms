import pandas as pd

SRC = "data/online_retail.xlsx"
DST = "data/online_retail.csv"

df = pd.read_excel(SRC, dtype=str)
df.to_csv(DST, index=False, encoding="utf-8")

print("rows:", len(df))
print("columns:", list(df.columns))