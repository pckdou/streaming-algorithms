from collections import Counter
from src.stream_reader import stream_retail

PATH = "data/online_retail.csv"

n = 0
pairs = set()
stock_cnt = Counter()
customers = set()

for inv, stock, cust, country in stream_retail(PATH):
    n += 1
    pairs.add(f"{cust}_{stock}")
    stock_cnt[stock] += 1
    customers.add(cust)

print("필터링 후 레코드 수      :", n)
print("고유 고객 수             :", len(customers))
print("고유 상품(StockCode) 수  :", len(stock_cnt))
print("고유 (고객,상품) 쌍 수   :", len(pairs))
print("Top 5 상품               :", stock_cnt.most_common(5))