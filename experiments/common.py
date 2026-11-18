from src.stream_reader import stream_retail

PATH = "data/online_retail.csv"


def load_keys():
    """스트림을 한 번 읽어 (stock 키 리스트, pair 키 리스트) 반환"""
    stocks, pairs = [], []
    for _inv, stock, cust, _c in stream_retail(PATH):
        stocks.append(stock)
        pairs.append(f"{cust}_{stock}")
    return stocks, pairs