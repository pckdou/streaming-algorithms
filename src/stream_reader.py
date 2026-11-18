import csv

def stream_retail(path, drop_missing_customer=True, drop_cancel=True):
    """CSV를 한 줄씩 yield. 전체 데이터를 메모리에 올리지 않는다."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inv = row["InvoiceNo"].strip()
            cust = row["CustomerID"].strip()
            if drop_missing_customer and cust in ("", "nan"):
                continue
            if drop_cancel and inv.startswith("C"):
                continue
            cust = cust.split(".")[0]          # "17850.0" -> "17850"
            yield inv, row["StockCode"].strip(), cust, row["Country"].strip()