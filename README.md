## Dataset
- Online Retail (UCI), 541,909 records (필터링 후 약 39만 건 사용)
- https://archive.ics.uci.edu/dataset/352/online
- `data/online_retail.xlsx` 로 저장 후 CSV 변환:
  `python src/convert_to_csv.py`
- 필터링: CustomerID 결측 제외, 취소 주문(InvoiceNo가 'C'로 시작) 제외