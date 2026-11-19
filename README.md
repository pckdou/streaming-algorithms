# Streaming Algorithms: Bloom Filter & Count-Min Sketch

대용량 스트림 환경에서 전체 데이터를 저장하지 않고 근사 계산을 수행하는 두 알고리즘을 직접 구현하고, 정확도·메모리·처리시간 trade-off를 분석한 프로젝트입니다.

## Dataset
- Online Retail (UCI), 원본 541,909건 → 필터링 후 **397,924건** 사용
- https://archive.ics.uci.edu/dataset/352/online
- 필터링: CustomerID 결측 제외, 취소 주문(InvoiceNo가 `C`로 시작) 제외
- 고유 고객 4,339 / 고유 상품 3,665 / 고유 (고객,상품) 쌍 266,802

## Algorithms
| 알고리즘 | 대상 키 | 정답(Ground Truth) |
|---|---|---|
| Bloom Filter | `고객_상품` 쌍 (구매 이력 포함 여부) | `set` |
| Count-Min Sketch | `StockCode`, `고객_상품` 쌍 (빈도) | `Counter` |

## Setup
```bash
git clone https://github.com/pckdou/streaming-algorithms.git
cd streaming-algorithms
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation
1. https://archive.ics.uci.edu/dataset/352/online 에서 zip 다운로드
2. `Online Retail.xlsx`를 `data/online_retail.xlsx`로 저장
3. CSV 변환 (1회 전처리): `python src/convert_to_csv.py`

`data/`는 용량 문제로 git에 포함하지 않습니다.

## Reproduce
```bash
python -m experiments.test_basic          # 구현 정확성 검증 (FN=0, 과소추정=0)
python -m experiments.check_data          # 데이터 통계
python -m experiments.exp_main            # 기본 실험 (정확도/메모리/시간)
python -m experiments.exp_bloom_params    # Bloom 파라미터 실험
python -m experiments.exp_cms_params      # CMS 파라미터 실험
python -m experiments.plot_results        # 그래프 생성
```

## Structure
```
src/
  bloom_filter.py        Bloom Filter 구현
  count_min_sketch.py    Count-Min Sketch 구현
  stream_reader.py       CSV 제너레이터 스트림 리더
  utils.py               시드 기반 해시 함수 (MD5)
  convert_to_csv.py      xlsx -> csv 전처리
experiments/
  test_basic.py, check_data.py, exp_main.py,
  exp_bloom_params.py, exp_cms_params.py, plot_results.py, common.py
results/                 결과 CSV, env.json, figures/ (fig01~fig11)
```

## Implementation notes
- 자료구조와 연산은 직접 구현, 해시만 표준 라이브러리 `hashlib` MD5 사용 (seed를 접두어로 붙여 k개의 해시 함수 생성)
- 기본 실험은 CSV를 제너레이터로 한 줄씩 처리 (전체를 메모리에 올리지 않음)
- 파라미터 실험은 정확도 분석이 목적이라 키 리스트를 미리 로드, CMS는 선형성을 이용해 키별 총 빈도로 add (최종 테이블은 레코드 단위 add와 동일)
- 시간·처리량은 `exp_main`의 3회 반복 평균 ± 표준편차 사용
- 모든 실험 seed=42

## Key results
| 항목 | 결과 |
|---|---|
| Bloom (m=2.1M bits, k=7) | FN=0, FPR 0.99% (이론 1.00%), set 대비 메모리 98.66% 절감 |
| CMS stock (w=2000, d=5) | 과소추정 0, 상위 10% 상대오차 3.1%, Counter 대비 메모리 71.85% 절감 |
| CMS pair (w=50000, d=5) | 과소추정 0, Counter 대비 메모리 90.73% 절감 |

## Environment
Windows 11, 11th Gen Intel Core i7-1165G7 @ 2.80GHz, Python 3.14.3, numpy 2.5.5 (`results/env.json`)