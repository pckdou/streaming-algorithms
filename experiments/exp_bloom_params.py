import math
import random
import time

from experiments.common import load_keys
from experiments.exp_main import write_csv
from src.bloom_filter import BloomFilter

SEED = 42
TRAIN_RATIO = 0.8
N_POS = 20_000
N_NEG = 100_000
RESULTS = "results"

BITS_LIST = [2, 4, 6, 8, 10, 14, 20]          # 실험 A
K_LIST = [1, 2, 3, 5, 7, 10, 14, 20]          # 실험 B
K_FIXED_BITS = [4, 10]                         # 실험 B: 고정할 비트/원소


def run_config(bits_per_elem, k, n, train_unique, positives, negatives):
    m = int(bits_per_elem * n)
    bf = BloomFilter(m, k)

    t0 = time.perf_counter()
    for x in train_unique:
        bf.add(x)
    insert_time = time.perf_counter() - t0

    fn = sum(1 for x in positives if x not in bf)
    fp = sum(1 for x in negatives if x in bf)
    return {
        "bits_per_elem": bits_per_elem, "m_bits": m, "k": k,
        "bloom_bytes": bf.size_bytes(),
        "false_negative": fn,
        "false_positive": fp,
        "fpr_measured": fp / len(negatives),
        "fpr_theory": BloomFilter.theoretical_fpr(n, m, k),
        "fill_ratio": bf.fill_ratio(),
        "insert_time_s": insert_time,
        "insert_throughput_per_s": n / insert_time,
    }


def main():
    stocks, pairs = load_keys()
    train_limit = int(len(pairs) * TRAIN_RATIO)
    train_unique = list(dict.fromkeys(pairs[:train_limit]))   # 순서 유지 + 중복 제거
    all_pairs = set(pairs)
    n = len(train_unique)
    print(f"레코드 {len(pairs)}, 삽입 고유 쌍 n={n}")

    rng = random.Random(SEED)
    positives = rng.sample(train_unique, N_POS)

    customers = sorted({p.split("_")[0] for p in all_pairs})
    stock_codes = sorted(set(stocks))
    negatives = set()
    while len(negatives) < N_NEG:
        key = f"{rng.choice(customers)}_{rng.choice(stock_codes)}"
        if key not in all_pairs:
            negatives.add(key)
    negatives = sorted(negatives)

    # 실험 A: 비트/원소 변화 (k는 최적값)
    print("\n[Bloom A] bits/elem 변화")
    rows_a = []
    for b in BITS_LIST:
        k = max(1, round(b * math.log(2)))
        row = run_config(b, k, n, train_unique, positives, negatives)
        rows_a.append(row)
        print(f"  b={b:2d} k={k:2d} FPR={row['fpr_measured']:.5f} "
              f"(이론 {row['fpr_theory']:.5f}) FN={row['false_negative']} "
              f"fill={row['fill_ratio']:.3f} bytes={row['bloom_bytes']}")
    write_csv(f"{RESULTS}/bloom_params_bits.csv", rows_a)

    # 실험 B: k 변화 (비트/원소 고정)
    print("\n[Bloom B] k 변화")
    rows_b = []
    for b in K_FIXED_BITS:
        for k in K_LIST:
            row = run_config(b, k, n, train_unique, positives, negatives)
            rows_b.append(row)
            print(f"  b={b:2d} k={k:2d} FPR={row['fpr_measured']:.5f} "
                  f"(이론 {row['fpr_theory']:.5f}) fill={row['fill_ratio']:.3f} "
                  f"time={row['insert_time_s']:.1f}s")
    write_csv(f"{RESULTS}/bloom_params_k.csv", rows_b)
    print("\nDONE")


if __name__ == "__main__":
    main()