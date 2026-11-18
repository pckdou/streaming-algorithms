import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from collections import Counter

import numpy as np

from src.stream_reader import stream_retail
from src.bloom_filter import BloomFilter
from src.count_min_sketch import CountMinSketch

# ---------------- 설정 ----------------
PATH = "data/online_retail.csv"
RESULTS = "results"
SEED = 42
REPEAT = 3              # 시간 측정 반복 횟수
TRAIN_RATIO = 0.8       # Bloom: 앞 80%만 삽입
N_POS = 100_000         # Bloom: positive 테스트 수
N_NEG = 100_000         # Bloom: negative 테스트 수
BLOOM_TARGET_FPR = 0.01
TOPK_LIST = [10, 50, 100]

# 키 생성 함수 (인자: cust, stock)
def stock_key(cust, stock):
    return stock

def pair_key(cust, stock):
    return f"{cust}_{stock}"

CMS_CONFIGS = [
    # (이름, 키 함수, width, depth)
    ("stock", stock_key, 2000, 5),
    ("pair", pair_key, 50000, 5),
]


# ---------------- 공통 유틸 ----------------
def time_runs(fn, repeat=REPEAT):
    """fn을 repeat번 실행해 (마지막 결과, 시간 리스트) 반환"""
    times, result = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, times


def mem_current(fn):
    """fn이 만든 객체가 차지하는 메모리(bytes). 시간 측정과 분리해서 실행할 것"""
    tracemalloc.start()
    obj = fn()
    cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del obj
    return cur


def mean_std(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> saved {path}")


def save_env():
    env = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "seed": SEED,
        "repeat": REPEAT,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/env.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    print("[환경]", env)


# ---------------- 스트림 처리 함수들 ----------------
def count_records():
    """읽기만 수행 (기준 시간 측정용)"""
    n = 0
    for _ in stream_retail(PATH):
        n += 1
    return n


def build_set(limit):
    """Ground Truth: 앞 limit개 레코드의 쌍 set"""
    s = set()
    for i, (_inv, stock, cust, _c) in enumerate(stream_retail(PATH)):
        if i >= limit:
            break
        s.add(pair_key(cust, stock))
    return s


def build_counter(key_fn):
    """Ground Truth: 전체 레코드의 키별 정확한 빈도"""
    c = Counter()
    for _inv, stock, cust, _c in stream_retail(PATH):
        c[key_fn(cust, stock)] += 1
    return c


def build_bloom(m, k, limit):
    bf = BloomFilter(m, k)
    for i, (_inv, stock, cust, _c) in enumerate(stream_retail(PATH)):
        if i >= limit:
            break
        bf.add(pair_key(cust, stock))
    return bf


def build_cms(width, depth, key_fn):
    cms = CountMinSketch(width, depth)
    for _inv, stock, cust, _c in stream_retail(PATH):
        cms.add(key_fn(cust, stock))
    return cms


# ---------------- Bloom Filter 실험 ----------------
def run_bloom(n_total, train_limit, read_times):
    print("\n=== Bloom Filter ===")
    train_pairs = build_set(train_limit)
    all_pairs = set(build_counter(pair_key).keys())
    n_unique = len(train_pairs)

    customers = sorted({p.split("_")[0] for p in all_pairs})
    stocks = sorted({p.split("_", 1)[1] for p in all_pairs})

    rng = random.Random(SEED)
    positives = rng.sample(sorted(train_pairs), N_POS)

    negatives = set()
    while len(negatives) < N_NEG:
        key = f"{rng.choice(customers)}_{rng.choice(stocks)}"
        if key not in all_pairs:            # 한 번도 등장하지 않은 쌍만
            negatives.add(key)
    negatives = sorted(negatives)

    m, k = BloomFilter.optimal(n_unique, BLOOM_TARGET_FPR)
    print(f"삽입 대상 고유 쌍 n={n_unique}, m={m}, k={k}")

    # 시간: 삽입 (Bloom vs set)
    bf, bf_times = time_runs(lambda: build_bloom(m, k, train_limit))
    _, set_times = time_runs(lambda: build_set(train_limit))

    # 메모리
    set_bytes = mem_current(lambda: build_set(train_limit))
    bf_bytes = bf.size_bytes()

    # 정확도
    t0 = time.perf_counter()
    fn = sum(1 for x in positives if x not in bf)
    fp = sum(1 for x in negatives if x in bf)
    q_time = time.perf_counter() - t0
    n_query = len(positives) + len(negatives)

    bf_mean, bf_std = mean_std(bf_times)
    set_mean, set_std = mean_std(set_times)
    read_mean = statistics.mean(read_times) * TRAIN_RATIO   # 80%만 읽은 기준 시간 근사

    row = {
        "n_unique_inserted": n_unique,
        "m_bits": m, "k": k, "bits_per_elem": round(m / n_unique, 2),
        "false_negative": fn,
        "false_positive": fp,
        "fpr_measured": fp / len(negatives),
        "fpr_theory": BloomFilter.theoretical_fpr(n_unique, m, k),
        "fill_ratio": bf.fill_ratio(),
        "bloom_bytes": bf_bytes,
        "set_bytes": set_bytes,
        "memory_saving_pct": round((1 - bf_bytes / set_bytes) * 100, 2),
        "bloom_time_mean_s": bf_mean, "bloom_time_std_s": bf_std,
        "set_time_mean_s": set_mean, "set_time_std_s": set_std,
        "read_only_time_s": read_mean,
        "bloom_throughput_rec_s": train_limit / bf_mean,
        "set_throughput_rec_s": train_limit / set_mean,
        "query_throughput_per_s": n_query / q_time,
    }
    for key, v in row.items():
        print(f"  {key:26s}: {v}")
    write_csv(f"{RESULTS}/bloom_main.csv", [row])


# ---------------- CMS 실험 ----------------
def evaluate_cms(cms, truth, n_total):
    keys = list(truth.keys())
    true = np.array([truth[k] for k in keys], dtype=np.int64)
    est = np.array([cms.estimate(k) for k in keys], dtype=np.int64)
    err = est - true
    rel = err / true

    eps = math.e / cms.w
    delta = math.exp(-cms.d)
    bound = eps * n_total

    # Top-K 정확도 (동률은 키 이름순으로 고정)
    true_rank = sorted(range(len(keys)), key=lambda i: (-true[i], keys[i]))
    est_rank = sorted(range(len(keys)), key=lambda i: (-est[i], keys[i]))
    topk = {}
    for K in TOPK_LIST:
        if K <= len(keys):
            topk[f"top{K}_acc"] = len(set(true_rank[:K]) & set(est_rank[:K])) / K

    # 빈도 구간별 상대오차: 상위 10% / 중간 40% / 하위 50%
    n = len(keys)
    a, b = int(n * 0.1), int(n * 0.5)
    order = np.array(true_rank)
    tiers = {
        "rel_err_head10pct": rel[order[:a]].mean(),
        "rel_err_mid40pct": rel[order[a:b]].mean(),
        "rel_err_tail50pct": rel[order[b:]].mean(),
    }

    result = {
        "unique_keys": n,
        "epsilon": eps, "delta": delta, "error_bound_eps_N": bound,
        "under_estimate_count": int((err < 0).sum()),
        "mean_abs_err": float(np.abs(err).mean()),
        "max_abs_err": int(np.abs(err).max()),
        "mean_rel_err": float(rel.mean()),
        "bound_violation_ratio": float((err > bound).mean()),
        **topk, **tiers,
    }
    return result, keys, true, est


def run_cms(n_total):
    print("\n=== Count-Min Sketch ===")
    rows = []
    for name, key_fn, width, depth in CMS_CONFIGS:
        print(f"\n[{name}] width={width}, depth={depth}")
        truth = build_counter(key_fn)

        cms, cms_times = time_runs(lambda: build_cms(width, depth, key_fn))
        _, cnt_times = time_runs(lambda: build_counter(key_fn))

        counter_bytes = mem_current(lambda: build_counter(key_fn))
        cms_bytes = cms.size_bytes()

        metrics, keys, true, est = evaluate_cms(cms, truth, n_total)
        cm, cs = mean_std(cms_times)
        tm, ts = mean_std(cnt_times)

        row = {
            "target": name, "width": width, "depth": depth,
            "N_records": n_total,
            "cms_bytes": cms_bytes, "counter_bytes": counter_bytes,
            "memory_saving_pct": round((1 - cms_bytes / counter_bytes) * 100, 2),
            "cms_time_mean_s": cm, "cms_time_std_s": cs,
            "counter_time_mean_s": tm, "counter_time_std_s": ts,
            "cms_throughput_rec_s": n_total / cm,
            "counter_throughput_rec_s": n_total / tm,
            **metrics,
        }
        for key, v in row.items():
            print(f"  {key:26s}: {v}")
        rows.append(row)

        if name == "stock":   # 산점도용 (실제 vs 추정)
            write_csv(f"{RESULTS}/cms_stock_scatter.csv",
                      [{"key": k, "true": int(t), "est": int(e)}
                       for k, t, e in zip(keys, true, est)])

    write_csv(f"{RESULTS}/cms_main.csv", rows)


# ---------------- main ----------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    save_env()

    n_total, read_times = time_runs(count_records)
    train_limit = int(n_total * TRAIN_RATIO)
    print(f"\n전체 레코드 {n_total}, Bloom 삽입 구간 {train_limit}")
    print(f"읽기만 한 시간(평균): {statistics.mean(read_times):.2f}s")

    run_bloom(n_total, train_limit, read_times)
    run_cms(n_total)
    print("\nDONE")


if __name__ == "__main__":
    main()