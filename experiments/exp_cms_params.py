import time

from experiments.common import load_keys
from experiments.exp_main import evaluate_cms, write_csv
from src.count_min_sketch import CountMinSketch
from collections import Counter

RESULTS = "results"
BENCH_N = 20_000          # add() 처리량 측정용 호출 수

WIDTHS_STOCK = [50, 100, 250, 500, 1000, 2000, 5000, 10000]
DEPTHS_STOCK = [1, 2, 3, 5, 8, 12]
EQUAL_MEM = [(5000, 1), (2500, 2), (1000, 5), (500, 10), (250, 20)]
WIDTHS_PAIR = [5000, 20000, 50000, 200000, 500000]


def build_from_counter(width, depth, truth):
    """키별 총 빈도를 한 번에 add. 레코드 단위 add와 최종 테이블이 동일(선형성)"""
    cms = CountMinSketch(width, depth)
    for key, c in truth.items():
        cms.add(key, c)
    return cms


def bench_add(width, depth, sample_keys):
    cms = CountMinSketch(width, depth)
    t0 = time.perf_counter()
    for x in sample_keys:
        cms.add(x)
    return len(sample_keys) / (time.perf_counter() - t0)


def run(label, truth, n_total, sample_keys, configs, out_name, depth_fixed=None):
    print(f"\n[{label}]")
    rows = []
    for w, d in configs:
        cms = build_from_counter(w, d, truth)
        metrics, _k, _t, _e = evaluate_cms(cms, truth, n_total)
        row = {
            "width": w, "depth": d, "cms_bytes": cms.size_bytes(),
            "add_ops_per_s": bench_add(w, d, sample_keys),
            **metrics,
        }
        rows.append(row)
        print(f"  w={w:6d} d={d:2d} bytes={row['cms_bytes']:8d} "
              f"meanRel={row['mean_rel_err']:.4f} head={row['rel_err_head10pct']:.4f} "
              f"tail={row['rel_err_tail50pct']:.4f} maxAbs={row['max_abs_err']} "
              f"top10={row.get('top10_acc')} add/s={row['add_ops_per_s']:.0f}")
    write_csv(f"{RESULTS}/{out_name}", rows)


def main():
    stocks, pairs = load_keys()
    n_total = len(stocks)
    truth_stock = Counter(stocks)
    truth_pair = Counter(pairs)
    sample_stock = stocks[:BENCH_N]
    sample_pair = pairs[:BENCH_N]
    print(f"N={n_total}, 고유 stock={len(truth_stock)}, 고유 pair={len(truth_pair)}")

    run("CMS A: width 변화 (stock, depth=5)", truth_stock, n_total, sample_stock,
        [(w, 5) for w in WIDTHS_STOCK], "cms_params_width.csv")
    run("CMS B: depth 변화 (stock, width=2000)", truth_stock, n_total, sample_stock,
        [(2000, d) for d in DEPTHS_STOCK], "cms_params_depth.csv")
    run("CMS C: 동일 메모리 w*d=5000 (stock)", truth_stock, n_total, sample_stock,
        EQUAL_MEM, "cms_params_equalmem.csv")
    run("CMS D: width 변화 (pair, depth=5)", truth_pair, n_total, sample_pair,
        [(w, 5) for w in WIDTHS_PAIR], "cms_params_pair_width.csv")
    print("\nDONE")


if __name__ == "__main__":
    main()