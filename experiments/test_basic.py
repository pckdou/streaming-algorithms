import random
from collections import Counter
from src.bloom_filter import BloomFilter
from src.count_min_sketch import CountMinSketch


def test_bloom():
    random.seed(42)
    n = 10000
    m, k = BloomFilter.optimal(n, 0.01)
    bf = BloomFilter(m, k)

    items = [f"item_{i}" for i in range(n)]
    for x in items:
        bf.add(x)

    # 1) False Negative는 0이어야 함
    fn = sum(1 for x in items if x not in bf)
    assert fn == 0, f"False Negative 발생: {fn}"

    # 2) FPR이 목표(1%) 근처인지
    negatives = [f"other_{i}" for i in range(20000)]
    fp = sum(1 for x in negatives if x in bf)
    fpr = fp / len(negatives)
    print(f"[Bloom] m={m}, k={k}, FN={fn}, 실측FPR={fpr:.4f}, 이론FPR={BloomFilter.theoretical_fpr(n, m, k):.4f}")
    assert fpr < 0.02


def test_cms():
    random.seed(42)
    cms = CountMinSketch(width=500, depth=5)
    truth = Counter()

    for _ in range(50000):
        x = f"key_{int(random.paretovariate(1.2))}"
        cms.add(x)
        truth[x] += 1

    # 항상 과대추정(est >= true)이어야 함
    under = sum(1 for k, v in truth.items() if cms.estimate(k) < v)
    assert under == 0, f"과소추정 발생: {under}"

    rel = [(cms.estimate(k) - v) / v for k, v in truth.items()]
    print(f"[CMS] 키 수={len(truth)}, 과소추정={under}, 평균 상대오차={sum(rel)/len(rel):.4f}")


if __name__ == "__main__":
    test_bloom()
    test_cms()
    print("ALL PASSED")