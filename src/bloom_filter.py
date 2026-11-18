import math
from src.utils import make_hash


class BloomFilter:
    def __init__(self, m, k):
        self.m = m                              # 비트 배열 크기(bit)
        self.k = k                              # 해시 함수 개수
        self.bits = bytearray((m + 7) // 8)     # 비트를 바이트 배열에 압축 저장

    def _positions(self, item):
        return [make_hash(item, i, self.m) for i in range(self.k)]

    def add(self, item):
        for p in self._positions(item):
            self.bits[p >> 3] |= (1 << (p & 7))

    def __contains__(self, item):
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._positions(item))

    def size_bytes(self):
        return len(self.bits)

    def fill_ratio(self):
        """1로 켜진 비트 비율 (포화도 분석용)"""
        ones = sum(bin(b).count("1") for b in self.bits)
        return ones / self.m

    @staticmethod
    def optimal(n, fpr):
        """원소 n개, 목표 FPR일 때 최적 (m, k)"""
        m = int(-n * math.log(fpr) / (math.log(2) ** 2))
        k = max(1, round(m / n * math.log(2)))
        return m, k

    @staticmethod
    def theoretical_fpr(n, m, k):
        return (1 - math.exp(-k * n / m)) ** k