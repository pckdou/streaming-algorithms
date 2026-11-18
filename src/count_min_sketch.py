import math
import numpy as np
from src.utils import make_hash


class CountMinSketch:
    def __init__(self, width, depth):
        self.w = width
        self.d = depth
        self.table = np.zeros((depth, width), dtype=np.int64)

    def add(self, item, count=1):
        for i in range(self.d):
            self.table[i, make_hash(item, i, self.w)] += count

    def estimate(self, item):
        return int(min(self.table[i, make_hash(item, i, self.w)] for i in range(self.d)))

    def size_bytes(self):
        return self.table.nbytes

    @staticmethod
    def from_error(epsilon, delta):
        """오차 epsilon, 실패확률 delta에 맞는 (width, depth)"""
        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1 / delta))
        return width, depth