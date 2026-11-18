import hashlib


def make_hash(item, seed, mod):
    """seed별로 서로 다른 해시 값을 만든다. (MD5 기반, 직접 구현)"""
    data = f"{seed}:{item}".encode("utf-8")
    h = hashlib.md5(data).digest()
    return int.from_bytes(h[:8], "little") % mod