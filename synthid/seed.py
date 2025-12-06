from dataclasses import dataclass
from typing import List
import hashlib
import hmac


def hmac_prf(key: bytes, msg: bytes, digest_size: int = 32) -> bytes:
    """
    Simple PRF using HMAC-SHA256.
    Returns `digest_size` bytes (<= 32).
    """
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    if digest_size < len(digest):
        return digest[:digest_size]
    return digest


def prf_to_int(prf_out: bytes) -> int:
    """Convert PRF output bytes to non-negative integer."""
    return int.from_bytes(prf_out, byteorder="big", signed=False)


@dataclass
class RandomSeedGenerator:
    """
    Sliding-window random seed generator:

        r_t = PRF(key, context[-H:])

    context 用 token id实现和论文里的 sliding-window hash 类似。
    """
    key: bytes
    window_size: int = 4  # H

    def get_seed(self, context_ids: List[int]) -> bytes:
        # last H tokens (or all if shorter)
        window = context_ids[-self.window_size:]
        msg = ",".join(map(str, window)).encode("utf-8")
        return hmac_prf(self.key, msg, digest_size=16)
