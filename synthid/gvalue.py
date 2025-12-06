from dataclasses import dataclass
from .seed import hmac_prf, prf_to_int


@dataclass
class GValueGenerator:
    """
    g_l(x, r_t) = Bernoulli(0.5) derived from PRF(key, token_id, layer, r_t)

    这里实现的就是论文里的随机 watermarking functions g_ℓ(x, r_t)。
    """
    key: bytes
    n_sec_bits: int = 128 # 安全参数，PRF 输出的位数

    def g_value(self, token_id: int, layer: int, seed: bytes) -> int:
        """
        Generate g ∈ {0,1} from PRF(token_id, layer, seed, key).
        """
        msg = f"{token_id}|{layer}".encode("utf-8") + b"|" + seed
        prf_out = hmac_prf(self.key, msg, digest_size=16)
        val = prf_to_int(prf_out)
        # 取最低 bit 当作 Bernoulli(0.5)
        return val & 1  # 0 or 1
