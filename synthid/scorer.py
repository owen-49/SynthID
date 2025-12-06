from dataclasses import dataclass
from typing import List, Sequence

import math
from transformers import PreTrainedTokenizerBase

from .seed import RandomSeedGenerator
from .gvalue import GValueGenerator


@dataclass
class MeanGScorer:
    """
    论文里的基础 scoring：
        Score = (1 / (mT)) * sum_{t,l} g_l(x_t, r_t)
    """
    seed_gen: RandomSeedGenerator
    g_gen: GValueGenerator
    num_layers: int

    def score_tokens(self, token_ids: List[int]) -> float:
        m = self.num_layers
        T = len(token_ids)
        if T == 0:
            return 0.5

        total = 0.0
        count = 0
        for t in range(T):
            context = token_ids[max(0, t - self.seed_gen.window_size):t]
            seed_bytes = self.seed_gen.get_seed(context)
            for layer in range(1, m + 1):
                g_val = self.g_gen.g_value(token_ids[t], layer, seed_bytes)
                total += g_val
                count += 1

        return total / count

    def score_text(self, text: str, tokenizer: PreTrainedTokenizerBase) -> float:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return self.score_tokens(token_ids)


@dataclass
class BayesianScorer:
    """
    非严格版 Bayesian scoring：

    假设：
      H0: g ~ Bernoulli(0.5)
      H1: g ~ Bernoulli(p1), p1 > 0.5

    先用一批“水印文本”估计 p1，然后对新文本计算 log-likelihood ratio，
    最后给出 posterior P(H1 | g_seq)。
    """
    seed_gen: RandomSeedGenerator
    g_gen: GValueGenerator
    num_layers: int
    prior_watermarked: float = 0.5

    # 通过 calibration 学到：
    p1: float = 0.6  # default, 会被 fit() 覆盖

    # -------- Calibration --------

    def fit_from_texts(
        self,
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
    ):
        """
        用多条“确定是水印文本”的数据估计 p1。
        """
        g_values = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            g_values.extend(self._collect_g_bits(token_ids))

        if not g_values:
            self.p1 = 0.6
        else:
            self.p1 = sum(g_values) / len(g_values)
            # 稍微 regularize 避免完全 0/1
            eps = 1e-3
            self.p1 = min(max(self.p1, 0.5 + eps), 1.0 - eps)

    # -------- Scoring --------

    def _collect_g_bits(self, token_ids: List[int]) -> List[int]:
        m = self.num_layers
        bits = []
        for t in range(len(token_ids)):
            context = token_ids[max(0, t - self.seed_gen.window_size):t]
            seed_bytes = self.seed_gen.get_seed(context)
            for layer in range(1, m + 1):
                bits.append(self.g_gen.g_value(token_ids[t], layer, seed_bytes))
        return bits

    def score_tokens(self, token_ids: List[int]) -> float:
        """
        返回 posterior P(H1 | g_seq)，数值越大越像水印文本。
        """
        bits = self._collect_g_bits(token_ids)
        if not bits:
            return 0.5

        p0 = 0.5
        p1 = self.p1

        log_likelihood_ratio = 0.0
        for g in bits:
            if g == 1:
                log_likelihood_ratio += math.log(p1 / p0)
            else:
                log_likelihood_ratio += math.log((1 - p1) / (1 - p0))

        # Bayes rule: posterior = sigmoid(logit(prior) + LLR)
        prior = self.prior_watermarked
        logit_prior = math.log(prior / (1 - prior))
        logit_posterior = logit_prior + log_likelihood_ratio
        posterior = 1 / (1 + math.exp(-logit_posterior))
        return posterior

    def score_text(self, text: str, tokenizer: PreTrainedTokenizerBase) -> float:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return self.score_tokens(token_ids)
