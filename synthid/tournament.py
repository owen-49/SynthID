from dataclasses import dataclass
from typing import List, Optional

import torch

from .gvalue import GValueGenerator
from .utils import pairwise, softmax


@dataclass
class TournamentSampler:
    """
    Tournament sampling watermarking.

    为了 demo，默认 num_layers=4 → 16 个候选。
    论文中可以用更大的 m（如 30）配合工程优化。
    """
    g_gen: GValueGenerator
    num_layers: int = 4
    num_candidates: Optional[int] = None

    def __post_init__(self):
        if self.num_candidates is None:
            self.num_candidates = 2 ** self.num_layers

    def sample_candidates(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """
        从 LLM 分布中采样 num_candidates 个候选 token。
        """
        probs = softmax(logits, temperature=temperature)

        if top_k is not None and top_k < probs.size(-1):
            topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
            topk_probs = topk_vals / topk_vals.sum()
            local_ids = torch.multinomial(topk_probs, num_samples=self.num_candidates, replacement=True)
            token_ids = topk_idx[local_ids]
        else:
            token_ids = torch.multinomial(probs, num_samples=self.num_candidates, replacement=True)

        return token_ids.tolist()

    def tournament_sample(
        self,
        logits: torch.Tensor,
        context_ids: List[int],
        seed_bytes: bytes,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> int:
        """
        主流程：
        - 从 LLM 分布采样多个候选
        - 按层使用 g-values 打“锦标赛”
        - 返回 winner token id
        """
        candidates = self.sample_candidates(logits, temperature=temperature, top_k=top_k)

        for layer in range(1, self.num_layers + 1):
            new_candidates = []
            for a, b in pairwise(candidates):
                ga = self.g_gen.g_value(a, layer, seed_bytes)
                gb = self.g_gen.g_value(b, layer, seed_bytes)
                if ga > gb:
                    winner = a
                elif gb > ga:
                    winner = b
                else:
                    winner = torch.randint(0, 2, (1,)).item() and a or b
                new_candidates.append(winner)

            # 奇数个：最后一个直接晋级（不严格但够 demo 用）
            if len(candidates) % 2 == 1:
                new_candidates.append(candidates[-1])

            candidates = new_candidates
            if len(candidates) == 1:
                break

        return candidates[0]
