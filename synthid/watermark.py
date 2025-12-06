from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .seed import RandomSeedGenerator
from .gvalue import GValueGenerator
from .tournament import TournamentSampler


class SynthIDTextWatermarker:
    """
    高层封装：
    - 使用 Tournament Sampling 生成带水印文本
    - 提供简单的 mean-g score（更完整的在 scorer.py）
    """

    def __init__(
        self,
        key: bytes,
        window_size: int = 4,
        num_layers: int = 4,
    ):
        self.key = key
        self.seed_gen = RandomSeedGenerator(key=key, window_size=window_size)
        self.g_gen = GValueGenerator(key=key)
        self.sampler = TournamentSampler(g_gen=self.g_gen, num_layers=num_layers)

    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        device: str = "cpu",
    ) -> str:
        """
        使用带水印采样生成文本。
        """
        model.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(torch.tensor([generated], device=device))
                logits = outputs.logits[0, -1, :]

            seed_bytes = self.seed_gen.get_seed(generated)

            token_id = self.sampler.tournament_sample(
                logits,
                context_ids=generated,
                seed_bytes=seed_bytes,
                temperature=temperature,
                top_k=top_k,
            )
            generated.append(token_id)

            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

        return tokenizer.decode(generated, skip_special_tokens=True)

    # 下面提供一个简单打分（平均 g），更完整的 detection 在 scorer.py 里

    def _score_tokens_mean_g(self, token_ids: List[int]) -> float:
        m = self.sampler.num_layers
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

    def score_text_mean_g(self, text: str, tokenizer: PreTrainedTokenizerBase) -> float:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return self._score_tokens_mean_g(token_ids)
