from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .seed import RandomSeedGenerator
from .gvalue import GValueGenerator
from .tournament import TournamentSampler


@dataclass
class WatermarkedSpeculativeGenerator:
    """
    教学版 High-detectability speculative sampling + watermarking.

    假设：
    - draft_model 和 target_model 使用同一个 tokenizer
    - 每个最终提交的 token 都来自 target_model + TournamentSampler
    - draft_model 只提供“候选前缀”，在 candidate token 恰好等于
      watermarked token 时，我们算它“被接受”。

    注意：这个版本主要用于理解逻辑，速度上不会比普通 sampling 快太多。
    """

    key: bytes
    window_size: int = 4
    num_layers: int = 4

    def __post_init__(self):
        self.seed_gen = RandomSeedGenerator(key=self.key, window_size=self.window_size)
        self.g_gen = GValueGenerator(key=self.key)
        self.sampler = TournamentSampler(g_gen=self.g_gen, num_layers=self.num_layers)

    def _watermarked_next_token(
        self,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        committed_ids: List[int],
        device: str,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> int:
        """
        用 target_model + TournamentSampling 生成下一个带水印 token。
        """
        with torch.no_grad():
            inputs = torch.tensor([committed_ids], device=device)
            outputs = target_model(inputs)
            logits = outputs.logits[0, -1, :]

        seed_bytes = self.seed_gen.get_seed(committed_ids)
        token_id = self.sampler.tournament_sample(
            logits=logits,
            context_ids=committed_ids,
            seed_bytes=seed_bytes,
            temperature=temperature,
            top_k=top_k,
        )
        return token_id

    def generate(
        self,
        target_model: PreTrainedModel,
        draft_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_new_tokens: int = 64,
        draft_steps: int = 4,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        device: str = "cpu",
    ) -> str:
        """
        Speculative 生成过程：

        while 总 token 数 < max_new_tokens:
            1) 用 draft_model 从当前 committed_ids 生成 draft_steps 个 token（普通采样）
            2) 逐个 token 遍历：
                - 用 target_model + TournamentSampling 生成真正的水印 token wm_tok
                - 如果 wm_tok == draft_tok，则“接受” draft token
                - 否则，用 wm_tok 覆盖，并丢弃 draft 之后所有 token（回滚）

        最终：每个 committed_ids 中的新增 token 都是由 target+watermark 产生的，
        但在“好运气”时可以直接沿用 draft 提出的 token。
        """
        target_model.eval()
        draft_model.eval()

        # 初始 committed 序列
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        committed = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            # 1) draft 从当前 committed 生成一个小段前缀
            with torch.no_grad():
                draft_inputs = torch.tensor([committed], device=device)
                draft_out = draft_model.generate(
                    draft_inputs,
                    max_new_tokens=draft_steps,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                )
            # 新生成的部分
            draft_suffix = draft_out[0].tolist()[len(committed):]

            # 2) 逐个消费 draft token
            for draft_tok in draft_suffix:
                wm_tok = self._watermarked_next_token(
                    target_model=target_model,
                    tokenizer=tokenizer,
                    committed_ids=committed,
                    device=device,
                    temperature=temperature,
                    top_k=top_k,
                )

                committed.append(wm_tok)

                # EOS 检测
                if tokenizer.eos_token_id is not None and wm_tok == tokenizer.eos_token_id:
                    return tokenizer.decode(committed, skip_special_tokens=True)

                # mismatch：draft 提案和水印 token 不一致，丢弃剩余 draft
                if wm_tok != draft_tok:
                    break

            # 循环下一轮（从新的 committed 继续）

        return tokenizer.decode(committed, skip_special_tokens=True)
