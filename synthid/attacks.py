import random
from typing import List

from transformers import PreTrainedTokenizerBase


def random_delete_tokens(token_ids: List[int], rate: float = 0.1) -> List[int]:
    """
    随机删掉若干 token。
    """
    keep = []
    for tid in token_ids:
        if random.random() >= rate:
            keep.append(tid)
    return keep or token_ids  # avoid empty


def random_replace_tokens(token_ids: List[int], vocab_size: int, rate: float = 0.1) -> List[int]:
    """
    随机把部分 token 替换成随机 token。
    """
    out = []
    for tid in token_ids:
        if random.random() < rate:
            out.append(random.randint(0, vocab_size - 1))
        else:
            out.append(tid)
    return out


def truncate_tokens(token_ids: List[int], keep_ratio: float = 0.5) -> List[int]:
    """
    前半截或前 keep_ratio 部分。
    """
    k = max(1, int(len(token_ids) * keep_ratio))
    return token_ids[:k]


def shuffle_sentences(text: str) -> str:
    """
    简单按句子打乱顺序（极粗糙版）。
    """
    import re
    sents = re.split(r'([.!?])', text)
    # 把句子和符号合并回片段
    chunks = []
    for i in range(0, len(sents) - 1, 2):
        chunks.append(sents[i] + sents[i + 1])
    if len(sents) % 2 == 1:
        chunks.append(sents[-1])
    random.shuffle(chunks)
    return " ".join(chunks)


def paraphrase_with_model(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model,
    device: str = "cpu",
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> str:
    """
    简单 paraphrase：提示另一个模型“用不同的表述重写这段话”。

    建议用一个通用指令模型当 paraphraser。
    """
    prompt = f"Paraphrase the following text while preserving the meaning:\n\n{text}\n\nParaphrased:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 粗暴获取 "Paraphrased:" 后面的部分
    return full.split("Paraphrased:")[-1].strip()
