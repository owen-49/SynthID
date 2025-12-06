import random
from typing import Iterable, List, Tuple

import torch


def pairwise(lst: List[int]) -> Iterable[Tuple[int, int]]:
    """Yield (a, b) pairs from list, ignore last if odd length."""
    it = iter(lst)
    for a in it:
        try:
            b = next(it)
        except StopIteration:
            return
        yield a, b


def softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature
    return torch.softmax(logits, dim=-1)


def set_all_seeds(seed: int = 42):
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
