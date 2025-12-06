

import argparse
import hashlib
import hmac
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


# =========================
#  Utils
# =========================

def pairwise(lst):
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


# =========================
#  PRF & Random Seed Generator
# =========================

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
    """
    key: bytes
    window_size: int = 4  # H in the paper

    def get_seed(self, context_ids: List[int]) -> bytes:
        # Take last H tokens (or all if shorter)
        window = context_ids[-self.window_size:]
        msg = ",".join(map(str, window)).encode("utf-8")
        return hmac_prf(self.key, msg, digest_size=16)  # 128-bit seed is enough


# =========================
#  g-values (Bernoulli)
# =========================

@dataclass
class GValueGenerator:
    """
    g_l(x, r_t) = Bernoulli(0.5) derived from PRF(key, token_id, layer, r_t)
    """
    key: bytes
    n_sec_bits: int = 128  # security parameter (not heavily used in demo)

    def g_value(self, token_id: int, layer: int, seed: bytes) -> int:
        """
        Generate g ∈ {0,1} from PRF(token_id, layer, seed, key).
        """
        msg = f"{token_id}|{layer}".encode("utf-8") + b"|" + seed
        prf_out = hmac_prf(self.key, msg, digest_size=16)
        val = prf_to_int(prf_out)
        # Use lowest bit as Bernoulli(0.5)
        return val & 1  # 0 or 1


# =========================
#  Tournament Sampling
# =========================

@dataclass
class TournamentSampler:
    """
    Tournament sampling watermarking.

    For demonstration we keep `m` small (e.g., 4 → 16 candidates).
    In the paper m can be ~30 with engineered optimizations.
    """
    g_gen: GValueGenerator
    num_layers: int = 4
    num_candidates: Optional[int] = None

    def __post_init__(self):
        if self.num_candidates is None:
            # Default: 2^m candidates
            self.num_candidates = 2 ** self.num_layers

    def sample_candidates(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """
        Sample `num_candidates` tokens from model distribution.
        logits: [vocab_size]
        """
        probs = softmax(logits, temperature=temperature)

        if top_k is not None and top_k < probs.size(-1):
            # top-k filtering
            topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
            topk_probs = topk_vals / topk_vals.sum()
            # sample in reduced vocab
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
        Main tournament process:
        - Sample candidates from LLM distribution
        - Run layered tournament using g-values
        - Return winning token_id
        """
        candidates = self.sample_candidates(logits, temperature=temperature, top_k=top_k)

        # Multi-layer tournament
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
                    winner = random.choice([a, b])
                new_candidates.append(winner)

            # 如果是奇数个，最后一个直接晋级（不太严谨，但够用）
            if len(candidates) % 2 == 1:
                new_candidates.append(candidates[-1])

            candidates = new_candidates
            if len(candidates) == 1:
                break

        return candidates[0]


# =========================
#  SynthID-Text Wrapper
# =========================

class SynthIDTextWatermarker:
    """
    High-level wrapper for watermarking & scoring.
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
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: str = "cpu",
    ) -> str:
        """
        Generate watermarked text using tournament sampling.
        """
        model.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            # model forward for last position
            with torch.no_grad():
                outputs = model(torch.tensor([generated], device=device))
                logits = outputs.logits[0, -1, :]  # [vocab_size]

            # random seed from recent context
            seed_bytes = self.seed_gen.get_seed(generated)

            # tournament sampling
            token_id = self.sampler.tournament_sample(
                logits,
                context_ids=generated,
                seed_bytes=seed_bytes,
                temperature=temperature,
                top_k=top_k,
            )

            generated.append(token_id)

            # stop at EOS if model has it
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

        return tokenizer.decode(generated, skip_special_tokens=True)

    # ---------- Detection / scoring ----------

    def score_tokens(
        self,
        token_ids: List[int],
    ) -> float:
        """
        Simple mean-g scoring over the whole sequence as in the paper.

        Score(x) = 1/(mT) * sum_{t,l} g_l(x_t, r_t)
        """
        if not token_ids:
            return 0.5

        m = self.sampler.num_layers
        T = len(token_ids)

        total = 0.0
        count = 0

        for t in range(T):
            context = token_ids[max(0, t - self.seed_gen.window_size):t]
            seed_bytes = self.seed_gen.get_seed(context)
            for layer in range(1, m + 1):
                g_val = self.g_gen.g_value(token_ids[t], layer, seed_bytes)
                total += g_val
                count += 1

        if count == 0:
            return 0.5
        return total / count

    def score_text(self, text: str, tokenizer) -> float:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return self.score_tokens(token_ids)


# =========================
#  Simple experiment
# =========================

def run_demo(
    model_name: str = "gpt2",
    prompt: str = "My favourite tropical fruit is",
    device: str = "cpu",
):
    print(f"Loading model `{model_name}` on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # fixed seed for reproducibility
    set_seed(42)
    random.seed(42)

    # create a random 256-bit key
    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)

    print("\n=== Generate plain (no watermark) ===")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        plain_out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=1.0,
            top_k=50,
        )
    plain_text = tokenizer.decode(plain_out[0], skip_special_tokens=True)
    print(plain_text)
    plain_score = wm.score_text(plain_text, tokenizer)
    print(f"[Plain score] {plain_score:.4f}")

    print("\n=== Generate watermarked ===")
    wm_text = wm.generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=64,
        temperature=1.0,
        top_k=50,
        device=device,
    )
    print(wm_text)
    wm_score = wm.score_text(wm_text, tokenizer)
    print(f"[Watermarked score] {wm_score:.4f}")

    print("\n=== Quick conclusion ===")
    print("Higher score ⇒ more likely watermarked.")
    print(f"Plain:       {plain_score:.4f}")
    print(f"Watermarked: {wm_score:.4f}")
    print("（真实论文里会跑很多样本画 ROC curve，这里是最小演示版。）")


# =========================
#  CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Minimal SynthID-Text style watermark demo.")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name (causal LM).")
    parser.add_argument("--prompt", type=str, default="My favourite tropical fruit is", help="Prompt text.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    run_demo(model_name=args.model, prompt=args.prompt, device=args.device)


if __name__ == "__main__":
    main()
