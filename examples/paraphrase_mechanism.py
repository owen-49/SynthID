#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from synthid import SynthIDTextWatermarker
from synthid.utils import set_all_seeds

# ---------- text metrics ----------

def ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else []

def distinct_n(token_lists, n):
    all_ngrams = []
    for toks in token_lists:
        all_ngrams.extend(ngrams(toks, n))
    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def unigram_entropy(token_lists):
    # entropy over token ids (unigram)
    counts = {}
    total = 0
    for toks in token_lists:
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(-(probs * np.log(probs + 1e-12)).sum())

@torch.no_grad()
def avg_nll_ppl(texts, tok, lm, device):
    """
    Returns:
      mean_nll, mean_ppl, std_nll, n_used, n_skipped
    We skip texts whose tokenized length < 2 (cannot compute causal LM loss meaningfully).
    """
    nlls = []
    n_used = 0
    n_skipped = 0

    for t in texts:
        if t is None or len(t.strip()) == 0:
            n_skipped += 1
            continue

        enc = tok(t, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"]
        if input_ids.numel() < 2 or input_ids.shape[-1] < 2:
            n_skipped += 1
            continue

        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", torch.ones_like(input_ids))

        outputs = lm(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = outputs.loss.detach().float().cpu().item()

        nlls.append(loss)
        n_used += 1

    if n_used == 0:
        # avoid crashing; return NaNs so you notice
        return float("nan"), float("nan"), float("nan"), 0, n_skipped

    mean_nll = float(np.mean(nlls))
    std_nll = float(np.std(nlls))
    mean_ppl = float(np.exp(mean_nll)) if mean_nll < 50 else float("inf")
    return mean_nll, mean_ppl, std_nll, n_used, n_skipped


# ---------- paraphrase ----------

def paraphrase_once(text, ptok, pmodel, device, max_new_tokens=128):
    prompt = (
        "Paraphrase the following text while preserving the meaning. "
        "Keep it fluent and natural.\n\n"
        f"{text}\n\nParaphrased:"
    )
    inputs = ptok(prompt, return_tensors="pt", truncation=True).to(device)
    out = pmodel.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
    )
    s = ptok.decode(out[0], skip_special_tokens=True).strip()
    if "Paraphrased:" in s:
        s = s.split("Paraphrased:")[-1].strip()
    return s

def paraphrase_k(texts, k, ptok, pmodel, device, max_new_tokens=128):
    out = texts
    for _ in range(k):
        out = [paraphrase_once(t, ptok, pmodel, device, max_new_tokens=max_new_tokens) for t in out]
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--para_model", type=str, default="google/flan-t5-small")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default="Explain the benefits of exercise.")
    parser.add_argument("--n_wm", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--K_list", type=str, default="1,2,3")
    args = parser.parse_args()

    set_all_seeds(2025)
    device = args.device
    K_list = [int(x.strip()) for x in args.K_list.split(",") if x.strip()]

    # base LM (for watermark generation + ppl scoring)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    lm.eval()

    # paraphraser
    ptok = AutoTokenizer.from_pretrained(args.para_model)
    pmodel = AutoModelForSeq2SeqLM.from_pretrained(args.para_model).to(device)
    pmodel.eval()

    # generate watermarked texts
    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)

    wm_texts = [
        wm.generate(
            model=lm,
            tokenizer=tok,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=1.0,
            top_k=50,
            device=device,
        )
        for _ in range(args.n_wm)
    ]

    groups = {"wm": wm_texts}
    for K in K_list:
        groups[f"para{K}"] = paraphrase_k(
            wm_texts, K, ptok, pmodel, device, max_new_tokens=args.max_new_tokens + 32
        )

    # tokenize groups for diversity/entropy metrics
    tok_groups = {}
    for name, texts in groups.items():
        tok_groups[name] = [tok.encode(t, add_special_tokens=False) for t in texts]

    # report
    print("\n=== Paraphrase Mechanism Report ===")
    header = "group | mean_len | std_len | distinct-1 | distinct-2 | unigram_entropy | mean_nll | ppl | std_nll"
    print(header)
    print("-" * len(header))

    for name in ["wm"] + [f"para{k}" for k in K_list]:
        token_lists = tok_groups[name]
        lens = np.array([len(x) for x in token_lists], dtype=np.float64)
        d1 = distinct_n(token_lists, 1)
        d2 = distinct_n(token_lists, 2)
        ent = unigram_entropy(token_lists)
        mean_nll, ppl, std_nll, n_used, n_skipped = avg_nll_ppl(groups[name], tok, lm, device)


        print(
        f"{name:5s} | {lens.mean():8.2f} | {lens.std():7.2f} | {d1:10.4f} | {d2:10.4f} | "
        f"{ent:14.4f} | {mean_nll:8.4f} | {ppl:6.2f} | {std_nll:7.4f} | "
        f"used={n_used:2d} skipped={n_skipped:2d}"
        )    


    print("\nInterpretation hints:")
    print("- K increases but distinct-n decreases / std_len decreases => paraphraser convergence (lower diversity).")
    print("- std_nll decreases => outputs become more uniform under the base LM (canonicalization).")

if __name__ == "__main__":
    main()
