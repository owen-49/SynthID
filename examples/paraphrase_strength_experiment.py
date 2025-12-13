#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 3.2: Paraphrase attack strength vs detectability.

Attack strength = number of paraphrase rounds K.
For each K:
  - paraphrase each watermarked sample K times (iteratively)
  - compute ROC/AUC for distinguishing wm (positive) vs paraphrased_K (negative)

Also prints baseline AUC for wm vs plain for reference.

Outputs:
  - table of AUC vs K
  - plot 'paraphrase_strength_auc.png'
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from synthid import SynthIDTextWatermarker, MeanGScorer, BayesianScorer
from synthid.seed import RandomSeedGenerator
from synthid.gvalue import GValueGenerator
from synthid.utils import set_all_seeds


def generate_plain_texts(model, tokenizer, prompt, n_samples, max_new_tokens, device):
    texts = []
    for _ in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=50,
            )
        texts.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return texts


def generate_watermarked_texts(wm, model, tokenizer, prompt, n_samples, max_new_tokens, device):
    return [
        wm.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            device=device,
        )
        for _ in range(n_samples)
    ]


def paraphrase_once(text, para_tokenizer, para_model, device, max_new_tokens=128):
    prompt = (
        "Paraphrase the following text while preserving the meaning. "
        "Keep it fluent and natural.\n\n"
        f"{text}\n\nParaphrased:"
    )
    inputs = para_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = para_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    out = para_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if "Paraphrased:" in out:
        out = out.split("Paraphrased:")[-1].strip()
    return out


def paraphrase_k_rounds(texts, k, para_tokenizer, para_model, device, max_new_tokens=128):
    out = texts
    for _ in range(k):
        out = [
            paraphrase_once(t, para_tokenizer, para_model, device, max_new_tokens=max_new_tokens)
            for t in out
        ]
    return out


def auc_for_pos_neg(scores_pos, scores_neg):
    y_true = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    y_score = np.array(scores_pos + scores_neg)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--para_model", type=str, default="google/flan-t5-small")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default="Explain the benefits of exercise.")
    parser.add_argument("--n_plain", type=int, default=50)
    parser.add_argument("--n_wm", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--K_list", type=str, default="1,2,3")
    parser.add_argument("--out", type=str, default="paraphrase_strength_auc.png")
    args = parser.parse_args()

    set_all_seeds(2025)
    device = args.device
    K_list = [int(x.strip()) for x in args.K_list.split(",") if x.strip()]

    print(f"Loading base model {args.model} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    print(f"Loading paraphraser {args.para_model} on {device}...")
    ptok = AutoTokenizer.from_pretrained(args.para_model)
    pmodel = AutoModelForSeq2SeqLM.from_pretrained(args.para_model).to(device)
    pmodel.eval()

    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)
    seed_gen = RandomSeedGenerator(key=key, window_size=4)
    g_gen = GValueGenerator(key=key)
    mean_scorer = MeanGScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)
    bayes_scorer = BayesianScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)

    print(f"\nGenerating {args.n_plain} plain samples...")
    plain_texts = generate_plain_texts(base_model, tok, args.prompt, args.n_plain, args.max_new_tokens, device)

    print(f"\nGenerating {args.n_wm} watermarked samples...")
    wm_texts = generate_watermarked_texts(wm, base_model, tok, args.prompt, args.n_wm, args.max_new_tokens, device)

    print("\nCalibrating Bayesian scorer on watermarked samples...")
    bayes_scorer.fit_from_texts(wm_texts, tok)
    print(f"Estimated p1 = {bayes_scorer.p1:.4f}")

    # baseline AUC (wm vs plain) for reference
    mean_plain = [mean_scorer.score_text(t, tok) for t in plain_texts]
    mean_wm = [mean_scorer.score_text(t, tok) for t in wm_texts]
    bayes_plain = [bayes_scorer.score_text(t, tok) for t in plain_texts]
    bayes_wm = [bayes_scorer.score_text(t, tok) for t in wm_texts]

    auc_mean_baseline = auc_for_pos_neg(scores_pos=mean_wm, scores_neg=mean_plain)
    auc_bayes_baseline = auc_for_pos_neg(scores_pos=bayes_wm, scores_neg=bayes_plain)

    print("\n=== Baseline AUC (wm vs plain) ===")
    print(f"Mean-g:   {auc_mean_baseline:.4f}")
    print(f"Bayesian: {auc_bayes_baseline:.4f}")

    # paraphrase strength experiments
    results = []  # (K, auc_mean, auc_bayes)
    for K in K_list:
        print(f"\n=== Paraphrase rounds K = {K} ===")
        para_texts = paraphrase_k_rounds(
            wm_texts, K, ptok, pmodel, device, max_new_tokens=args.max_new_tokens + 32
        )

        mean_para = [mean_scorer.score_text(t, tok) for t in para_texts]
        bayes_para = [bayes_scorer.score_text(t, tok) for t in para_texts]

        auc_mean = auc_for_pos_neg(scores_pos=mean_wm, scores_neg=mean_para)
        auc_bayes = auc_for_pos_neg(scores_pos=bayes_wm, scores_neg=bayes_para)

        print(f"AUC Mean-g (wm vs para{K}):   {auc_mean:.4f}")
        print(f"AUC Bayesian (wm vs para{K}): {auc_bayes:.4f}")

        results.append((K, auc_mean, auc_bayes))

    # summary table
    print("\n=== Summary: AUC vs paraphrase rounds K ===")
    print("K | Mean-g AUC | Bayesian AUC")
    print("--+-----------+------------")
    for K, am, ab in results:
        print(f"{K:1d} | {am:9.4f} | {ab:10.4f}")

    # plot
    Ks = [r[0] for r in results]
    auc_means = [r[1] for r in results]
    auc_bayes = [r[2] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(Ks, auc_means, marker="o", label="Mean-g AUC (wm vs paraK)")
    plt.plot(Ks, auc_bayes, marker="o", label="Bayesian AUC (wm vs paraK)")
    plt.axhline(auc_mean_baseline, linestyle="--", label="Mean-g baseline (wm vs plain)")
    plt.axhline(auc_bayes_baseline, linestyle="--", label="Bayesian baseline (wm vs plain)")
    plt.xlabel("Paraphrase rounds K")
    plt.ylabel("AUC")
    plt.ylim(0.5, 1.01)
    plt.title("Detectability vs Paraphrase Strength")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"\nPlot saved to {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
