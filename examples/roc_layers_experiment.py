#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 3.1: Effect of tournament depth (num_layers) on detection performance.

For each num_layers in LAYER_LIST:
    - build a SynthIDTextWatermarker with that num_layers
    - generate N_plain plain samples and N_wm watermarked samples
    - calibrate BayesianScorer on watermarked samples
    - compute ROC + AUC for Mean-g and Bayesian
    - record AUC and finally plot AUC vs num_layers

Outputs:
    - Prints a result table
    - Saves a plot 'roc_layers_auc.png' in the current directory
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        txt = tokenizer.decode(out[0], skip_special_tokens=True)
        texts.append(txt)
    return texts


def generate_watermarked_texts(wm, model, tokenizer, prompt, n_samples, max_new_tokens, device):
    texts = []
    for _ in range(n_samples):
        txt = wm.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            device=device,
        )
        texts.append(txt)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="base causal LM")
    parser.add_argument("--prompt", type=str, default="Explain the benefits of exercise.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_plain", type=int, default=50)
    parser.add_argument("--n_wm", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out", type=str, default="roc_layers_auc.png")
    args = parser.parse_args()

    set_all_seeds(1234)
    device = args.device

    # 可以根据需要增加/减少深度
    LAYER_LIST = [1, 2, 4, 8]

    print(f"Loading model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # 为了方便比较，对所有 num_layers 使用同一个 key
    key = os.urandom(32)

    results = []  # 每项: (num_layers, mean_auc, bayes_auc)

    for m in LAYER_LIST:
        print(f"\n=== Running experiment for num_layers = {m} ===")

        # 初始化水印组件和检测器
        wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=m)
        seed_gen = RandomSeedGenerator(key=key, window_size=4)
        g_gen = GValueGenerator(key=key)
        mean_scorer = MeanGScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=m)
        bayes_scorer = BayesianScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=m)

        # 生成数据
        print(f"Generating {args.n_plain} plain samples...")
        plain_texts = generate_plain_texts(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            n_samples=args.n_plain,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        print(f"Generating {args.n_wm} watermarked samples...")
        wm_texts = generate_watermarked_texts(
            wm=wm,
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            n_samples=args.n_wm,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        # 用水印样本校准 BayesianScorer
        print("Calibrating Bayesian scorer...")
        bayes_scorer.fit_from_texts(wm_texts, tokenizer)
        print(f"Estimated p1 = {bayes_scorer.p1:.4f}")

        # 打分
        print("Scoring samples...")
        y_true = np.array([0] * len(plain_texts) + [1] * len(wm_texts))

        mean_scores_plain = [mean_scorer.score_text(t, tokenizer) for t in plain_texts]
        mean_scores_wm = [mean_scorer.score_text(t, tokenizer) for t in wm_texts]
        mean_scores = np.array(mean_scores_plain + mean_scores_wm)

        bayes_scores_plain = [bayes_scorer.score_text(t, tokenizer) for t in plain_texts]
        bayes_scores_wm = [bayes_scorer.score_text(t, tokenizer) for t in wm_texts]
        bayes_scores = np.array(bayes_scores_plain + bayes_scores_wm)

        # ROC & AUC
        fpr_m, tpr_m, _ = roc_curve(y_true, mean_scores)
        auc_m = auc(fpr_m, tpr_m)

        fpr_b, tpr_b, _ = roc_curve(y_true, bayes_scores)
        auc_b = auc(fpr_b, tpr_b)

        print(f"Mean-g AUC (m={m}):   {auc_m:.4f}")
        print(f"Bayesian AUC (m={m}): {auc_b:.4f}")

        results.append((m, auc_m, auc_b))

    # 打印总表
    print("\n=== Summary: AUC vs num_layers ===")
    print("num_layers | Mean-g AUC | Bayesian AUC")
    print("-----------+-----------+-------------")
    for m, auc_m, auc_b in results:
        print(f"{m:10d} | {auc_m:9.4f} | {auc_b:11.4f}")

    # 绘制 AUC vs num_layers 曲线
    ms = [r[0] for r in results]
    mean_aucs = [r[1] for r in results]
    bayes_aucs = [r[2] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(ms, mean_aucs, marker="o", label="Mean-g AUC")
    plt.plot(ms, bayes_aucs, marker="o", label="Bayesian AUC")
    plt.xlabel("num_layers (tournament depth)")
    plt.ylabel("AUC")
    plt.ylim(0.5, 1.01)
    plt.title("Detection AUC vs Tournament Depth")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"\nAUC plot saved to {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
