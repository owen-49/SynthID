#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROC experiment for SynthID-Text-style watermarking.

- 生成一批无水印文本（plain）
- 生成一批带水印文本（watermarked）
- 对每条文本计算 Mean-g 和 Bayesian score
- 使用 sklearn 画 ROC 曲线，并保存为 PNG 图片
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from synthid import SynthIDTextWatermarker, MeanGScorer, BayesianScorer
from synthid.seed import RandomSeedGenerator
from synthid.gvalue import GValueGenerator
from synthid.utils import set_all_seeds


def generate_plain_texts(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    texts = []
    for i in range(n_samples):
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


def generate_watermarked_texts(
    wm,
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    texts = []
    for i in range(n_samples):
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
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name (causal LM).")
    parser.add_argument("--prompt", type=str, default="Explain the benefits of exercise.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_plain", type=int, default=50, help="Number of non-watermarked samples.")
    parser.add_argument("--n_wm", type=int, default=50, help="Number of watermarked samples.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out", type=str, default="roc_synthid.png", help="Output PNG file.")
    args = parser.parse_args()

    device = args.device
    set_all_seeds(123)

    print(f"Loading model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    # 生成随机密钥 & 初始化水印组件
    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)
    seed_gen = RandomSeedGenerator(key=key, window_size=4)
    g_gen = GValueGenerator(key=key)
    mean_scorer = MeanGScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)
    bayes_scorer = BayesianScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)

    # ============== 生成数据集 ==============
    print(f"\nGenerating {args.n_plain} plain (non-watermarked) samples...")
    plain_texts = generate_plain_texts(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        n_samples=args.n_plain,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    print(f"\nGenerating {args.n_wm} watermarked samples...")
    wm_texts = generate_watermarked_texts(
        wm=wm,
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        n_samples=args.n_wm,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    # ============== 用水印文本校准 BayesianScorer ==============
    print("\nCalibrating Bayesian scorer on watermarked samples...")
    # 这里直接用全部 wm_texts 做 calibration（也可只用一部分）
    bayes_scorer.fit_from_texts(wm_texts, tokenizer)
    print(f"Estimated p1 (P(g=1|H1)) = {bayes_scorer.p1:.4f}")

    # ============== 对所有文本打分 ==============
    print("\nScoring all samples...")

    y_true = np.array([0] * len(plain_texts) + [1] * len(wm_texts))

    mean_scores_plain = [mean_scorer.score_text(t, tokenizer) for t in plain_texts]
    mean_scores_wm = [mean_scorer.score_text(t, tokenizer) for t in wm_texts]
    mean_scores = np.array(mean_scores_plain + mean_scores_wm)

    bayes_scores_plain = [bayes_scorer.score_text(t, tokenizer) for t in plain_texts]
    bayes_scores_wm = [bayes_scorer.score_text(t, tokenizer) for t in wm_texts]
    bayes_scores = np.array(bayes_scores_plain + bayes_scores_wm)

    # ============== 计算 ROC & AUC ==============
    fpr_mean, tpr_mean, _ = roc_curve(y_true, mean_scores)
    auc_mean = auc(fpr_mean, tpr_mean)

    fpr_bayes, tpr_bayes, _ = roc_curve(y_true, bayes_scores)
    auc_bayes = auc(fpr_bayes, tpr_bayes)

    print("\n=== AUC scores ===")
    print(f"Mean-g AUC:   {auc_mean:.4f}")
    print(f"Bayesian AUC: {auc_bayes:.4f}")

    # ============== 画 ROC 图 ==============
    print(f"\nPlotting ROC curves to {args.out} ...")

    plt.figure()
    plt.plot(fpr_mean, tpr_mean, label=f"Mean-g (AUC = {auc_mean:.3f})")
    plt.plot(fpr_bayes, tpr_bayes, label=f"Bayesian (AUC = {auc_bayes:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for SynthID-Style Watermark Detection")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
