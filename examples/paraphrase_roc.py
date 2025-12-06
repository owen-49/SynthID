#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paraphrase attack ROC experiment for SynthID-style watermarking.

- 生成 plain & watermarked 文本
- 用一个 paraphraser 模型重写 watermarked 文本
- 比较：
    1) wm vs plain 的 ROC
    2) wm vs paraphrased 的 ROC
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from synthid import SynthIDTextWatermarker, MeanGScorer, BayesianScorer
from synthid.seed import RandomSeedGenerator
from synthid.gvalue import GValueGenerator
from synthid.utils import set_all_seeds


# -------- 基本生成 --------

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


# -------- paraphrase 工具 --------

def paraphrase_texts(
    texts,
    para_tokenizer,
    para_model,
    device,
    max_new_tokens=128,
):
    """
    使用 seq2seq 模型批量 paraphrase 文本。
    推荐 paraphraser: google/flan-t5-small 或 t5-small。
    """
    out_texts = []
    para_model.eval()

    for text in texts:
        prompt = f"Paraphrase the following text while preserving the meaning:\n\n{text}\n\nParaphrased:"
        inputs = para_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = para_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        full = para_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 简单 heuristic：取 "Paraphrased:" 之后的内容
        if "Paraphrased:" in full:
            para = full.split("Paraphrased:")[-1].strip()
        else:
            para = full.strip()
        out_texts.append(para)

    return out_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="base causal LM for watermarking")
    parser.add_argument("--prompt", type=str, default="Explain the benefits of exercise.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_plain", type=int, default=50)
    parser.add_argument("--n_wm", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--para_model", type=str, default="google/flan-t5-small",
                        help="seq2seq paraphraser model")
    parser.add_argument("--out", type=str, default="paraphrase_roc.png")
    args = parser.parse_args()

    set_all_seeds(2025)
    device = args.device

    # ---------- 加载 base LM ----------
    print(f"Loading base model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # ---------- 加载 paraphraser ----------
    print(f"Loading paraphraser {args.para_model} on {device}...")
    para_tokenizer = AutoTokenizer.from_pretrained(args.para_model)
    para_model = AutoModelForSeq2SeqLM.from_pretrained(args.para_model).to(device)

    # ---------- 初始化水印组件 ----------
    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)
    seed_gen = RandomSeedGenerator(key=key, window_size=4)
    g_gen = GValueGenerator(key=key)
    mean_scorer = MeanGScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)
    bayes_scorer = BayesianScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)

    # ---------- 生成 plain & wm ----------
    print(f"\nGenerating {args.n_plain} plain samples...")
    plain_texts = generate_plain_texts(
        model, tokenizer, args.prompt, args.n_plain, args.max_new_tokens, device
    )

    print(f"\nGenerating {args.n_wm} watermarked samples...")
    wm_texts = generate_watermarked_texts(
        wm, model, tokenizer, args.prompt, args.n_wm, args.max_new_tokens, device
    )

    # ---------- 校准 Bayesian scorer ----------
    print("\nCalibrating Bayesian scorer on watermarked samples...")
    bayes_scorer.fit_from_texts(wm_texts, tokenizer)
    print(f"Estimated p1 = {bayes_scorer.p1:.4f}")

    # ---------- paraphrase 攻击 ----------
    print("\nParaphrasing watermarked samples...")
    para_texts = paraphrase_texts(
        wm_texts,
        para_tokenizer=para_tokenizer,
        para_model=para_model,
        device=device,
        max_new_tokens=args.max_new_tokens + 32,
    )

    # ---------- 打分 ----------
    print("\nScoring samples...")

    # baseline: wm vs plain
    y_true_base = np.array([0] * len(plain_texts) + [1] * len(wm_texts))
    mean_base = np.array(
        [mean_scorer.score_text(t, tokenizer) for t in plain_texts]
        + [mean_scorer.score_text(t, tokenizer) for t in wm_texts]
    )
    bayes_base = np.array(
        [bayes_scorer.score_text(t, tokenizer) for t in plain_texts]
        + [bayes_scorer.score_text(t, tokenizer) for t in wm_texts]
    )

    # attack: wm vs paraphrased
    y_true_attack = np.array([0] * len(para_texts) + [1] * len(wm_texts))
    mean_attack = np.array(
        [mean_scorer.score_text(t, tokenizer) for t in para_texts]
        + [mean_scorer.score_text(t, tokenizer) for t in wm_texts]
    )
    bayes_attack = np.array(
        [bayes_scorer.score_text(t, tokenizer) for t in para_texts]
        + [bayes_scorer.score_text(t, tokenizer) for t in wm_texts]
    )

    # ---------- 计算 ROC ----------
    from sklearn.metrics import roc_curve, auc

    fpr_m_base, tpr_m_base, _ = roc_curve(y_true_base, mean_base)
    auc_m_base = auc(fpr_m_base, tpr_m_base)
    fpr_b_base, tpr_b_base, _ = roc_curve(y_true_base, bayes_base)
    auc_b_base = auc(fpr_b_base, tpr_b_base)

    fpr_m_att, tpr_m_att, _ = roc_curve(y_true_attack, mean_attack)
    auc_m_att = auc(fpr_m_att, tpr_m_att)
    fpr_b_att, tpr_b_att, _ = roc_curve(y_true_attack, bayes_attack)
    auc_b_att = auc(fpr_b_att, tpr_b_att)

    print("\n=== AUC (baseline: wm vs plain) ===")
    print(f"Mean-g:   {auc_m_base:.4f}")
    print(f"Bayesian: {auc_b_base:.4f}")

    print("\n=== AUC (attack: wm vs paraphrased) ===")
    print(f"Mean-g:   {auc_m_att:.4f}")
    print(f"Bayesian: {auc_b_att:.4f}")

    # ---------- 画图 ----------
    print(f"\nPlotting ROC curves to {args.out} ...")

    plt.figure(figsize=(8, 6))
    # baseline
    plt.plot(fpr_m_base, tpr_m_base, label=f"Mean-g (baseline, AUC={auc_m_base:.3f})", linestyle="-")
    plt.plot(fpr_b_base, tpr_b_base, label=f"Bayesian (baseline, AUC={auc_b_base:.3f})", linestyle="-")

    # paraphrase attack
    plt.plot(fpr_m_att, tpr_m_att, label=f"Mean-g (para, AUC={auc_m_att:.3f})", linestyle="--")
    plt.plot(fpr_b_att, tpr_b_att, label=f"Bayesian (para, AUC={auc_b_att:.3f})", linestyle="--")

    plt.plot([0, 1], [0, 1], "g--", label="Chance")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC under Paraphrase Attack")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
