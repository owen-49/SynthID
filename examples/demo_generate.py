#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from synthid import SynthIDTextWatermarker, MeanGScorer, BayesianScorer
from synthid.seed import RandomSeedGenerator
from synthid.gvalue import GValueGenerator
from synthid.utils import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="My favourite tropical fruit is")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    set_all_seeds(42)

    print(f"Loading model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    # 随机生成一个 256-bit key
    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)

    # ========== Plain generation ==========
    print("\n=== Plain generation (no watermark) ===")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
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

    # ========== Watermarked generation ==========
    print("\n=== Watermarked generation ===")
    wm_text = wm.generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=64,
        temperature=1.0,
        top_k=50,
        device=device,
    )
    print(wm_text)

    # ========== Scoring (MeanG + Bayesian) ==========
    seed_gen = RandomSeedGenerator(key=key, window_size=4)
    g_gen = GValueGenerator(key=key)

    mean_scorer = MeanGScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)
    bayes_scorer = BayesianScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)

    # 用几条水印文本做 calibration（这里就用重复生成几条）
    calib_texts = []
    for _ in range(5):
        t = wm.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=64,
            temperature=1.0,
            top_k=50,
            device=device,
        )
        calib_texts.append(t)

    bayes_scorer.fit_from_texts(calib_texts, tokenizer)

    plain_mean = mean_scorer.score_text(plain_text, tokenizer)
    wm_mean = mean_scorer.score_text(wm_text, tokenizer)

    plain_bayes = bayes_scorer.score_text(plain_text, tokenizer)
    wm_bayes = bayes_scorer.score_text(wm_text, tokenizer)

    print("\n=== Scores ===")
    print(f"Plain mean-g:   {plain_mean:.4f}")
    print(f"WM mean-g:      {wm_mean:.4f}")
    print(f"Plain Bayesian: {plain_bayes:.4f}")
    print(f"WM Bayesian:    {wm_bayes:.4f}")
    print("\nHigher score ⇒ more likely watermarked.")


if __name__ == "__main__":
    main()
