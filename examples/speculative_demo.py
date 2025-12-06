#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo: plain watermark vs speculative+watermark.

对比：
- 普通 SynthIDTextWatermarker.generate()
- WatermarkedSpeculativeGenerator.generate()
并且用 Bayesian scorer 打分。
"""

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
from synthid.speculative import WatermarkedSpeculativeGenerator
from synthid.utils import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_target", type=str, default="gpt2", help="target LM")
    parser.add_argument("--model_draft", type=str, default="gpt2", help="draft LM (can be smaller)")
    parser.add_argument("--prompt", type=str, default="Explain the benefits of exercise.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_all_seeds(2025)
    device = args.device

    # ---------- load target & draft ----------
    print(f"Loading target model {args.model_target} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.model_target)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(args.model_target).to(device)

    print(f"Loading draft model {args.model_draft} on {device}...")
    draft_model = AutoModelForCausalLM.from_pretrained(args.model_draft).to(device)

    # ---------- init watermarking ----------
    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)
    seed_gen = RandomSeedGenerator(key=key, window_size=4)
    g_gen = GValueGenerator(key=key)
    mean_scorer = MeanGScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)
    bayes_scorer = BayesianScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)

    # 用几条水印文本校准 p1
    calib_texts = []
    for _ in range(10):
        t = wm.generate(
            model=target_model,
            tokenizer=tok,
            prompt=args.prompt,
            max_new_tokens=80,
            temperature=1.0,
            top_k=50,
            device=device,
        )
        calib_texts.append(t)
    bayes_scorer.fit_from_texts(calib_texts, tok)

    # ---------- plain watermark generation ----------
    print("\n=== Plain watermark generation ===")
    plain_wm = wm.generate(
        model=target_model,
        tokenizer=tok,
        prompt=args.prompt,
        max_new_tokens=80,
        temperature=1.0,
        top_k=50,
        device=device,
    )
    print(plain_wm)
    print("\nScores for plain watermark:")
    print("Mean-g:   {:.4f}".format(mean_scorer.score_text(plain_wm, tok)))
    print("Bayesian: {:.4f}".format(bayes_scorer.score_text(plain_wm, tok)))

    # ---------- speculative + watermark ----------
    print("\n=== Speculative + watermark generation ===")
    spec_gen = WatermarkedSpeculativeGenerator(key=key, window_size=4, num_layers=4)
    spec_text = spec_gen.generate(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tok,
        prompt=args.prompt,
        max_new_tokens=80,
        draft_steps=4,
        temperature=1.0,
        top_k=50,
        device=device,
    )
    print(spec_text)
    print("\nScores for speculative watermark:")
    print("Mean-g:   {:.4f}".format(mean_scorer.score_text(spec_text, tok)))
    print("Bayesian: {:.4f}".format(bayes_scorer.score_text(spec_text, tok)))


if __name__ == "__main__":
    main()
