#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from synthid import SynthIDTextWatermarker, MeanGScorer, BayesianScorer, attacks
from synthid.seed import RandomSeedGenerator
from synthid.gvalue import GValueGenerator
from synthid.utils import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="Explain the benefits of exercise.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    set_all_seeds(123)

    print(f"Loading model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)
    seed_gen = RandomSeedGenerator(key=key, window_size=4)
    g_gen = GValueGenerator(key=key)

    mean_scorer = MeanGScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)
    bayes_scorer = BayesianScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)

    # 用水印文本做 calibration
    calib_texts = []
    for _ in range(10):
        t = wm.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=96,
            temperature=1.0,
            top_k=50,
            device=device,
        )
        calib_texts.append(t)
    bayes_scorer.fit_from_texts(calib_texts, tokenizer)

    # 取其中一条水印文本做攻击实验
    base_text = calib_texts[0]
    print("\n=== Original watermarked text ===")
    print(base_text)

    base_mean = mean_scorer.score_text(base_text, tokenizer)
    base_bayes = bayes_scorer.score_text(base_text, tokenizer)

    print("\n[Original scores]")
    print(f"Mean-g:   {base_mean:.4f}")
    print(f"Bayesian: {base_bayes:.4f}")

    # 转 token
    token_ids = tokenizer.encode(base_text, add_special_tokens=False)
    vocab_size = len(tokenizer)

    # 1) random delete
    del_ids = attacks.random_delete_tokens(token_ids, rate=0.1)
    del_text = tokenizer.decode(del_ids, skip_special_tokens=True)

    # 2) random replace
    rep_ids = attacks.random_replace_tokens(token_ids, vocab_size=vocab_size, rate=0.1)
    rep_text = tokenizer.decode(rep_ids, skip_special_tokens=True)

    # 3) truncate
    trunc_ids = attacks.truncate_tokens(token_ids, keep_ratio=0.5)
    trunc_text = tokenizer.decode(trunc_ids, skip_special_tokens=True)

    # 4) shuffle sentences
    shuf_text = attacks.shuffle_sentences(base_text)

    print("\n=== Attacks & scores ===")

    def show_case(name, text):
        mean = mean_scorer.score_text(text, tokenizer)
        bayes = bayes_scorer.score_text(text, tokenizer)
        print(f"\n[{name}]")
        print(text[:200].replace("\n", " ") + ("..." if len(text) > 200 else ""))
        print(f"Mean-g:   {mean:.4f}")
        print(f"Bayesian: {bayes:.4f}")

    show_case("Random delete (10%)", del_text)
    show_case("Random replace (10%)", rep_text)
    show_case("Truncate 50%", trunc_text)
    show_case("Shuffle sentences", shuf_text)

    # 如果你有一个 paraphraser 模型，也可以解除注释做 paraphrase attack
    # paraphrased = attacks.paraphrase_with_model(base_text, tokenizer, model, device=device)
    # show_case("Paraphrased", paraphrased)


if __name__ == "__main__":
    main()
