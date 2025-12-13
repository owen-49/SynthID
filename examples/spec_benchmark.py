#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 3.3: Speculative sampling benchmark (speed vs detectability).

For each draft_steps in STEPS_LIST:
  - generate M samples with speculative+watermark
For baseline (draft_steps=0):
  - generate M samples with standard watermark generator

Measure:
  - avg_time_sec per sample
  - avg_mean_g
  - avg_bayes

Outputs:
  - table
  - plot 'spec_benchmark.png'
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from synthid import SynthIDTextWatermarker, MeanGScorer, BayesianScorer
from synthid.seed import RandomSeedGenerator
from synthid.gvalue import GValueGenerator
from synthid.speculative import WatermarkedSpeculativeGenerator
from synthid.utils import set_all_seeds


def now():
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_target", type=str, default="gpt2")
    parser.add_argument("--model_draft", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default="Explain the benefits of exercise.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--M", type=int, default=10, help="repeats per setting")
    parser.add_argument("--steps_list", type=str, default="0,1,2,4,8", help="0=baseline")
    parser.add_argument("--out", type=str, default="spec_benchmark.png")
    args = parser.parse_args()

    set_all_seeds(2025)
    device = args.device
    STEPS_LIST = [int(x.strip()) for x in args.steps_list.split(",") if x.strip()]

    print(f"Loading target {args.model_target} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.model_target)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(args.model_target).to(device)
    target_model.eval()

    print(f"Loading draft {args.model_draft} on {device}...")
    draft_model = AutoModelForCausalLM.from_pretrained(args.model_draft).to(device)
    draft_model.eval()

    # init watermark
    key = os.urandom(32)
    wm = SynthIDTextWatermarker(key=key, window_size=4, num_layers=4)
    spec_gen = WatermarkedSpeculativeGenerator(key=key, window_size=4, num_layers=4)

    seed_gen = RandomSeedGenerator(key=key, window_size=4)
    g_gen = GValueGenerator(key=key)
    mean_scorer = MeanGScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)
    bayes_scorer = BayesianScorer(seed_gen=seed_gen, g_gen=g_gen, num_layers=4)

    # calibrate Bayesian on a small batch of watermarked texts (baseline generator)
    print("\nCalibrating Bayesian scorer...")
    calib = []
    for _ in range(10):
        t = wm.generate(
            model=target_model,
            tokenizer=tok,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=1.0,
            top_k=50,
            device=device,
        )
        calib.append(t)
    bayes_scorer.fit_from_texts(calib, tok)
    print(f"Estimated p1 = {bayes_scorer.p1:.4f}")

    # Warmup (reduce first-run overhead)
    print("\nWarming up...")
    _ = wm.generate(
        model=target_model,
        tokenizer=tok,
        prompt=args.prompt,
        max_new_tokens=8,
        temperature=1.0,
        top_k=50,
        device=device,
    )

    results = []  # (draft_steps, avg_time, avg_mean_g, avg_bayes)

    for steps in STEPS_LIST:
        mode = "baseline" if steps == 0 else f"spec({steps})"
        print(f"\n=== {mode} : running M={args.M} samples ===")

        times = []
        mean_scores = []
        bayes_scores = []

        for _ in range(args.M):
            t0 = now()
            if steps == 0:
                text = wm.generate(
                    model=target_model,
                    tokenizer=tok,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=1.0,
                    top_k=50,
                    device=device,
                )
            else:
                text = spec_gen.generate(
                    target_model=target_model,
                    draft_model=draft_model,
                    tokenizer=tok,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    draft_steps=steps,
                    temperature=1.0,
                    top_k=50,
                    device=device,
                )
            t1 = now()

            times.append(t1 - t0)
            mean_scores.append(mean_scorer.score_text(text, tok))
            bayes_scores.append(bayes_scorer.score_text(text, tok))

        avg_time = float(np.mean(times))
        avg_mean = float(np.mean(mean_scores))
        avg_bayes = float(np.mean(bayes_scores))

        print(f"avg_time_sec: {avg_time:.4f} | avg_mean_g: {avg_mean:.4f} | avg_bayes: {avg_bayes:.4f}")
        results.append((steps, avg_time, avg_mean, avg_bayes))

    # summary table
    print("\n=== Summary: Speculative benchmark ===")
    print("draft_steps | avg_time_sec | avg_mean_g | avg_bayes")
    print("----------+--------------+-----------+----------")
    for steps, tsec, mg, bz in results:
        print(f"{steps:10d} | {tsec:12.4f} | {mg:9.4f} | {bz:8.4f}")

    # plot: time vs steps + mean/bayes vs steps
    steps_list = [r[0] for r in results]
    times = [r[1] for r in results]
    means = [r[2] for r in results]
    bayes = [r[3] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(steps_list, times, marker="o", label="avg_time_sec (lower is better)")
    plt.xlabel("draft_steps (0 = baseline)")
    plt.ylabel("avg_time_sec per sample")
    plt.title("Speculative Sampling: Speed")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("spec_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(steps_list, means, marker="o", label="avg Mean-g")
    plt.plot(steps_list, bayes, marker="o", label="avg Bayesian")
    plt.xlabel("draft_steps (0 = baseline)")
    plt.ylabel("score")
    plt.ylim(0.0, 1.05)
    plt.title("Speculative Sampling: Detectability")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("spec_detectability.png", dpi=200)
    plt.close()

    # Combined plot (optional): normalize time by baseline
    base_time = None
    for steps, tsec, _, _ in results:
        if steps == 0:
            base_time = tsec
            break
    if base_time is None:
        base_time = times[0]

    speedup = [base_time / t for t in times]  # >1 means faster
    plt.figure(figsize=(7, 5))
    plt.plot(steps_list, speedup, marker="o", label="speedup vs baseline (>1 faster)")
    plt.xlabel("draft_steps (0 = baseline)")
    plt.ylabel("speedup")
    plt.title("Speculative Sampling: Speedup Curve")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"\nSaved plots: spec_time.png, spec_detectability.png, {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
