#!/usr/bin/env python3 -u
"""
Train DeltaSoft RL policy with best config.

Usage:
    python final/train.py              # train best config
    python final/train.py --sweep      # sweep θ_max
"""
import sys, time
from pathlib import Path

from env_rl import (
    DeltaSoftThresh, load_data, train, eval_sequential
)
import numpy as np
import torch

DATA_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / 'data/rl_exp/exp_data.npz')

def run_best():
    """Train with the best known config: θ=0.18, L=32, B=256, seed=1111"""
    lt, rt, tr, va, te = load_data(DATA_PATH)
    pi = DeltaSoftThresh(theta_max=0.18)
    result = train(pi, lt, rt, tr, va, te,
                   B=256, L=32, STEPS=10000, seed=1111)
    return result

def run_sweep():
    """Sweep θ_max to find the best threshold."""
    lt, rt, tr, va, te = load_data(DATA_PATH)
    print(f'{"θ":>6s}  {"Test SR":>7s}  {"TO":>7s}  {"Net SR":>7s}  {"Act":>7s}')
    print('─' * 42)
    for theta in [0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25]:
        pi = DeltaSoftThresh(theta_max=theta)
        r = train(pi, lt, rt, tr, va, te,
                  B=256, L=32, STEPS=10000, seed=1111, verbose=False)
        print(f'{theta:>6.2f}  {r["test_sr"]:>7.2f}  {r["test_to"]:>7.4f}  {r["test_net"]:>+7.2f}  {r["test_act"]:>7.2%}')

def run_multiseed(n_seeds=5):
    """Run multiple seeds and report mean ± std."""
    lt, rt, tr, va, te = load_data(DATA_PATH)
    results = []
    seeds = [42, 123, 456, 789, 1111][:n_seeds]
    for seed in seeds:
        pi = DeltaSoftThresh(theta_max=0.18)
        r = train(pi, lt, rt, tr, va, te,
                  B=256, L=32, STEPS=10000, seed=seed, verbose=False)
        results.append(r)
        print(f'  seed={seed:>4d}  SR={r["test_sr"]:.2f}  Net SR={r["test_net"]:+.2f}')
    
    srs = [r['test_sr'] for r in results]
    nets = [r['test_net'] for r in results]
    tos = [r['test_to'] for r in results]
    print(f'  MEAN: SR={np.mean(srs):.2f}±{np.std(srs):.2f}  '
          f'Net SR={np.mean(nets):+.2f}±{np.std(nets):.2f}  TO={np.mean(tos):.4f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--multiseed', type=int, default=0)
    args = parser.parse_args()
    
    t0 = time.time()
    if args.sweep:
        run_sweep()
    elif args.multiseed > 0:
        run_multiseed(args.multiseed)
    else:
        run_best()
    print(f'\nTotal: {time.time()-t0:.0f}s')
