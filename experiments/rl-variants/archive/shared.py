#!/usr/bin/env python3 -u
"""Shared utilities for RL experiment variants."""
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
DZ, NC, H = 16, 9, 128  # latent_dim, n_coins, hidden

def load_data(path='data/rl_exp/exp_data.npz'):
    d = np.load(path)
    return {k: d[k] for k in d}

def compute_sharpe(rets, annual_factor=24*365):
    """Compute annualized Sharpe ratio."""
    return rets.mean() / (rets.std() + 1e-8) * np.sqrt(annual_factor)

class Policy(nn.Module):
    """Shared policy network: latent → weights."""
    def __init__(self, out_dim=9, hidden=H):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DZ + out_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, z, prev_w=None):
        if prev_w is None:
            prev_w = torch.zeros(z.shape[0], self.net[-1].out_features, device=z.device)
        x = torch.cat([z, prev_w], dim=-1)
        return self.net(x)

class SizingPolicy(nn.Module):
    """Outputs a single sizing scalar ∈ [0,1]."""
    def __init__(self, hidden=H):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DZ + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, z, prev_s=None):
        if prev_s is None:
            prev_s = torch.zeros(z.shape[0], 1, device=z.device)
        x = torch.cat([z, prev_s], dim=-1)
        return self.net(x)

class DiffSharpePolicy(nn.Module):
    """Policy for differentiable Sharpe: tanh weights."""
    def __init__(self, hidden=H):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DZ + NC, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, NC), nn.Tanh(),
        )
    def forward(self, z, prev_w=None):
        if prev_w is None:
            prev_w = torch.zeros(z.shape[0], NC, device=z.device)
        x = torch.cat([z, prev_w], dim=-1)
        return self.net(x)
