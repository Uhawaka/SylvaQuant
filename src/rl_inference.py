#!/usr/bin/env python3 -u
"""
RL inference for online paper trading.
Wraps AE encoder + DeltaSoftThresh policy for real-time inference.

Flow:
  raw OHLCV → compute_features (62 per coin) → build 558-dim matrix
  → normalize → AE encode → 16-dim latents
  → DeltaSoft policy (with w_prev) → 9-dim weights
"""
import warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn

from pipeline_cpcv import compute_features, load_binance, SYMBOLS, FEATS

# ── Constants ──
NC = 9          # number of coins
DZ = 16         # latent dimension
N_FEATS = len(FEATS)  # features per coin (62)
INPUT_DIM = N_FEATS * NC  # 558

DATA_DIR = ROOT / 'data'
AE_PATH = DATA_DIR / 'market_latent_ae.pt'
NORM_PATH = DATA_DIR / 'latent_norm.npz'
POLICY_PATH = ROOT / 'model' / 'rl_policy.pt'

FEE = 0.0004

# ═══════════════════════════════════════════
# AE (mirrors train_market_latent.py)
# ═══════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )
    def forward(self, x):
        return nn.functional.silu(self.net(x) + x)

class MarketLatentAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden=256, depth=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        enc = [nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth):
            enc.append(ResBlock(hidden, dropout))
        enc += [nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.SiLU(), nn.Dropout(dropout)]
        enc.append(nn.Linear(hidden//2, latent_dim))
        self.encoder = nn.Sequential(*enc)
    def forward(self, x):
        return self.encoder(x)

# ═══════════════════════════════════════════
# DeltaSoft RL Policy
# ═══════════════════════════════════════════
class DeltaSoftThresh(nn.Module):
    def __init__(self, theta_max=0.18):
        super().__init__()
        self.theta_max = theta_max
        H = 16
        self.encoder = nn.Sequential(
            nn.Linear(DZ + NC, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
        )
        self.score_head = nn.Linear(H, NC)
        self.thresh_head = nn.Linear(H, NC)
    def forward(self, z, w_prev):
        x = torch.cat([z, w_prev], dim=-1)
        h = self.encoder(x)
        score = self.score_head(h)
        theta = torch.sigmoid(self.thresh_head(h)) * self.theta_max
        delta = score.sign() * (score.abs() - theta).clamp(min=0)
        return (w_prev + delta).clamp(-1, 1)

# ═══════════════════════════════════════════
# Feature → Latent encoder
# ═══════════════════════════════════════════
class FeatureEncoder:
    """
    Encodes raw OHLCV data into 16-dim market latents.
    
    Usage:
        encoder = FeatureEncoder()
        latent = encoder.encode(ohlcv_data_dict)  # (16,) numpy
    """
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load AE
        ckpt = torch.load(AE_PATH, map_location=self.device)
        self.ae = MarketLatentAE(
            input_dim=ckpt.get('input_dim', INPUT_DIM),
            latent_dim=DZ
        ).to(self.device)
        self.ae.load_state_dict(ckpt['model_state'], strict=False)
        self.ae.eval()
        
        # Load normalization stats
        norm = np.load(NORM_PATH)
        self.mu = torch.from_numpy(norm['mu']).float().to(self.device)
        self.sd = torch.from_numpy(norm['sd']).float().to(self.device)
        
        self._feats_cache = {}  # cache computed features per symbol
    
    def compute_features_for_all(self, df_dict):
        """Compute 62 features for all 9 coins, align to common dates."""
        feat_dict = {}
        date_dict = {}
        for sym in SYMBOLS:
            if sym in df_dict and df_dict[sym] is not None and len(df_dict[sym]) > 200:
                df = df_dict[sym]
                df, fn = compute_features(df)
                feat_df = df[fn].iloc[192:].reset_index(drop=True)
                feat_dict[sym] = feat_df.to_numpy(np.float32)
                date_dict[sym] = pd.to_datetime(df['date'].iloc[192:].values)
            else:
                # Use cached if available
                if sym in self._feats_cache:
                    feat_dict[sym] = self._feats_cache[sym]['feats']
                    date_dict[sym] = self._feats_cache[sym]['dates']
        
        # Find common dates
        all_dates = list(date_dict.values())
        if not all_dates:
            return None
        if len(all_dates) == 1:
            common = sorted(all_dates[0])
        else:
            common = sorted(set.intersection(*[set(d) for d in all_dates]))
        
        if len(common) == 0:
            return None
        
        # Build feature matrix
        N = len(common)
        X = np.zeros((N, INPUT_DIM), np.float32)
        dl = {d: i for i, d in enumerate(common)}
        for j, sym in enumerate(SYMBOLS):
            if sym not in feat_dict:
                continue
            offset = j * N_FEATS
            for k, dt in enumerate(date_dict[sym]):
                idx = dl.get(dt)
                if idx is not None:
                    X[idx, offset:offset + N_FEATS] = feat_dict[sym][k]
        
        return X, common
    
    @torch.no_grad()
    def encode(self, X):
        """X: (N, 558) numpy → (N, 16) numpy latents"""
        x = torch.from_numpy(X).float().to(self.device)
        x_norm = (x - self.mu) / self.sd
        x_norm = torch.nan_to_num(x_norm)
        z = self.ae(x_norm)
        return z.cpu().numpy()

# ═══════════════════════════════════════════
# RL Inference
# ═══════════════════════════════════════════
class RLInferrer:
    """
    Online RL policy inference with w_prev tracking.
    
    Usage:
        rl = RLInferrer()
        weights = rl.step(latent)  # returns (9,) numpy in [-1, 1]
    """
    def __init__(self, policy_path=POLICY_PATH, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        ckpt = torch.load(policy_path, map_location=self.device)
        self.theta_max = ckpt.get('theta_max', 0.18)
        self.policy = DeltaSoftThresh(theta_max=self.theta_max).to(self.device)
        
        # Handle different checkpoint formats
        if 'model_state' in ckpt:
            self.policy.load_state_dict(ckpt['model_state'])
        else:
            self.policy.load_state_dict(ckpt)
        
        self.policy.eval()
        self.w_prev = torch.zeros(1, NC, device=self.device)
    
    @torch.no_grad()
    def step(self, latent):
        """
        latent: (16,) numpy array from AE
        returns: (9,) numpy array of weights in [-1, 1]
        """
        z = torch.from_numpy(latent).float().to(self.device).unsqueeze(0)
        w_new = self.policy(z, self.w_prev)  # (1, NC)
        self.w_prev = w_new
        return w_new.squeeze(0).cpu().numpy()
    
    def reset(self):
        """Reset w_prev to zeros (e.g. at start of trading day)."""
        self.w_prev = torch.zeros(1, NC, device=self.device)

# ═══════════════════════════════════════════
# Main: stand-alone RL signal computation
# ═══════════════════════════════════════════
if __name__ == '__main__':
    from datetime import datetime
    
    print('═══ RL Inference Test ═══')
    
    # Load AE encoder
    print('Loading AE...')
    encoder = FeatureEncoder()
    
    # Load data for all coins
    print('Loading data...')
    df_dict = {}
    for sym in SYMBOLS:
        df = load_binance(sym)
        if df is not None and len(df) > 200:
            df_dict[sym] = df
    
    # Compute features and encode
    print('Computing features...')
    result = encoder.compute_features_for_all(df_dict)
    if result is None:
        print('No common data found')
        sys.exit(1)
    
    X, dates = result
    print(f'Feature matrix: {X.shape}')
    
    # Encode to latents
    print('Encoding to latents...')
    latents = encoder.encode(X)
    print(f'Latents: {latents.shape}')
    
    # Load RL policy and infer signals
    print('Loading RL policy...')
    rl = RLInferrer()
    
    # Compute weights for each bar
    weights = []
    for i in range(len(latents)):
        w = rl.step(latents[i])
        weights.append(w)
    
    weights = np.array(weights)
    print(f'Weights: {weights.shape}')
    print(f'Weight range: [{weights.min():.4f}, {weights.max():.4f}]')
    print(f'Mean absolute weight: {np.abs(weights).mean():.4f}')
    
    # Summary per coin
    print(f'\n── Per-coin average weight ──')
    for j, sym in enumerate(SYMBOLS):
        w_mean = weights[:, j].mean()
        w_std = weights[:, j].std()
        w_act = (np.abs(weights[:, j]) > 0.001).mean()
        print(f'  {sym:<10s}  mean={w_mean:>+7.4f}  std={w_std:.4f}  active={w_act:.1%}')
    
    print(f'\n✅ RL inference test complete')
