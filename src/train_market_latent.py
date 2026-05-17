#!/usr/bin/env python3 -u
"""Stage 1: Train Market Latent AE.

Compress 62 features × 9 coins → 16-dim unified market representation.
Uses CPCV-aligned OOS signals (157K bars common across all coins).

Output: data/market_latent_ae.pt (model), data/market_latent.npy (16-dim latents)
"""
import sys, warnings, time, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline_cpcv import load_binance, compute_features, SYMBOLS, FEATS

# ── Config ──
LATENT_DIM = 16
HIDDEN = 256
DEPTH = 2
DROPOUT = 0.1
BATCH_SIZE = 2048
LR = 5e-4
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = Path('data')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Dumbbell AE ──
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
        # Encoder
        enc = [
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout),
        ]
        for _ in range(depth):
            enc.append(ResBlock(hidden, dropout))
        enc += [
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.SiLU(), nn.Dropout(dropout),
        ]
        enc.append(nn.Linear(hidden // 2, latent_dim))
        self.encoder = nn.Sequential(*enc)
        # Decoder
        dec = [
            nn.Linear(latent_dim, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.SiLU(), nn.Dropout(dropout),
        ]
        for _ in range(depth):
            dec.append(ResBlock(hidden // 2, dropout))
        dec += [
            nn.Linear(hidden // 2, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout),
        ]
        dec.append(nn.Linear(hidden, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# ── Prepare aligned features ──
print('═══ Stage 1: Market Latent AE ═══')
print(f'Device: {DEVICE}, Latent dim: {LATENT_DIM}\n')

# Load and align 62 features for all coins
all_feats = {}
all_dates = {}
for sym in SYMBOLS:
    df = load_binance(sym)
    df, fn = compute_features(df)
    feat_df = df[fn].iloc[192:].reset_index(drop=True)
    all_feats[sym] = feat_df.to_numpy(np.float32)
    all_dates[sym] = pd.to_datetime(df['date'].iloc[192:].values)

common = sorted(set.intersection(*[set(d) for d in all_dates.values()]))
N = len(common)
dl = {d: i for i, d in enumerate(common)}
print(f'Common aligned bars: {N:,}')

# Build feature matrix: N × (62 × 9)
n_feats = len(FEATS)
X = np.zeros((N, n_feats * len(SYMBOLS)), np.float32)
for j, sym in enumerate(SYMBOLS):
    offset = j * n_feats
    for k, dt in enumerate(all_dates[sym]):
        idx = dl.get(dt)
        if idx is not None:
            X[idx, offset:offset + n_feats] = all_feats[sym][k]

print(f'Feature matrix shape: {X.shape} ({N:,} bars × {X.shape[1]:,} dims)')

# Normalize per column
mu = np.nanmean(X, axis=0)
sd = np.nanstd(X, axis=0).clip(1e-8)
X_norm = (X - mu) / sd
X_norm = np.nan_to_num(X_norm)

# Save normalization stats
np.savez(OUTPUT_DIR / 'latent_norm.npz', mu=mu, sd=sd)

# ── Train AE ──
dataset = TensorDataset(torch.from_numpy(X_norm))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = MarketLatentAE(X.shape[1], LATENT_DIM, HIDDEN, DEPTH, DROPOUT).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

t0 = time.time()
print(f'\nTraining...  ({EPOCHS} epochs × {len(loader)} batches/epoch)')
for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0.0
    for batch in loader:
        x = batch[0].to(DEVICE)
        # Denoising
        x_noisy = x + torch.randn_like(x) * 0.05
        recon, z = model(x_noisy)
        loss = nn.functional.mse_loss(recon, x)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * len(x)
    scheduler.step()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'  Epoch {epoch + 1:>3d}/{EPOCHS}  loss={loss_sum / len(dataset):.6f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}')

elapsed = time.time() - t0
print(f'\nTraining done: {elapsed:.0f}s')

# ── Extract latents ──
model.eval()
all_z = []
with torch.no_grad():
    for i in range(0, len(X_norm), BATCH_SIZE):
        batch = torch.from_numpy(X_norm[i:i + BATCH_SIZE]).to(DEVICE)
        _, z = model(batch)
        all_z.append(z.cpu().numpy())
latents = np.concatenate(all_z, axis=0)
print(f'Latents shape: {latents.shape}')

# Save
torch.save({'model_state': model.state_dict(), 'input_dim': X.shape[1]},
           OUTPUT_DIR / 'market_latent_ae.pt')
np.save(OUTPUT_DIR / 'market_latent.npy', latents)
np.save(OUTPUT_DIR / 'market_latent_dates.npy', np.array(common))
print(f'\nSaved:')
print(f'  data/market_latent_ae.pt  ({Path(OUTPUT_DIR / "market_latent_ae.pt").stat().st_size // 1024} KB)')
print(f'  data/market_latent.npy     ({Path(OUTPUT_DIR / "market_latent.npy").stat().st_size // 1024} KB)')
print(f'  data/market_latent_dates.npy')
print(f'  data/latent_norm.npz')

# Stats
print(f'\nLatent stats (16 dims):')
for i in range(LATENT_DIM):
    print(f'  z_{i:02d}: mean={latents[:, i].mean():+.4f} std={latents[:, i].std():.4f}')
print(f'\nStage 1 complete. Latent is 16-dim market state, aligned with CPCV OOS bars.')
