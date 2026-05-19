#!/usr/bin/env python3 -u
"""
CFM (Conditional Flow Matching) — Synthetic (latent, return) pairs.
Lipman et al. 2022 "Flow Matching for Generative Modeling"
Generates 200K synthetic pairs for Dream World RL training.
"""
import sys, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
DZ, NC = 16, 9

# ── Load real data (for distribution matching) ──
d = np.load(ROOT / 'data/rl_exp/exp_data.npz')
lat = d['latents'].astype(np.float32)  # (N, 16)
ret = d['raw_ret'].astype(np.float32)  # (N, 9)
tr = d['train_idx']

# Use only training data for CFM
lat_tr = lat[tr[0]:tr[1]]  # (132768, 16)
ret_tr = ret[tr[0]:tr[1]]  # (132768, 9)

# Remove zero-return bars (not in CPCV OOS)
mask = np.any(ret_tr != 0, axis=1)
lat_tr = lat_tr[mask]
ret_tr = ret_tr[mask]
print(f'Training pairs: {len(lat_tr):,}')

# ── Normalize ──
lm = lat_tr.mean(0); ls = lat_tr.std(0).clip(1e-6)
rm = ret_tr.mean(0); rs = ret_tr.std(0).clip(1e-6)

lat_norm = (lat_tr - lm) / ls
ret_norm = (ret_tr - rm) / rs

# Combine into one joint distribution for CFM
joint = np.concatenate([lat_norm, ret_norm], axis=1).astype(np.float32)  # (N, 25)
DZ_R = joint.shape[1]  # 25

print(f'Joint dim: {DZ_R}, Latent mean/std: {lm[:3].round(3)}/{ls[:3].round(3)}')

# ── CFM Model ──
class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding (same as DDPM)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class CFM(nn.Module):
    """Time-conditioned MLP that predicts velocity field."""
    def __init__(self, in_dim, H=256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(H // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim + H // 4, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU(),
            nn.Linear(H, in_dim)
        )
    def forward(self, x_t, t):
        te = self.time_embed(t)
        return self.net(torch.cat([x_t, te], dim=1))

def train_cfm(model, data, n_steps=20000, batch_size=4096, lr=3e-4):
    """Train CFM with conditional flow matching loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    
    X = torch.from_numpy(data).to(DEV)
    N = len(X)
    
    for step in range(n_steps):
        # Sample data points
        idx = torch.randint(0, N, (batch_size,), device=DEV)
        x1 = X[idx]  # target: real data point
        
        # Sample noise
        x0 = torch.randn_like(x1)  # source: Gaussian noise
        
        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=DEV)  # t in [0, 1]
        
        # Linear interpolation between noise and data
        x_t = (1 - t) * x0 + t * x1
        
        # Conditional velocity: d/dt x_t = x1 - x0
        u_t = x1 - x0
        
        # Predict velocity
        v_t = model(x_t, t.squeeze(1))
        
        # MSE loss on velocity prediction
        loss = F.mse_loss(v_t, u_t)
        
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        
        if step % 5000 == 0 or step == 0:
            print(f'  Step {step:>6d}  loss={loss.item():.6f}')
    
    print(f'  CFM training done: {time.time()-t0:.0f}s')
    return model

def sample_cfm(model, n_samples=200000):
    """Generate samples from trained CFM via ODE integration."""
    model.eval()
    
    # Euler integration from t=0 to t=1
    n_steps = 100
    dt = 1.0 / n_steps
    
    # Start from noise
    x = torch.randn(n_samples, DZ_R, device=DEV)
    
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt, device=DEV)
            v = model(x, t)
            x = x + v * dt
    
    return x.cpu().numpy()

# ════════════════ RUN ════════════════
print(f'═══ CFM — Synthetic (latent, return) Generation ═══')
print(f'Device: {DEV}, Joint dim: {DZ_R}')

model = CFM(DZ_R).to(DEV)
print(f'\nTraining CFM...')
model = train_cfm(model, joint)

print(f'\nSampling 200K synthetic pairs...')
samples = sample_cfm(model, 200000)

# Split back into latent + return
lat_syn = samples[:, :DZ]
ret_syn = samples[:, DZ:]

# Un-normalize
lat_syn = lat_syn * ls + lm
ret_syn = ret_syn * rs + rm

# Clip returns to realistic range
ret_syn = np.clip(ret_syn, -0.05, 0.05)

print(f'\nSynthetic stats:')
print(f'  Latent: mean={lat_syn.mean(0)[:3].round(4)} std={lat_syn.std(0)[:3].round(4)}')
print(f'  Returns: mean={ret_syn.mean():.6f} std={ret_syn.std():.6f}')
print(f'  Real returns: mean={ret_tr.mean():.6f} std={ret_tr.std():.6f}')

# Save
np.savez(ROOT / 'data' / 'synthetic_cfm.npz',
         latent=lat_syn.astype(np.float32),
         returns=ret_syn.astype(np.float32))
torch.save({'model_state': model.state_dict(), 'data_mean': (rm, rs, lm, ls)},
           ROOT / 'data' / 'cfm_joint.pt')

print(f'\n✅ Saved: data/synthetic_cfm.npz (200K pairs)')
print(f'✅ Saved: data/cfm_joint.pt (model)')
