#!/usr/bin/env python3 -u
"""Backtest RL Policy — raw weights, fee on turnover."""
import sys,warnings
import numpy as np,pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
import torch,torch.nn as nn
from pipeline_cpcv import SYMBOLS,OUTPUT_DIR

SEED=42;torch.manual_seed(SEED);np.random.seed(SEED)
DEV='mps'if torch.backends.mps.is_available()else'cpu'
NC=len(SYMBOLS);FEE=0.0004

class GP(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(16,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.mu_head=nn.Linear(128,NC)
    def forward(self,s):return torch.tanh(self.mu_head(self.shared(s)))

ckpt=torch.load('data/rl_policy.pt',map_location='cpu',weights_only=False)
pi=GP().to(DEV);pi.load_state_dict(ckpt['model_state'],strict=False);pi.eval()
rlm,rls=ckpt['latent_mean'],ckpt['latent_std']

lat=np.load('data/market_latent.npy').astype(np.float32)
ld=np.load('data/market_latent_dates.npy',allow_pickle=True)
dm={d:i for i,d in enumerate(ld)}
vm=np.zeros((len(lat),NC),np.float32)
for j,s in enumerate(SYMBOLS):
    v=np.load(OUTPUT_DIR/f'cpcv_vwap_{s}.npy')
    ds=pd.to_datetime(np.load(OUTPUT_DIR/f'cpcv_dates_{s}.npy'))
    for k,dt in enumerate(ds):
        i=dm.get(dt)
        if i is not None:vm[i,j]=v[k]
N_ret=len(lat)-2
ret=np.zeros((N_ret,NC),np.float32)
for j in range(NC):
    r=vm[2:,j]/vm[1:-1,j]-1.;ret[:,j]=np.where(np.isfinite(r),r,0.)

lat_n=((lat[:N_ret]-rlm)/rls.clip(1e-6)).astype(np.float32)
with torch.no_grad():
    w=pi(torch.from_numpy(lat_n).to(DEV)).cpu().numpy()

print(f'═══ RL Backtest (raw weights, fee={FEE}) ═══')
print(f'Weights: {w.shape}, range=[{w.min():.3f},{w.max():.3f}]')
sw=np.abs(w).sum(1).mean();to=np.abs(np.diff(w,axis=0)).sum(1).mean()
print(f'Σ|w|={sw:.2f}, TO={to:.4f}')

# Forward eval (no fee)
pnl_raw=(w*ret).sum(1)
sr0=pnl_raw.mean()/(pnl_raw.std()+1e-8)*np.sqrt(252*96)
print(f'No fee: SR={sr0:.2f}  PnL/bar={pnl_raw.mean():.6f}')

# Full backtest with fee
eq=[10000.0]
for t in range(N_ret):
    fee=FEE*(np.abs(w[t]).sum() if t==0 else np.abs(w[t]-w[t-1]).sum())
    eq.append(eq[-1]*(1+np.dot(w[t],ret[t])-fee))
eq=np.array(eq)
pnl_s=np.diff(eq)
sr=pnl_s.mean()/(pnl_s.std()+1e-8)*np.sqrt(252*96)
peak=np.maximum.accumulate(eq)
dd=(eq-peak)/peak

ew=ret.mean(1);ew_eq=10000*np.cumprod(1+ew)
ew_sr=ew.mean()/(ew.std()+1e-8)*np.sqrt(252*96)

print(f'\n═══ Results (fee={FEE}) ═══')
print(f'  Initial:  $10,000')
print(f'  Final:    ${eq[-1]:>12,.0f}')
print(f'  Return:   {(eq[-1]/10000-1)*100:>8.1f}%')
print(f'  SR:       {sr:.2f}')
print(f'  Max DD:   {dd.min():.1%}')
print(f'  EW Final: ${ew_eq[-1]:>12,.0f} (SR={ew_sr:.2f})')

print(f'\nAvg weights:')
for j,sym in enumerate(SYMBOLS):
    print(f'  {sym[:8]:<10} {w[:,j].mean():+.3f}')

np.savez('data/backtest_rl_results.npz',equity=eq,weights=w)
print(f'\nSaved: data/backtest_rl_results.npz')