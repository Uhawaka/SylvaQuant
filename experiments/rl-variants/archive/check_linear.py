#!/usr/bin/env python3 -u
"""Focused: verify exp_data.npz alignment after fix, and check what policy learns."""
import sys,warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/'src'))

# Load data
d=np.load(ROOT/'data/rl_exp/exp_data.npz')
lat=d['latents'].astype(np.float32)
ret=d['raw_ret'].astype(np.float32)
tr,va,te=d['train_idx'],d['val_idx'],d['test_idx']
ANNUAL=np.sqrt(24*365)

print("=== exp_data.npz Verification ===")
print(f"lat shape: {lat.shape}, ret shape: {ret.shape}")
print(f"train: {tr}, val: {va}, test: {te}")

# Check alignment: are latents correct?
print(f"\nlat[0:5, :3]:\n{lat[:5, :3]}")
print(f"lat[-5:, :3]:\n{lat[-5:, :3]}")

# EW Sharpe on raw returns per split
for name,sl in [('Train',tr),('Val',va),('Test',te)]:
    ew=ret[sl[0]:sl[1]].mean(1)
    sr=ew.mean()/max(ew.std(),1e-8)*ANNUAL
    print(f"EW {name}: {sr:.4f}")

# Now test: train a tiny Diff Sharpe and see if it's learning real structure
# using a simple per-coin linear model (like the ones in exp2 eval)
print("\n=== Quick RL: Simple Linear Policy ===")
import torch
import torch.nn as nn

DEV='cpu'
torch.manual_seed(42)
DZ,NC,H=16,9,64

class LinPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Linear(DZ,NC)
    def forward(self,s):return torch.tanh(self.net(s))

lt=torch.from_numpy(lat).float()
rt=torch.from_numpy(ret).float()

# Normalize latents (same as run_all.py)
lm=lat[:tr[1]].mean(0,keepdims=True)
ls=lat[:tr[1]].std(0,keepdims=True).clip(1e-6)
lat_n=((lat-lm)/ls).astype(np.float32)
lt_n=torch.from_numpy(lat_n).float()

pi=LinPolicy()
opt=torch.optim.AdamW(pi.parameters(),lr=1e-3)
B=4096;N_STEPS=5000

va_ret=ret[va[0]:va[1]]
te_ret=ret[te[0]:te[1]]

for step in range(N_STEPS):
    ts=np.random.randint(tr[0],tr[1]-B-1)
    z=lt_n[ts:ts+B];r=rt[ts:ts+B]
    w=pi(z)
    pr=(w*r).sum(1)
    loss=-pr.mean()/(pr.std()+1e-8)
    opt.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(pi.parameters(),1.0);opt.step()
    
    if(step+1)%500==0 or step==0:
        with torch.no_grad():
            wv=pi(lt_n[va[0]:va[1]])
            pv=(wv*rt[va[0]:va[1]]).sum(1).numpy()
            sv=pv.mean()/max(pv.std(),1e-8)*ANNUAL
            
            wt=pi(lt_n[te[0]:te[1]])
            pt=(wt*rt[te[0]:te[1]]).sum(1).numpy()
            st=pt.mean()/max(pt.std(),1e-8)*ANNUAL
        
        if sv<-10 or sv>30 or step==0:
            print(f"  Step {step+1:5d}  Val={sv:+.2f}  Test={st:+.2f}")

# Final weights inspection
with torch.no_grad():
    w_all=pi(lt_n).numpy()
print(f"\n=== Trained Policy Weights ===")
print(f"  Weight range: [{w_all.min():.3f}, {w_all.max():.3f}]")
print(f"  Mean weight per coin:")
for c in range(NC):
    print(f"    coin[{c}]: mean={w_all[:,c].mean():+.4f} std={w_all[:,c].std():.4f}")

# Final SR
with torch.no_grad():
    p_all=(pi(lt_n)*rt).sum(1).numpy()
    print(f"\n  Final All-data SR: {p_all.mean()/max(p_all.std(),1e-8)*ANNUAL:.2f}")
    p_test=(pi(lt_n[te[0]:te[1]])*rt[te[0]:te[1]]).sum(1).numpy()
    print(f"  Final Test SR: {p_test.mean()/max(p_test.std(),1e-8)*ANNUAL:.2f}")
