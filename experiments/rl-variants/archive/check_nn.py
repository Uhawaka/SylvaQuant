#!/usr/bin/env python3 -u
"""Check: is NN overfitting or is there a bug? Compare linear vs NN."""
import sys,warnings
from pathlib import Path
import numpy as np
import torch,torch.nn as nn
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/'src'))

d=np.load(ROOT/'data/rl_exp/exp_data.npz')
lat=d['latents'].astype(np.float32)
ret=d['raw_ret'].astype(np.float32)
tr,va,te=d['train_idx'],d['val_idx'],d['test_idx']
ANNUAL=np.sqrt(24*365)
DEV='mps'if torch.backends.mps.is_available()else'cpu'

# Normalize
lm=lat[:tr[1]].mean(0,keepdims=True)
ls=lat[:tr[1]].std(0,keepdims=True).clip(1e-6)
lat_n=((lat-lm)/ls).astype(np.float32)

DZ,NC=16,9

# Test 1: NN on shuffled returns (should give SR≈0)
print("=== Test 1: NN on SHUFFLED returns ===")
ret_shuf=ret.copy()
np.random.seed(99)
np.random.shuffle(ret_shuf)

H=128
pi=nn.Sequential(nn.Linear(DZ,H),nn.SiLU(),nn.Linear(H,H),nn.SiLU(),nn.Linear(H,NC),nn.Tanh()).to(DEV)
opt=torch.optim.AdamW(pi.parameters(),lr=3e-4)
lt=torch.from_numpy(lat_n).to(DEV)
rs=torch.from_numpy(ret_shuf).to(DEV)
B=8192;STEPS=5000

for step in range(STEPS):
    ts=np.random.randint(tr[0],tr[1]-B-1)
    z=lt[ts:ts+B];r=rs[ts:ts+B]
    w=pi(z)
    pr=(w*r).sum(1)
    loss=-pr.mean()/(pr.std()+1e-8)
    opt.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(pi.parameters(),0.5);opt.step()
    if(step+1)%1000==0:
        with torch.no_grad():
            wv=pi(lt[va[0]:va[1]])
            pv=(wv*rs[va[0]:va[1]]).sum(1).cpu().numpy()
            sv=pv.mean()/max(pv.std(),1e-8)*ANNUAL
        print(f"  Shuffled Step {step+1}: Val SR={sv:.2f}")

# Test 2: NN on real returns (reproduce run_all.py exp2)
print("\n=== Test 2: NN on REAL returns ===")
pi2=nn.Sequential(nn.Linear(DZ,H),nn.SiLU(),nn.Linear(H,H),nn.SiLU(),nn.Linear(H,NC),nn.Tanh()).to(DEV)
opt2=torch.optim.AdamW(pi2.parameters(),lr=3e-4)
rt=torch.from_numpy(ret).to(DEV)

best_val=-10
for step in range(10000):
    ts=np.random.randint(tr[0],tr[1]-B-1)
    z=lt[ts:ts+B];r=rt[ts:ts+B]
    w=pi2(z)
    pr=(w*r).sum(1)
    loss=-pr.mean()/(pr.std()+1e-8)
    opt2.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(pi2.parameters(),0.5);opt2.step()
    if(step+1)%1000==0:
        with torch.no_grad():
            wv=pi2(lt[va[0]:va[1]])
            pv=(wv*rt[va[0]:va[1]]).sum(1).cpu().numpy()
            sv=pv.mean()/max(pv.std(),1e-8)*ANNUAL
            if sv>best_val:best_val=sv
        print(f"  Real Step {step+1}: Val SR={sv:.2f}")

with torch.no_grad():
    wt=pi2(lt[te[0]:te[1]])
    pt=(wt*rt[te[0]:te[1]]).sum(1).cpu().numpy()
    st=pt.mean()/max(pt.std(),1e-8)*ANNUAL
print(f"Real NN: Best Val={best_val:.2f}  Test={st:.2f}")

# Test 3: Smaller NN (reduced capacity)
print("\n=== Test 3: Small NN (H=16) on REAL returns ===")
H3=16
pi3=nn.Sequential(nn.Linear(DZ,H3),nn.SiLU(),nn.Linear(H3,H3),nn.SiLU(),nn.Linear(H3,NC),nn.Tanh()).to(DEV)
opt3=torch.optim.AdamW(pi3.parameters(),lr=3e-4)

for step in range(10000):
    ts=np.random.randint(tr[0],tr[1]-B-1)
    z=lt[ts:ts+B];r=rt[ts:ts+B]
    w=pi3(z)
    pr=(w*r).sum(1)
    loss=-pr.mean()/(pr.std()+1e-8)
    opt3.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(pi3.parameters(),0.5);opt3.step()
    if(step+1)%1000==0:
        with torch.no_grad():
            wv=pi3(lt[va[0]:va[1]])
            pv=(wv*rt[va[0]:va[1]]).sum(1).cpu().numpy()
            sv=pv.mean()/max(pv.std(),1e-8)*ANNUAL
        print(f"  Small NN Step {step+1}: Val SR={sv:.2f}")

with torch.no_grad():
    wt=pi3(lt[te[0]:te[1]])
    pt=(wt*rt[te[0]:te[1]]).sum(1).cpu().numpy()
    st=pt.mean()/max(pt.std(),1e-8)*ANNUAL
print(f"Small NN: Test={st:.2f}")
