#!/usr/bin/env python3 -u
"""Exp B: Differentiable Sharpe — No RL algo, direct Sharpe optimization via autograd."""
import sys,warnings,time
from pathlib import Path
import numpy as np
import torch,torch.nn as nn
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/rl-variants'))
from shared import DiffSharpePolicy as Policy,compute_sharpe,load_data,DEV,NC,DZ

LR=3e-4;N_STEPS=100000;BATCH=2048;EVAL=5000;WINDOW=1024  # Sharpe calc window
SEED=42
torch.manual_seed(SEED);np.random.seed(SEED)

print(f'═══ Exp B: Differentiable Sharpe ═══\nDevice: {DEV}')

data=load_data()
lat=torch.from_numpy(data['latents']).to(DEV)
ret=torch.from_numpy(data['raw_ret']).to(DEV)
te=data['train_idx'][1];vs,ve=data['val_idx']
print(f'Train: 0:{te}  Val: {vs}:{ve}')

policy=Policy().to(DEV)
opt=torch.optim.AdamW(policy.parameters(),lr=LR,weight_decay=1e-5)

def eval_policy(silent=False):
    policy.eval()
    with torch.no_grad():
        pw=torch.zeros(1,NC,device=DEV);rs=[]
        for t in range(vs,ve):
            w=policy(lat[t:t+1],pw);pw=w
            rs.append((w*ret[t:t+1]).sum(-1).item())
    sr=compute_sharpe(np.array(rs))
    if not silent:print(f'  Val Sharpe: {sr:.4f}')
    policy.train();return sr

best_sr=-10;t0=time.time()
for step in range(N_STEPS):
    ts=np.random.randint(WINDOW,te-BATCH-1)
    # Rollout BATCH steps with sequential weights
    pw=torch.zeros(1,NC,device=DEV)
    port_ret=[]
    for t in range(BATCH):
        w=policy(lat[ts+t:ts+t+1],pw)
        pw=w
        port_ret.append((w*ret[ts+t:ts+t+1]).sum(-1))
    port_ret=torch.cat(port_ret)
    
    # Differentiable Sharpe over the batch window
    mu=port_ret.mean()
    sd=port_ret.std()+1e-8
    loss=-mu/sd  # minimize negative Sharpe
    
    opt.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(),1.0);opt.step()
    
    if step%2000==0:
        print(f'  step{step:6d} | loss={loss.item():.4f} | mean_r={mu.item():.6f} | SR_batch={(-loss.item())*np.sqrt(24*365):.2f}')
    
    if step%EVAL==0 and step>0:
        sr=eval_policy()
        if sr>best_sr:
            best_sr=sr
            torch.save(policy.state_dict(),ROOT/'data/rl_exp/exp_b_policy.pt')
            print(f'  -> Best Val Sharpe: {best_sr:.4f}')

print(f'\n{"="*50}\nTrain: {time.time()-t0:.0f}s')
print(f'Best Val Sharpe: {best_sr:.4f}')

# Test
policy.eval()
with torch.no_grad():
    pw=torch.zeros(1,NC,device=DEV);trs=[]
    for t in range(data['test_idx'][0],data['test_idx'][1]):
        w=policy(lat[t:t+1],pw);pw=w
        trs.append((w*ret[t:t+1]).sum(-1).item())
tsr=compute_sharpe(np.array(trs))
ew=np.array(data['raw_ret'][data['test_idx'][0]:data['test_idx'][1]]).mean(1)
print(f'Test Sharpe:       {tsr:.4f}')
print(f'EW Test Sharpe:    {compute_sharpe(ew):.4f}')
