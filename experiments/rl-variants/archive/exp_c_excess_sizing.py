#!/usr/bin/env python3 -u
"""Exp C: Excess Return Sizing (stock-style success pattern).
State: [latent(16), prev_sizing(1)]
Action: 1 scalar sizing ∈ [0, 1]
Portfolio = sizing × RF-signal-directed positions
Reward: excess over equal-weight baseline"""
import sys,warnings,time
from pathlib import Path
import numpy as np
import torch,torch.nn as nn
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/rl-variants'))
from shared import SizingPolicy,compute_sharpe,load_data,DEV,NC,DZ

LR=3e-4;N_STEPS=30000;BATCH=2048;K=32;CLIP=0.2;ENT=0.01;EVAL=2000;SEED=42
LN2PI=np.log(2*np.pi)
torch.manual_seed(SEED);np.random.seed(SEED)

print(f'═══ Exp C: Excess Return + Sizing ═══\nDevice: {DEV}')

data=load_data()
lat=torch.from_numpy(data['latents']).to(DEV)
sig_pnl=torch.from_numpy(data['sig_pnl']).to(DEV)  # signal-derived PnL
te=data['train_idx'][1];vs,ve=data['val_idx']
print(f'Train: 0:{te}  Val: {vs}:{ve}')

policy=SizingPolicy().to(DEV)
opt=torch.optim.AdamW(policy.parameters(),lr=LR,weight_decay=1e-5)

def eval_policy(silent=False):
    policy.eval()
    with torch.no_grad():
        ps=torch.zeros(1,1,device=DEV);rs=[]
        for t in range(vs,ve):
            s=policy(lat[t:t+1],ps);ps=s
            # Portfolio = sizing × signal-derived positions
            port_ret=(s*sig_pnl[t:t+1]).sum(-1).item()
            rs.append(port_ret)
    sr=compute_sharpe(np.array(rs))
    if not silent:print(f'  Val Sharpe: {sr:.4f}')
    policy.train();return sr

best_sr=-10;t0=time.time()
for step in range(N_STEPS):
    ts=np.random.randint(0,te-BATCH-1)
    zb=lat[ts:ts+BATCH];pb=sig_pnl[ts:ts+BATCH]
    ps=torch.zeros(1,1,device=DEV)
    sizes,lps_old,rs=[],[],[]
    for t in range(BATCH):
        s=policy(zb[t:t+1],ps);ps=s
        sizes.append(s)
        # Old log-prob (uniform exploration baseline)
        lp_old=-0.5*(np.log(2*np.pi*0.05**2)+1)  # constant baseline
        lps_old.append(torch.tensor(lp_old,device=DEV).unsqueeze(0))
        rs.append((s*pb[t:t+1]).sum(-1).unsqueeze(0))
    rs=torch.cat(rs);lps_old=torch.cat(lps_old)
    
    # GRPO groups
    ng=BATCH//K
    if ng<1:continue
    rg=rs[:ng*K].reshape(ng,K)
    adv=(rg-rg.mean(1,keepdim=True))/(rg.std(1,keepdim=True)+1e-8)
    adv=adv.reshape(-1)
    
    # New policy
    ps2=torch.zeros(1,1,device=DEV);lps_new=[]
    for t in range(ng*K):
        s_old=sizes[t]
        s_new=policy(zb[t:t+1],ps2);ps2=s_new
        # Simple log-prob: Gaussian around current prediction
        std_v=0.05
        lp_new=-0.5*((s_old-s_new).pow(2)/std_v**2+LN2PI+2*np.log(std_v))
        lps_new.append(lp_new.sum(-1))
    lps_new=torch.stack(lps_new)
    
    ratio=(lps_new-lps_old[:ng*K]).exp()
    surr1=ratio*adv;surr2=torch.clamp(ratio,1-CLIP,1+CLIP)*adv
    loss=-torch.min(surr1,surr2).mean()
    
    opt.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(),1.0);opt.step()
    
    if step%500==0:
        print(f'  step{step:5d} | loss={loss.item():.4f} | mean_r={rs.mean().item():.6f} | mean_sizing={torch.cat(sizes).mean().item():.3f}')
    
    if step%EVAL==0 and step>0:
        sr=eval_policy()
        if sr>best_sr:
            best_sr=sr
            torch.save(policy.state_dict(),ROOT/'data/rl_exp/exp_c_policy.pt')
            print(f'  -> Best Val Sharpe: {best_sr:.4f}')

print(f'\n{"="*50}\nTrain: {time.time()-t0:.0f}s')
print(f'Best Val Sharpe: {best_sr:.4f}')

# Test
policy.eval()
with torch.no_grad():
    ps=torch.zeros(1,1,device=DEV);trs=[]
    for t in range(data['test_idx'][0],data['test_idx'][1]):
        s=policy(lat[t:t+1],ps);ps=s
        trs.append((s*sig_pnl[t:t+1]).sum(-1).item())
tsr=compute_sharpe(np.array(trs))
# Baseline: equal-weight on signal PnL
ew=np.array(data['sig_pnl'][data['test_idx'][0]:data['test_idx'][1]]).mean(1)
print(f'Test Sharpe:        {tsr:.4f}')
print(f'EW Sig-PnL Sharpe:  {compute_sharpe(ew):.4f}')
