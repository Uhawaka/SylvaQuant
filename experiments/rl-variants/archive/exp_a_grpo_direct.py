#!/usr/bin/env python3 -u
"""Exp A: GRPO Direct on Real Data (stock-style)."""
import sys,warnings,time
from pathlib import Path
import numpy as np
import torch,torch.nn as nn
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/rl-variants'))
from shared import Policy,compute_sharpe,load_data,DEV,DZ,NC,H

K=32;LR=3e-4;N_STEPS=30000;BATCH=2048;CLIP=0.2;ENT=0.01;EVAL=2000;SEED=42
torch.manual_seed(SEED);np.random.seed(SEED)
LN2PI=np.log(2*np.pi)

print(f'═══ Exp A: GRPO Direct ═══\nDevice: {DEV}')

data=load_data()
lat=torch.from_numpy(data['latents']).to(DEV)
ret=torch.from_numpy(data['raw_ret']).to(DEV)
te=data['train_idx'][1];vs,ve=data['val_idx']
print(f'Train: 0:{te}  Val: {vs}:{ve}')

policy=Policy(out_dim=NC).to(DEV)
opt=torch.optim.AdamW(policy.parameters(),lr=LR,weight_decay=1e-5)
old_policy=Policy(out_dim=NC).to(DEV)
old_policy.load_state_dict(policy.state_dict())

def act(z,pw,old=True):
    p=old_policy if old else policy
    mu=torch.tanh(p(z,pw))
    std=0.1+0.4*torch.sigmoid(p.net[-1].weight.mean().detach())
    eps=torch.randn_like(mu)*std
    a=torch.tanh(mu+eps)
    lp=-0.5*((a-mu).pow(2)/std**2+LN2PI+2*std.log())
    return a,lp.sum(-1),0.5*(1+LN2PI+2*std.log())*NC

def eval_policy(silent=False):
    policy.eval()
    with torch.no_grad():
        pw=torch.zeros(1,NC,device=DEV);rs=[]
        for t in range(vs,ve):
            a,_,_=act(lat[t:t+1],pw,old=False);pw=a
            rs.append((a*ret[t:t+1]).sum(-1).item())
    sr=compute_sharpe(np.array(rs))
    if not silent:print(f'  Val Sharpe: {sr:.4f}')
    policy.train();return sr

best_sr=-10;t0=time.time()
for step in range(N_STEPS):
    ts=np.random.randint(0,te-BATCH-1)
    zb=lat[ts:ts+BATCH];rb=ret[ts:ts+BATCH]
    pw=torch.zeros(1,NC,device=DEV)
    acts,lps_old,rs,ents=[],[],[],[]
    for t in range(BATCH):
        a,lp,ent=act(zb[t:t+1],pw,old=True)
        acts.append(a);lps_old.append(lp);ents.append(ent)
        rs.append((a*rb[t:t+1]).sum(-1).unsqueeze(0));pw=a
    
    acts=torch.cat(acts);lps_old=torch.stack(lps_old);rs=torch.cat(rs)
    
    # GRPO groups
    ng=BATCH//K
    if ng<1:continue
    rg=rs[:ng*K].reshape(ng,K)
    adv=(rg-rg.mean(1,keepdim=True))/(rg.std(1,keepdim=True)+1e-8)
    adv=adv.reshape(-1)
    
    # New policy
    pw2=torch.zeros(1,NC,device=DEV);lps_new=[]
    for t in range(ng*K):
        a_old=acts[t:t+1]
        logits=policy(zb[t:t+1],pw2)
        mu=torch.tanh(logits)
        std_v=0.1+0.4*torch.sigmoid(policy.net[-1].weight.mean().detach())
        eps=torch.randn_like(a_old)*std_v*0.1
        mu2=torch.tanh(mu+eps);pw2=mu2
        lp_new=-0.5*((a_old-mu2).pow(2)/std_v**2+LN2PI+2*std_v.log())
        lps_new.append(lp_new.sum(-1))
    
    lps_new=torch.stack(lps_new)
    ratio=(lps_new-lps_old[:ng*K]).exp()
    surr1=ratio*adv;surr2=torch.clamp(ratio,1-CLIP,1+CLIP)*adv
    loss=-torch.min(surr1,surr2).mean()-ENT*torch.stack(ents)[:ng*K].mean()
    
    opt.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(),1.0);opt.step()
    
    if step%500==0:
        print(f'  step{step:5d} | loss={loss.item():.4f} | mean_r={rs.mean().item():.6f} | adv_std={adv.std().item():.4f}')
    
    if step%EVAL==0 and step>0:
        sr=eval_policy()
        if sr>best_sr:
            best_sr=sr
            torch.save(policy.state_dict(),ROOT/'data/rl_exp/exp_a_policy.pt')
            print(f'  -> Best Val Sharpe: {best_sr:.4f}')
    
    if step%100==0:
        old_policy.load_state_dict(policy.state_dict())

print(f'\n{"="*50}\nTrain: {time.time()-t0:.0f}s')
print(f'Best Val Sharpe: {best_sr:.4f}')

# Test
policy.eval()
with torch.no_grad():
    pw=torch.zeros(1,NC,device=DEV);trs=[]
    for t in range(data['test_idx'][0],data['test_idx'][1]):
        a,_,_=act(lat[t:t+1],pw,old=False);pw=a
        trs.append((a*ret[t:t+1]).sum(-1).item())
tsr=compute_sharpe(np.array(trs))
ew=np.array(data['raw_ret'][data['test_idx'][0]:data['test_idx'][1]]).mean(1)
print(f'Test Sharpe:       {tsr:.4f}')
print(f'EW Test Sharpe:    {compute_sharpe(ew):.4f}')
