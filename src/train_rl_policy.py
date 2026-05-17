#!/usr/bin/env python3 -u
"""GRPO — synthetic dream world training."""
import sys,warnings,time
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
import torch,torch.nn as nn,torch.nn.functional as F
from pipeline_cpcv import SYMBOLS,OUTPUT_DIR

SEED=42;torch.manual_seed(SEED);np.random.seed(SEED)
DEV='mps'if torch.backends.mps.is_available()else'cpu'
DZ,NC,H=16,9,128;LR=3e-4;B=4000;L=5;K=32;N_STEPS=5000

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(DZ,H),nn.SiLU(),nn.Linear(H,H),nn.SiLU())
        self.mu_head=nn.Linear(H,NC);self.log_std=nn.Parameter(torch.zeros(NC))
    def forward(self,s,det=False):
        h=self.shared(s);mu=torch.tanh(self.mu_head(h))
        if det:return mu
        return torch.tanh(mu+torch.randn_like(mu)*self.log_std.exp())

# ── Load & slice into parallel segments ──
d=np.load('data/synthetic_cfm.npz')
lat=d['latent'].astype(np.float32);ret=d['returns'].astype(np.float32)
import pandas as pd
lr=np.load('data/market_latent.npy').astype(np.float32)
ld=np.load('data/market_latent_dates.npy',allow_pickle=True)
dm={d:i for i,d in enumerate(ld)}
vm=np.zeros((len(lr),NC),np.float32)
for j,s in enumerate(SYMBOLS):
    v=np.load(OUTPUT_DIR/f'cpcv_vwap_{s}.npy')
    ds=pd.to_datetime(np.load(OUTPUT_DIR/f'cpcv_dates_{s}.npy'))
    for k,dt in enumerate(ds):
        i=dm.get(dt)
        if i is not None:vm[i,j]=v[k]
Nrr=len(lr)-2
rr=np.zeros((Nrr,NC),np.float32)
for j in range(NC):
    r=vm[2:,j]/vm[1:-1,j]-1.;rr[:,j]=np.where(np.isfinite(r),r,0.)
sc=rr.std()/(ret.std()+1e-8);ret*=sc

# Slice into segments
N_seg=len(lat)//L;lat=lat[:N_seg*L].reshape(N_seg,L,DZ);ret=ret[:N_seg*L].reshape(N_seg,L,NC)
lm=lat.mean(axis=(0,1),keepdims=True);ls=lat.std(axis=(0,1),keepdims=True).clip(1e-6)
lat_n=((lat-lm)/ls).astype(np.float32)
rla=lr[:Nrr];rlm=rla.mean(0);rls=rla.std(0).clip(1e-6)
Sr=torch.from_numpy(((rla-rlm)/rls).astype(np.float32)).to(DEV)
Rr=torch.from_numpy(rr).to(DEV)
S_tr=torch.from_numpy(lat_n).to(DEV);R_tr=torch.from_numpy(ret).to(DEV)

print(f'Segments: {N_seg} × L={L} = {N_seg*L} total steps')

@torch.no_grad()
def ev(p):
    w=p.forward(Sr,det=True);pr=(w*Rr).sum(1)
    sr=pr.mean()/(pr.std()+1e-8)*np.sqrt(252*96)
    to=np.abs(np.diff(w.cpu().numpy(),axis=0)).sum(1).mean()
    return sr.item(),pr.mean().item(),w.mean(0).cpu().numpy(),w.abs().sum(1).cpu().mean().item(),to

pi=Policy().to(DEV);opt=torch.optim.AdamW(pi.parameters(),lr=LR)
print(f'═══ GRPO (no fee in training, B={B}, L={L}, K={K}) ═══')

t0=time.time()
for step in range(N_STEPS):
    perm=torch.randperm(N_seg,device=DEV)[:B]
    s_seg=S_tr[perm];r_seg=R_tr[perm]  # (B, L, DZ), (B, L, NC)

    # Process all L steps in parallel segments (L-step loop, batched across segments)
    total_loss=0
    for l in range(L):
        s_t=s_seg[:,l]  # (B, DZ)
        r_t=r_seg[:,l]  # (B, NC)

        # K samples per state (parallel across segments)
        sk=s_t.unsqueeze(1).expand(B,K,DZ).reshape(B*K,DZ)
        h=pi.shared(sk);mu=torch.tanh(pi.mu_head(h))
        std=pi.log_std.exp().expand_as(mu)
        eps=torch.randn_like(mu);wk=torch.tanh(mu+eps*std)

        # Log prob
        lp=-.5*(eps**2+2*pi.log_std+np.log(2*np.pi))
        lp=lp-(2*(np.log(2)-wk-F.softplus(-2*wk)))
        lp=lp.sum(-1).view(B,K);wk=wk.view(B,K,NC)

        # Reward = portfolio_return (no fee — see backtest for cost handling)
        rk=r_t.unsqueeze(1).expand(-1,K,-1)
        rew=(wk*rk).sum(-1)

        # GRPO: group norm within each (segment, step) group
        ad=(rew-rew.mean(1,keepdim=True))/(rew.std(1,keepdim=True)+1e-8)

        # Loss: PG + entropy + KL
        pl=-(lp*ad.detach()).mean();el=-lp.mean()
        with torch.no_grad():mr=torch.tanh(pi.mu_head(pi.shared(s_t)))
        kl=.5*(mr**2+pi.log_std.exp()**2-1-2*pi.log_std).mean()
        total_loss+=pl+.0005*el+.01*kl

    opt.zero_grad();total_loss.backward()
    nn.utils.clip_grad_norm_(pi.parameters(),1.0);opt.step()

    if(step+1)%1000==0 or step==0:
        sr,pn,wm,sw,to=ev(pi);s=pi.log_std.exp().mean().item()
        print(f'  Step {step+1:>5d}  SR={sr:.2f}  PnL={pn:.6f}  σ={s:.3f}  Σ|w|={sw:.2f}  TO={to:.4f}  '
              f'w=[{wm[0]:+.3f}/{wm[1]:+.3f}/{wm[2]:+.3f}..{wm[3]:+.3f}/{wm[4]:+.3f}/{wm[5]:+.3f}..{wm[6]:+.3f}/{wm[7]:+.3f}/{wm[8]:+.3f}]')

sr,pn,wm,sw,to=ev(pi)
torch.save({'model_state':pi.state_dict(),'n_coins':NC,'latent_dim':DZ,
            'latent_mean':rlm,'latent_std':rls},'data/rl_policy.pt')
print(f'\nDone: {time.time()-t0:.0f}s')
print(f'═══ Final: SR={sr:.2f}  PnL={pn:.6f}  Σ|w|={sw:.2f}  TO={to:.4f}')
