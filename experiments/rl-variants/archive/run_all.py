#!/usr/bin/env python3 -u
"""
RL Experiment Suite — GRPO variants on 9-coin crypto portfolio.
Uses efficient segmented batch training (B parallel segments × L steps).
"""
import sys,warnings,time
from pathlib import Path
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parent.parent.parent
sys.path.insert(0,str(ROOT/'src'))

DZ,NC,H=16,9,128;SEED=42
torch.manual_seed(SEED);np.random.seed(SEED)
DEV='mps'if torch.backends.mps.is_available()else'cpu'
ANNUAL=np.sqrt(24*365)

# ── Shared modules ──
class Policy(nn.Module):
    """Latent → tanh weights. Evaluated point-wise (no prev_w needed for efficient batch)."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(DZ,H),nn.SiLU(),nn.Linear(H,H),nn.SiLU())
        self.mu=nn.Linear(H,NC);self.log_s=nn.Parameter(torch.zeros(NC))
    def forward(self,s,det=False):
        h=self.net(s);m=torch.tanh(self.mu(h))
        if det:return m
        return torch.tanh(m+torch.randn_like(m)*self.log_s.exp())

class SizPolicy(nn.Module):
    """Latent → 1 scalar ∈ [0,1]."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(DZ,H),nn.SiLU(),nn.Linear(H,H),nn.SiLU(),
            nn.Linear(H,1),nn.Sigmoid())
    def forward(self,s):return self.net(s)

# ── Load data ──
print('═══ RL Experiment Suite (Efficient) ═══')
print(f'Device: {DEV}\n')
d=np.load(str(ROOT/'data/rl_exp/exp_data.npz'))
lat=d['latents'].astype(np.float32)
ret=d['raw_ret'].astype(np.float32)
spnl=d['sig_pnl'].astype(np.float32)
tr,va,te=d['train_idx'],d['val_idx'],d['test_idx']
N=len(lat)

# Normalize latents
lm=lat[:tr[1]].mean(0,keepdims=True);ls=lat[:tr[1]].std(0,keepdims=True).clip(1e-6)
lat_n=((lat-lm)/ls).astype(np.float32)

L=5   # segment length (short — enough for GRPO group comparison)
print(f'Data: {N:,} bars, segment L={L}')
for name,sl in [('Train',tr),('Val',va),('Test',te)]:
    ew=ret[sl[0]:sl[1]].mean(1);sr=ew.mean()/max(ew.std(),1e-8)*ANNUAL
    print(f'  {name}: {sl[1]-sl[0]:,} bars, EW Sharpe={sr:.4f}')

lt=torch.from_numpy(lat_n).to(DEV)
rt=torch.from_numpy(ret).to(DEV)
st=torch.from_numpy(spnl).to(DEV)

def segment_data(data,start,end):
    """Slice data into (N_seg, L, dim) segments."""
    n=end-start;n_use=n//L*L
    return data[start:start+n_use].reshape(-1,L,data.shape[-1])

S_tr=segment_data(lt,tr[0],tr[1])  # (N_seg, L, 16)
R_tr=segment_data(rt,tr[0],tr[1])  # (N_seg, L, 9)
P_tr=segment_data(st,tr[0],tr[1])  # (N_seg, L, 9) for sig-pnl experiments

def evaluate(policy,ret_src,sz=False):
    """Point-wise evaluate on validation set (no sequential loop)."""
    policy.eval()
    with torch.no_grad():
        w=policy(lt[va[0]:va[1]],det=True)
        if sz:w=w.mean(-1,keepdim=True).expand(-1,NC)
        pr=(w*ret_src[va[0]:va[1]]).sum(-1).cpu().numpy()
    sr=pr.mean()/max(pr.std(),1e-8)*ANNUAL
    policy.train();return sr

def test_eval(policy,ret_src,sz=False):
    """Test evaluation."""
    policy.eval()
    with torch.no_grad():
        w=policy(lt[te[0]:te[1]],det=True)
        if sz:w=w.mean(-1,keepdim=True).expand(-1,NC)
        pr=(w*ret_src[te[0]:te[1]]).sum(-1).cpu().numpy()
    sr=pr.mean()/max(pr.std(),1e-8)*ANNUAL
    policy.train();return sr

# ══════════════════════════════════════════════════════════
# Exp 1: GRPO Direct on Raw Returns (stock-style)
# ══════════════════════════════════════════════════════════
def exp1():
    print(f'\n{"="*60}\n[Exp 1] GRPO Direct on Raw Returns\n{"="*60}')
    B,K,LR,STEPS,EVAL=2000,32,3e-4,10000,500
    
    pi=Policy().to(DEV);opt=torch.optim.AdamW(pi.parameters(),lr=LR)
    best_sr=-10;t0=time.time()
    
    for step in range(STEPS):
        perm=torch.randperm(len(S_tr),device=DEV)[:B]
        ss=S_tr[perm];rs=R_tr[perm]
        
        for l in range(L):
            s_t=ss[:,l];r_t=rs[:,l]
            sk=s_t.unsqueeze(1).expand(B,K,DZ).reshape(B*K,DZ)
            h=pi.net(sk);mu=torch.tanh(pi.mu(h));std=pi.log_s.exp().expand_as(mu)
            eps=torch.randn_like(mu);wk=torch.tanh(mu+eps*std)
            
            lp=-.5*(eps**2+2*pi.log_s+np.log(2*np.pi))
            lp=lp-(2*(np.log(2)-wk-F.softplus(-2*wk)))
            lp=lp.sum(-1).view(B,K);wk=wk.view(B,K,NC)
            
            rk=r_t.unsqueeze(1).expand(-1,K,-1)
            rew=(wk*rk).sum(-1)
            
            ad=(rew-rew.mean(1,keepdim=True))/(rew.std(1,keepdim=True)+1e-8)
            pl=-(lp*ad.detach()).mean();el=-lp.mean()
            with torch.no_grad():mr=torch.tanh(pi.mu(pi.net(s_t)))
            kl=.5*(mr**2+pi.log_s.exp()**2-1-2*pi.log_s).mean()
            loss=pl+.0005*el+.01*kl
            
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(pi.parameters(),1.0);opt.step()
        
        if step%EVAL==0:
            sr=evaluate(pi,rt)
            if sr>best_sr:best_sr=sr
            print(f'  Step {step:5d}  SR={sr:.2f}  σ={pi.log_s.exp().mean().item():.3f}')
    
    tsr=test_eval(pi,rt)
    print(f'[Exp 1] Time={time.time()-t0:.0f}s  Best Val={best_sr:.2f}  Test={tsr:.2f}')
    return best_sr,tsr

# ══════════════════════════════════════════════════════════
# Exp 2: Differentiable Sharpe (no GRPO, direct)
# ══════════════════════════════════════════════════════════
def exp2():
    print(f'\n{"="*60}\n[Exp 2] Differentiable Sharpe (direct optimization)\n{"="*60}')
    B,LR,STEPS,EVAL=8192,3e-4,20000,1000
    
    pi=Policy().to(DEV);opt=torch.optim.AdamW(pi.parameters(),lr=LR)
    best_sr=-10;t0=time.time()
    
    for step in range(STEPS):
        ts=np.random.randint(tr[0],tr[1]-B-1)
        z=lt[ts:ts+B];r=rt[ts:ts+B]
        w=pi(z)
        pr=(w*r).sum(-1)
        mu=pr.mean();sd=pr.std()+1e-8
        loss=-mu/sd
        
        opt.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(),0.5);opt.step()
        
        if step%EVAL==0:
            sr=evaluate(pi,rt)
            if sr>best_sr:best_sr=sr
            print(f'  Step {step:5d}  SR={sr:.2f}')
    
    tsr=test_eval(pi,rt)
    print(f'[Exp 2] Time={time.time()-t0:.0f}s  Best Val={best_sr:.2f}  Test={tsr:.2f}')
    return best_sr,tsr

# ══════════════════════════════════════════════════════════
# Exp 3: GRPO on Signal-Derived PnL
# ══════════════════════════════════════════════════════════
def exp3():
    print(f'\n{"="*60}\n[Exp 3] GRPO on Signal-Derived PnL\n{"="*60}')
    B,K,LR,STEPS,EVAL=2000,32,3e-4,10000,500
    
    pi=Policy().to(DEV);opt=torch.optim.AdamW(pi.parameters(),lr=LR)
    best_sr=-10;t0=time.time()
    
    for step in range(STEPS):
        perm=torch.randperm(len(S_tr),device=DEV)[:B]
        ss=S_tr[perm];ps=P_tr[perm]
        
        for l in range(L):
            s_t=ss[:,l];r_t=ps[:,l]
            sk=s_t.unsqueeze(1).expand(B,K,DZ).reshape(B*K,DZ)
            h=pi.net(sk);mu=torch.tanh(pi.mu(h));std=pi.log_s.exp().expand_as(mu)
            eps=torch.randn_like(mu);wk=torch.tanh(mu+eps*std)
            
            lp=-.5*(eps**2+2*pi.log_s+np.log(2*np.pi))
            lp=lp-(2*(np.log(2)-wk-F.softplus(-2*wk)))
            lp=lp.sum(-1).view(B,K);wk=wk.view(B,K,NC)
            
            rk=r_t.unsqueeze(1).expand(-1,K,-1)
            rew=(wk*rk).sum(-1)
            
            ad=(rew-rew.mean(1,keepdim=True))/(rew.std(1,keepdim=True)+1e-8)
            pl=-(lp*ad.detach()).mean();el=-lp.mean()
            with torch.no_grad():mr=torch.tanh(pi.mu(pi.net(s_t)))
            kl=.5*(mr**2+pi.log_s.exp()**2-1-2*pi.log_s).mean()
            loss=pl+.0005*el+.01*kl
            
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(pi.parameters(),1.0);opt.step()
        
        if step%EVAL==0:
            sr=evaluate(pi,st)
            if sr>best_sr:best_sr=sr
            print(f'  Step {step:5d}  SR={sr:.2f}')
    
    tsr=test_eval(pi,st)
    ew=st[te[0]:te[1]].mean(1).cpu().numpy()
    ews=ew.mean()/max(ew.std(),1e-8)*ANNUAL
    print(f'[Exp 3] Time={time.time()-t0:.0f}s  Best Val={best_sr:.2f}  Test={tsr:.2f}  EW={ews:.2f}')
    return best_sr,tsr,ews

# ══════════════════════════════════════════════════════════
# Exp 4: Diff Sharpe on Signal-Derived PnL
# ══════════════════════════════════════════════════════════
def exp4():
    print(f'\n{"="*60}\n[Exp 4] Diff Sharpe on Signal-Derived PnL\n{"="*60}')
    B,LR,STEPS,EVAL=8192,3e-4,20000,1000
    
    pi=Policy().to(DEV);opt=torch.optim.AdamW(pi.parameters(),lr=LR)
    best_sr=-10;t0=time.time()
    
    for step in range(STEPS):
        ts=np.random.randint(tr[0],tr[1]-B-1)
        z=lt[ts:ts+B];r=st[ts:ts+B]
        w=pi(z)
        pr=(w*r).sum(-1)
        mu=pr.mean();sd=pr.std()+1e-8
        loss=-mu/sd
        
        opt.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(),0.5);opt.step()
        
        if step%EVAL==0:
            sr=evaluate(pi,st)
            if sr>best_sr:best_sr=sr
            print(f'  Step {step:5d}  SR={sr:.2f}')
    
    tsr=test_eval(pi,st)
    print(f'[Exp 4] Time={time.time()-t0:.0f}s  Best Val={best_sr:.2f}  Test={tsr:.2f}')
    return best_sr,tsr

# ══════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════
res={}
t0=time.time()
# Fast ones first (random batches, no segment loop)
res['exp2_diff_raw']=exp2();print(f'⏱ {time.time()-t0:.0f}s')
res['exp4_diff_pnl']=exp4();print(f'⏱ {time.time()-t0:.0f}s')
# GRPO (segmented, slightly slower per step but fewer steps needed)
res['exp1_grpo_raw']=exp1();print(f'⏱ {time.time()-t0:.0f}s')
res['exp3_grpo_pnl']=exp3();print(f'⏱ {time.time()-t0:.0f}s')

print(f'\n{"="*60}')
print('RESULTS')
print(f'{"="*60}')
print(f'{"Experiment":<25} {"Val SR":>8} {"Test SR":>8}')
print(f'{"-"*45}')
for n,(v,t,*_) in res.items():
    print(f'{n:<25} {v:>8.2f} {t:>8.2f}')
ew_test=ret[te[0]:te[1]].mean(1)
print(f'{"EW Baseline":<25} {"—":>8} {ew_test.mean()/max(ew_test.std(),1e-8)*ANNUAL:>8.2f}')
print(f'\nTotal: {time.time()-t0:.0f}s')
