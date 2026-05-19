# RL Portfolio Trading — Experiment Summary

**🏆 Final Champion: GRPO Gate** (`exp_r_grpo_gate.py`)
- Net SR=+1.19, TO=0.06, Σ|w|=0.11
- Synthetic CFM training → real market inference
- Direction(tanh) × Gate(sigmoid, init bias=-5) + entry fee

## Structure

```
exp_r_grpo_gate.py    — 🏆 GRPO Gate (final champion)
EXPERIMENT_LOG.md     — full experiment log (v1~v5)
env_rl.py             — Gym env utility
archive/              — all archived experiment scripts
```

## Quick Start

```bash
python exp_r_grpo_gate.py         # train GRPO Gate
```

## Experiment Archive

All intermediate experiments moved to `archive/`:
- DeltaSoft series (exp_s, exp_t)
- DeltaGate v2 (exp_u)
- GRPO chunk variants (exp_p, exp_q, exp_r_grpo_chunk)
- Earlier experiments (exp_a through exp_k)
