#!/usr/bin/env python3 -u
"""
Online paper trading — full Cash + Position accounting.

State: cash + position(sym) per symbol.
Trade fills at VWAP, mark-to-market at Close.
Equity = Cash + Σ(units[sym] × Close[sym])
Return = (Equity_t / Equity_{t-1}) - 1
"""
import warnings, pickle, json, sys, zipfile, os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from pipeline_cpcv import compute_features, infer_signal, SYMBOLS, OUTPUT_DIR

DATA_DIR = ROOT / 'data'; DAILY_DIR = DATA_DIR / 'daily'; MODEL_DIR = ROOT / 'model'
STATE_PATH = OUTPUT_DIR / 'online_state.json'

TH = float(os.getenv('TH', '0.10'))
FEE = float(os.getenv('FEE', '0.0004'))
EMA_ALPHA = float(os.getenv('EMA_ALPHA', '0.50'))
CAPITAL = float(os.getenv('CAPITAL', '10000.0'))
WEIGHT = 1.0 / len(SYMBOLS)
TH_PER_COIN = os.getenv('TH_PER_COIN', '0') in ('1', 'true', 'True')
TH_MAP = {
    'BTCUSDT': 0.25,
    'ETHUSDT': 0.25,
    'SOLUSDT': 0.25,
    'BNBUSDT': 0.25,
    'ADAUSDT': 0.30,
    'XRPUSDT': 0.30,
    'DOGEUSDT': 0.30,
    'DOTUSDT': 0.30,
    'AVAXUSDT': 0.30,
}
TRADING_RULE = os.getenv('TRADING_RULE', 'signal').lower()
PTH = float(os.getenv('PTH', '0.55'))

models = {}
for sym in SYMBOLS:
    p = MODEL_DIR / f'{sym.lower()}_final.pkl'
    if p.exists():
        with open(p, 'rb') as f: models[sym] = pickle.load(f)

def load_window(sym, end=None, days=14):
    if end is None: end = datetime.utcnow()
    dfs = []; seen = set()
    for d in range(-days, 1):
        dt = end + timedelta(days=d); ym = (dt.year, dt.month)
        if ym in seen: continue
        seen.add(ym)
        y, m = ym; fname = f'{sym}-15m-{y}-{m:02d}.zip'
        local = DATA_DIR / fname
        if local.exists():
            with zipfile.ZipFile(local) as z:
                csv = f'{sym}-15m-{y}-{m:02d}.csv'
                if csv in z.namelist(): dfs.append(pd.read_csv(z.open(csv), header=None))
        else:
            # Fallback: daily zips (open_time in μs, no header)
            month_end_dt = (datetime(y, m % 12 + 1, 1) - timedelta(days=1)) if m < 12 else \
                           datetime(y + 1, 1, 1) - timedelta(days=1)
            for dd in range(1, month_end_dt.day + 1):
                day_fname = f'{sym}-15m-{y}-{m:02d}-{dd:02d}.zip'
                day_local = DAILY_DIR / day_fname
                if day_local.exists():
                    with zipfile.ZipFile(day_local) as z:
                        day_csv = f'{sym}-15m-{y}-{m:02d}-{dd:02d}.csv'
                        if day_csv in z.namelist():
                            dfs.append(pd.read_csv(z.open(day_csv), header=None))
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    cols = ['open_time','open','high','low','close','volume','close_time','quote_vol','trades','taker_buy_vol','taker_buy_quote','ignore']
    df.columns = cols[:len(df.columns)]
    ot = pd.to_numeric(df['open_time'], errors='coerce')
    if ot.max() > 1e15:
        ot = ot / 1000  # μs → ms
    df['date'] = pd.to_datetime(ot, unit='ms', errors='coerce')
    for c in ['open','high','low','close','volume','quote_vol','taker_buy_vol','taker_buy_quote']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values('date').dropna(subset=['date']).reset_index(drop=True)
    cutoff = end - timedelta(days=days)
    return df[df['date'] >= cutoff].reset_index(drop=True)

def _dir(sym, sig_val):
    th = TH_MAP.get(sym, TH) if TH_PER_COIN else TH
    return 1 if sig_val > th else (-1 if sig_val < -th else 0)


def _dir_prob(pl, ps):
    if pl > PTH and pl > ps:
        return 1
    if ps > PTH and ps > pl:
        return -1
    return 0

# ── Default state ──
state = {
    'last_bar_time': None,
    'cash': CAPITAL,
    'positions': {sym: {'dir': 0, 'units': 0.0, 'entry_vwap': 0.0, 'entry_equity': CAPITAL} for sym in SYMBOLS},
    'last_equity': CAPITAL,
    'sig_m2': {sym: 0.0 for sym in SYMBOLS},   # signal at t-2 (for position at bar[t])
    'sig_m1': {sym: 0.0 for sym in SYMBOLS},   # signal at t-1 (for position at bar[t+1])
    'sig_ema': {sym: 0.0 for sym in SYMBOLS},  # EMA-smoothed signal (last value)
    'vwap_m1': {sym: 0.0 for sym in SYMBOLS},  # VWAP at t-1 (for return at bar[t])
    'pl_m2': {sym: 0.0 for sym in SYMBOLS},
    'pl_m1': {sym: 0.0 for sym in SYMBOLS},
    'ps_m2': {sym: 0.0 for sym in SYMBOLS},
    'ps_m1': {sym: 0.0 for sym in SYMBOLS},
    'cum_ret': 0.0,
    'n_trades': 0,
    'start_time': str(datetime.utcnow()),
}
if STATE_PATH.exists():
    with open(STATE_PATH) as f: saved = json.load(f)
    for k in state:
        if k in saved: state[k] = saved[k]
    for sym in SYMBOLS:
        for k in ['sig_m2','sig_m1','vwap_m1','pl_m2','pl_m1','ps_m2','ps_m1']:
            if k not in state or not isinstance(state.get(k), dict):
                state[k] = {}
            state[k].setdefault(sym, 0.0)

last_time = state.get('last_bar_time')
cash = state['cash']
positions = {sym: state.get('positions', {}).get(sym, {'dir':0, 'units':0.0, 'entry_vwap':0.0, 'entry_equity':CAPITAL}) for sym in SYMBOLS}
last_equity = state.get('last_equity', CAPITAL)
n_trades = state.get('n_trades', 0)
sig_m2 = state.get('sig_m2', {})
sig_m1 = state.get('sig_m1', {})
sig_ema_state = state.get('sig_ema', {})
vwap_m1 = state.get('vwap_m1', {})
pl_m2 = state.get('pl_m2', {})
pl_m1 = state.get('pl_m1', {})
ps_m2 = state.get('ps_m2', {})
ps_m1 = state.get('ps_m1', {})

print(f'State: cash=${cash:.2f}, equity=${last_equity:.2f}, last_bar={last_time}')

# ══════════════════════════════════════════════════════════════
#  Inference — get latest signal per symbol
# ══════════════════════════════════════════════════════════════
print('\n=== Inference ===')
new_sigs = {}; new_pl = {}; new_ps = {}; new_closes = {}; new_vwaps = {}; new_times = {}

for sym in SYMBOLS:
    if sym not in models: continue
    df = load_window(sym)
    if df is None or len(df) < 200: continue
    df, fn = compute_features(df)
    sig, pl, ps, dates, vw, close = infer_signal(df, models[sym]['feat'], models[sym])
    if sig is None: continue
    new_sigs[sym] = sig; new_pl[sym] = pl; new_ps[sym] = ps
    new_closes[sym] = close; new_vwaps[sym] = vw
    new_times[sym] = pd.to_datetime(dates)
    if last_time is None:
        print(f'  {sym}: {len(sig)} bars, signal={sig[-1]:+.4f}')
    else:
        new_mask = new_times[sym] > pd.Timestamp(last_time)
        print(f'  {sym}: {new_mask.sum()} new bars, signal={sig[-1]:+.4f}')

if not new_sigs:
    print('No data. Saving state.')
    with open(STATE_PATH, 'w') as f: json.dump(state, f, indent=2, default=str)
    sys.exit(0)

# ══════════════════════════════════════════════════════════════
#  Process new bars — trade at VWAP, mark at Close
# ══════════════════════════════════════════════════════════════
print('\n=== Trading ===')

# Align new bars across symbols
all_series = {}
for sym in SYMBOLS:
    if sym not in new_sigs: continue
    all_series[f'{sym}_sig'] = pd.Series(new_sigs[sym], index=new_times[sym])
    all_series[f'{sym}_pl'] = pd.Series(new_pl[sym], index=new_times[sym])
    all_series[f'{sym}_ps'] = pd.Series(new_ps[sym], index=new_times[sym])
    all_series[f'{sym}_close'] = pd.Series(new_closes[sym], index=new_times[sym])
    all_series[f'{sym}_vwap'] = pd.Series(new_vwaps[sym], index=new_times[sym])

df_all = pd.DataFrame(all_series).sort_index()
if last_time is not None:
    df_all = df_all[df_all.index > pd.Timestamp(last_time)]

if len(df_all) < 1:
    print('No new bars.')
else:
    active_syms = [sym for sym in SYMBOLS if sym in new_sigs]
    df_all = df_all.ffill()
    need_cols = []
    for sym in active_syms:
        need_cols.extend([f'{sym}_vwap', f'{sym}_close'])
    df_all = df_all.dropna(subset=need_cols, how='any')

    T = len(df_all)
    if T < 1:
        print('No new bars.')
        with open(STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        sys.exit(0)

    sig_arr = {sym: df_all[f'{sym}_sig'].to_numpy(np.float64) for sym in active_syms}
    pl_arr = {sym: df_all[f'{sym}_pl'].to_numpy(np.float64) for sym in active_syms}
    ps_arr = {sym: df_all[f'{sym}_ps'].to_numpy(np.float64) for sym in active_syms}
    vwap_arr = {sym: df_all[f'{sym}_vwap'].to_numpy(np.float64) for sym in active_syms}
    close_arr = {sym: df_all[f'{sym}_close'].to_numpy(np.float64) for sym in active_syms}

    # -- Apply EMA smoothing to signals before trading --
    if EMA_ALPHA < 1.0:
        for sym in active_syms:
            sig = sig_arr[sym]
            prev = float(sig_ema_state.get(sym, sig[0]))
            ema = np.zeros(len(sig), np.float64)
            ema[0] = EMA_ALPHA * sig[0] + (1 - EMA_ALPHA) * prev
            for t in range(1, len(sig)):
                ema[t] = EMA_ALPHA * sig[t] + (1 - EMA_ALPHA) * ema[t-1]
            sig_arr[sym] = ema

    for bar_idx in range(T):
        bar_time = df_all.index[bar_idx]
        equity_before = last_equity

        for sym in active_syms:
            sig_series = sig_arr[sym]
            pl_series = pl_arr[sym]
            ps_series = ps_arr[sym]
            vwap_series = vwap_arr[sym]

            # Position from signal[t-2] (VWAP offset=2)
            if TRADING_RULE == 'prob':
                if bar_idx == 0:
                    pl_for_pos = float(pl_m2.get(sym, 0.0))
                    ps_for_pos = float(ps_m2.get(sym, 0.0))
                elif bar_idx == 1:
                    pl_for_pos = float(pl_m1.get(sym, 0.0))
                    ps_for_pos = float(ps_m1.get(sym, 0.0))
                else:
                    pl_for_pos = float(pl_series[bar_idx - 2]) if not np.isnan(pl_series[bar_idx - 2]) else 0.0
                    ps_for_pos = float(ps_series[bar_idx - 2]) if not np.isnan(ps_series[bar_idx - 2]) else 0.0
                new_dir = _dir_prob(pl_for_pos, ps_for_pos)
            else:
                if bar_idx == 0:
                    sig_for_pos = sig_m2.get(sym, 0.0)
                elif bar_idx == 1:
                    sig_for_pos = sig_m1.get(sym, 0.0)
                else:
                    s = sig_series[bar_idx - 2]
                    sig_for_pos = 0.0 if np.isnan(s) else float(s)
                new_dir = _dir(sym, sig_for_pos)
            p = positions.get(sym, {'dir': 0, 'units': 0.0, 'entry_vwap': 0.0, 'entry_equity': CAPITAL})
            old_dir = p['dir']; old_units = p['units']
            vwap_t = float(vwap_series[bar_idx])

            if new_dir != old_dir and vwap_t > 0:
                if old_dir != 0 and old_units != 0:
                    cash += old_units * vwap_t
                    cash -= abs(old_units) * vwap_t * FEE
                if new_dir != 0:
                    notional = WEIGHT * equity_before
                    new_units = new_dir * notional / vwap_t
                    cash -= new_dir * notional  # - for LONG (buy), + for SHORT (sell)
                    cash -= notional * FEE
                    positions[sym] = {'dir': new_dir, 'units': new_units,
                                      'entry_vwap': vwap_t, 'entry_equity': equity_before}
                    n_trades += 1
                else:
                    positions[sym] = {'dir': 0, 'units': 0.0, 'entry_vwap': 0.0, 'entry_equity': equity_before}

        # Mark to market at Close
        asset_val = 0.0
        for sym in active_syms:
            p = positions.get(sym, {'dir': 0, 'units': 0.0})
            if p['dir'] != 0:
                asset_val += p['units'] * float(close_arr[sym][bar_idx])

        equity = cash + asset_val
        if last_equity > 0:
            bar_ret = (equity - last_equity) / last_equity
        else:
            bar_ret = 0.0

        last_equity = equity
        state['last_bar_time'] = str(bar_time)
        state['cash'] = cash
        state['last_equity'] = equity

    # Update state signals for next run
    for sym in active_syms:
        s = sig_arr[sym]
        if len(s) >= 2:
            v2 = s[-2]; v1 = s[-1]
            state['sig_m2'][sym] = 0.0 if np.isnan(v2) else float(v2)
            state['sig_m1'][sym] = 0.0 if np.isnan(v1) else float(v1)
        elif len(s) == 1:
            v1 = s[-1]
            state['sig_m2'][sym] = float(sig_m1.get(sym, 0.0))
            state['sig_m1'][sym] = 0.0 if np.isnan(v1) else float(v1)
        plv = pl_arr[sym]
        psv = ps_arr[sym]
        if len(plv) >= 2:
            state['pl_m2'][sym] = 0.0 if np.isnan(plv[-2]) else float(plv[-2])
            state['pl_m1'][sym] = 0.0 if np.isnan(plv[-1]) else float(plv[-1])
        elif len(plv) == 1:
            state['pl_m2'][sym] = float(pl_m1.get(sym, 0.0))
            state['pl_m1'][sym] = 0.0 if np.isnan(plv[-1]) else float(plv[-1])
        if len(psv) >= 2:
            state['ps_m2'][sym] = 0.0 if np.isnan(psv[-2]) else float(psv[-2])
            state['ps_m1'][sym] = 0.0 if np.isnan(psv[-1]) else float(psv[-1])
        elif len(psv) == 1:
            state['ps_m2'][sym] = float(ps_m1.get(sym, 0.0))
            state['ps_m1'][sym] = 0.0 if np.isnan(psv[-1]) else float(psv[-1])
        v = vwap_arr[sym]
        if len(v) > 0:
            state['vwap_m1'][sym] = float(v[-1])
        # Save last EMA value for next run
        if EMA_ALPHA < 1.0:
            s = sig_arr[sym]
            state['sig_ema'][sym] = float(s[-1]) if len(s) > 0 else sig_ema_state.get(sym, 0.0)

    state['positions'] = positions
    state['n_trades'] = n_trades
    state['cum_ret'] = equity / CAPITAL - 1

    active = sum(1 for sym in SYMBOLS if positions.get(sym, {}).get('dir', 0) != 0)
    print(f'── Results ──')
    print(f'  PnL this run:  ${equity - CAPITAL:,.2f}')
    print(f'  Equity:        ${equity:,.2f}')
    print(f'  Cash:          ${cash:,.2f}')
    print(f'  Asset:         ${asset_val:,.2f}')
    print(f'  Active:        {active} symbols')
    print(f'  Trades:        {n_trades}')

    for sym in SYMBOLS:
        p = positions.get(sym, {})
        if p.get('dir', 0) != 0:
            dir_str = 'LONG' if p['dir'] == 1 else 'SHORT'
            print(f'    {sym}: {dir_str} {abs(p.get("units",0)):.4f} units @ ${p.get("entry_vwap",0):.2f}')

# ── Save ──
with open(STATE_PATH, 'w') as f:
    json.dump(state, f, indent=2, default=str)
print(f'\n✅ State saved.')

sig_out = {'timestamp': str(datetime.utcnow()), 'signals': {}}
for sym in SYMBOLS:
    if sym not in new_sigs: continue
    s = float(new_sigs[sym][-1])
    if TRADING_RULE == 'prob' and sym in new_pl and sym in new_ps:
        plv = float(new_pl[sym][-1])
        psv = float(new_ps[sym][-1])
        d = _dir_prob(plv, psv)
        sig_out['signals'][sym] = {
            'signal': s,
            'p_long': plv,
            'p_short': psv,
            'position': 'LONG' if d == 1 else ('SHORT' if d == -1 else 'FLAT'),
        }
    else:
        sig_out['signals'][sym] = {'signal': s, 'position': 'LONG' if s > TH else ('SHORT' if s < -TH else 'FLAT')}
json.dump(sig_out, open(OUTPUT_DIR/'signals_latest.json','w'), indent=2)
print(f'✅ Signals saved.')
