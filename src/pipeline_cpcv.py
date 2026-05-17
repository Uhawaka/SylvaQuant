#!/usr/bin/env python3 -u
"""
Pipeline: data loading -> features -> TB labels -> CPCV evaluation.
"""
import warnings, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output'
MODEL_DIR = ROOT / 'model'

# ── Shared constants ──
SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','ADAUSDT','XRPUSDT','DOGEUSDT','DOTUSDT','AVAXUSDT']
FEATS = [
    'ret_1','ret_2','ret_4','ret_8','ret_16','ret_24','ret_32','ret_48','ret_64','ret_96','ret_128',
    'abs_ret_1','abs_ret_2','abs_ret_4','abs_ret_8','abs_ret_16','abs_ret_24','abs_ret_32','abs_ret_48','abs_ret_64','abs_ret_96','abs_ret_128',
    'ret_vol_corr_16','ret_vol_corr_32','ret_vol_corr_64','ret_vol_corr_96',
    # === New features v2 ===
    'hl_range_16','hl_range_48','hl_range_96',
    'close_pos_16','close_pos_48','close_pos_96',
    'buy_frac_16','buy_frac_48','buy_frac_96',
    'vol_skew_48','vol_skew_96',
    'vol_ratio_16_96',
    # === New features v3 (trend/vol/volume dynamics) ===
    'consec_up_8','consec_up_24',
    'consec_vol_16','consec_vol_48',
    'ret_ma_dist_48','ret_ma_dist_96',
    'ret_range_pos_48','ret_range_pos_96',
    'vol_delta_16','vol_delta_48',
    'vol_max_16','vol_max_48',
    'vol_surge_16','vol_surge_48',
    'qv_surge_16','qv_surge_48',
    'up_wick_16','up_wick_48',
    'dn_wick_16','dn_wick_48',
    'ret_sharpe_48','ret_sharpe_96',
    'ret_acf_16','ret_acf_48',
]
MB = 4
RF_N_EST = 40
RF_DEPTH = 8
RF_LEAF = 50

# Per-coin optimal thresholds (optimized via CPCV OOS signals)
TH_MAP = {
    'BTCUSDT': 0.10, 'ETHUSDT': 0.10, 'SOLUSDT': 0.10, 'BNBUSDT': 0.19,
    'ADAUSDT': 0.06, 'XRPUSDT': 0.13, 'DOGEUSDT': 0.18, 'DOTUSDT': 0.14, 'AVAXUSDT': 0.10,
}

# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_binance(symbol='BTCUSDT', interval='15m'):
    """Load Binance data from monthly ZIP archives. Handles μs timestamps and CSV headers."""
    all_dfs = []
    for y in range(2020, 2027):
        for m in range(1, 13):
            f = DATA_DIR / f'{symbol}-{interval}-{y}-{m:02d}.zip'
            if not f.exists(): continue
            with zipfile.ZipFile(f) as z:
                csv = f'{symbol}-{interval}-{y}-{m:02d}.csv'
                if csv in z.namelist():
                    raw = z.read(csv).decode('utf-8')
                    first_line = raw.split('\n')[0].strip()
                    has_header = first_line.lower().startswith(('open_time', 'timestamp', 'opentime'))
                    from io import StringIO
                    df = pd.read_csv(StringIO(raw), header=0 if has_header else None)
                    # Normalize: strip column names regardless of format
                    df.columns = range(len(df.columns))
                    # Normalize μs timestamps to ms
                    ts_col = df.columns[0]  # first column
                    if df[ts_col].dtype in (np.int64, np.float64):
                        if df[ts_col].max() > 1e15:
                            df[ts_col] = df[ts_col] / 1000
                    all_dfs.append(df)
    if not all_dfs:
        raise FileNotFoundError(f'No data for {symbol}')
    df = pd.concat(all_dfs, ignore_index=True)
    cols = ['open_time','open','high','low','close','volume','close_time',
            'quote_vol','trades','taker_buy_vol','taker_buy_quote','ignore']
    df.columns = cols[:len(df.columns)]
    df['date'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
    for c in ['open','high','low','close','volume','quote_vol','taker_buy_vol','taker_buy_quote']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=['date']).reset_index(drop=True)
    return df


def compute_features(df):
    """Compute 26 ret features + 11 new v2 features. Returns (df, feat_names)."""
    df = df.copy()
    df['log_close'] = np.log(df['close'])
    df['log_vol'] = np.log1p(df['volume'])
    df['log_qv'] = np.log1p(df['quote_vol'])
    df['ret_1'] = df['log_close'].diff(); df['abs_ret_1'] = df['ret_1'].abs()
    for h in [2,4,8,16,24,32,48,64,96,128]:
        df[f'ret_{h}'] = df['log_close'].diff(h)
        df[f'abs_ret_{h}'] = df[f'ret_{h}'].abs()
    for w in [16,32,64,96]:
        df[f'ret_vol_corr_{w}'] = df['ret_1'].rolling(w).corr(df['log_vol']).fillna(0)

    # -- New v2 features --
    hl = (df['high'] - df['low']).clip(lower=1e-12)
    rng = hl / df['close'].clip(lower=1e-12)
    pos = (df['close'] - df['low']) / hl
    bfrac = df['taker_buy_vol'] / df['volume'].clip(lower=1e-12)
    for w in [16, 48, 96]:
        df[f'hl_range_{w}'] = rng.rolling(w).mean().fillna(0)
        df[f'close_pos_{w}'] = pos.rolling(w).mean().fillna(0.5)
        df[f'buy_frac_{w}'] = bfrac.rolling(w).mean().fillna(0.5)
    # Rolling skewness of ret_1
    def roll_skew(series, w):
        arr = series.to_numpy(np.float64)
        out = np.full(len(arr), np.nan, np.float64)
        from numpy.lib.stride_tricks import sliding_window_view
        if len(arr) >= w:
            sw = sliding_window_view(arr, w)
            m = sw.mean(axis=1)
            s = sw.std(axis=1, ddof=0).clip(1e-12)
            sk = np.mean((sw - m[:, None])**3, axis=1) / s**3
            out[w-1:] = sk
        return pd.Series(out, index=series.index).fillna(0)
    df['vol_skew_48'] = roll_skew(df['ret_1'], 48)
    df['vol_skew_96'] = roll_skew(df['ret_1'], 96)
    # Vol ratio: volatility term structure
    ret_arr = df['ret_1'].to_numpy(np.float64)
    vol_16 = pd.Series([np.std(ret_arr[max(0,i-16):i]) for i in range(len(ret_arr))])
    vol_96 = pd.Series([np.std(ret_arr[max(0,i-96):i]) for i in range(len(ret_arr))])
    df['vol_ratio_16_96'] = (vol_16 / vol_96.clip(lower=1e-12)).fillna(1.0)

    df = df.loc[:, ~df.columns.duplicated()]
    # Rolling stats helper
    def roll_mean(arr, w):
        return pd.Series(arr).rolling(w, min_periods=1).mean().to_numpy()
    def roll_std(arr, w):
        return pd.Series(arr).rolling(w, min_periods=1).std(ddof=0).to_numpy()
    def roll_max(arr, w):
        return pd.Series(arr).rolling(w, min_periods=1).max().to_numpy()

    # -- v3: trend quality --
    close = df['close'].to_numpy(np.float64)
    ret1 = df['ret_1'].to_numpy(np.float64)
    abs_ret1 = df['abs_ret_1'].to_numpy(np.float64)
    for w in [8, 24]:
        # Consecutive up count (streak of positive returns)
        ups = (ret1 > 0).astype(float)
        consec = np.zeros(len(ret1))
        cnt = 0
        for i in range(len(ret1)):
            cnt = cnt + 1 if ups[i] > 0 else 0
            consec[i] = cnt
        df[f'consec_up_{w}'] = pd.Series(consec).rolling(w, min_periods=1).mean().fillna(0)

    for w in [16, 48]:
        # Vol compression: fraction of bars with below-average vol
        abs_sma = roll_mean(abs_ret1, w).clip(1e-12)
        df[f'consec_vol_{w}'] = pd.Series((abs_ret1 < abs_sma).astype(float)).rolling(
            w, min_periods=1).mean().fillna(0.5)

    # -- v3: price structure --
    for w in [48, 96]:
        sma = roll_mean(close, w).clip(1e-12)
        df[f'ret_ma_dist_{w}'] = (close / sma - 1).clip(-0.2, 0.2)
        cmin = pd.Series(close).rolling(w, min_periods=1).min().to_numpy()
        cmax = pd.Series(close).rolling(w, min_periods=1).max().to_numpy()
        rng_pos = cmax - cmin
        df[f'ret_range_pos_{w}'] = np.where(
            rng_pos > 1e-8, (close - cmin) / rng_pos, 0.5)

    # -- v3: volatility dynamics --
    for w in [16, 48]:
        vol_now = roll_mean(abs_ret1, w)
        vol_prev = np.concatenate([[0], vol_now[:-1]])
        df[f'vol_delta_{w}'] = (vol_now - vol_prev).clip(-0.01, 0.01) * 100  # in %
        df[f'vol_max_{w}'] = roll_max(abs_ret1, w) * 100  # in %

    # -- v3: volume dynamics --
    vol_arr = df['volume'].to_numpy(np.float64)
    qv_arr = df['quote_vol'].to_numpy(np.float64)
    for w in [16, 48]:
        vol_sma = roll_mean(vol_arr, w).clip(1e-12)
        qv_sma = roll_mean(qv_arr, w).clip(1e-12)
        df[f'vol_surge_{w}'] = (vol_arr / vol_sma - 1).clip(-0.8, 3)
        df[f'qv_surge_{w}'] = (qv_arr / qv_sma - 1).clip(-0.8, 3)

    # -- v3: wick structure --
    hi = df['high'].to_numpy(np.float64)
    lo = df['low'].to_numpy(np.float64)
    op = df['open'].to_numpy(np.float64)
    hl = (hi - lo).clip(1e-12)
    up_wick = (hi - np.maximum(op, close)) / hl
    dn_wick = (np.minimum(op, close) - lo) / hl
    for w in [16, 48]:
        df[f'up_wick_{w}'] = roll_mean(up_wick, w)
        df[f'dn_wick_{w}'] = roll_mean(dn_wick, w)

    # -- v3: risk-adjusted return --
    for w in [48, 96]:
        m = roll_mean(ret1, w)
        s = roll_std(ret1, w).clip(1e-12)
        df[f'ret_sharpe_{w}'] = (m / s).clip(-3, 3)

    # -- v3: return autocorrelation --
    for w in [16, 48]:
        lag1 = np.concatenate([[0], ret1[:-1]])
        num = pd.Series(ret1 * lag1).rolling(w, min_periods=1).mean().to_numpy()
        den = pd.Series(ret1**2).rolling(w, min_periods=1).mean().clip(1e-12).to_numpy()
        df[f'ret_acf_{w}'] = (num / den).clip(-0.5, 0.5)

    df = df.loc[:, ~df.columns.duplicated()]
    feat_names = [c for c in df.columns if (
        c.startswith('ret_') or c.startswith('abs_ret_')
        or c.startswith('ret_vol_corr_')
        or c.startswith('hl_range_') or c.startswith('close_pos_')
        or c.startswith('buy_frac_') or c.startswith('vol_skew_')
        or c.startswith('vol_ratio_')
        or c.startswith('consec_up_') or c.startswith('consec_vol_')
        or c.startswith('ret_ma_dist_') or c.startswith('ret_range_pos_')
        or c.startswith('vol_delta_') or c.startswith('vol_max_')
        or c.startswith('vol_surge_') or c.startswith('qv_surge_')
        or c.startswith('up_wick_') or c.startswith('dn_wick_')
        or c.startswith('ret_sharpe_') or c.startswith('ret_acf_')
    )]
    return df, feat_names


# ═══════════════════════════════════════════════════════════════
# Triple Barrier Labels (vectorized via sliding_window_view)
# ═══════════════════════════════════════════════════════════════

def tb_labels(close, vol, upper=1.0, lower=1.0, mb=96, entry_offset=0):
    """Triple Barrier labels -- fully vectorized via sliding_window_view.

    entry_offset: number of bars to shift entry price forward.
        - 0: enter at close[i], measure close[i+1..i+mb] (close)
        - 1: enter at close[i+1], measure close[i+2..i+mb+1] (VWAP)
    Returns (label +/-1/0, pnl, hb) arrays.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    close = np.asarray(close, np.float64)
    n = len(close)
    ur = upper * vol
    lr = -lower * vol
    lbl = np.zeros(n, np.int8)
    pnl = np.zeros(n, np.float32)
    hb = np.zeros(n, np.int16)

    n_valid = n - mb - entry_offset if entry_offset > 0 else n - mb
    if n_valid <= 0:
        return lbl, pnl, hb

    start = entry_offset
    end = n_valid + entry_offset
    windows = sliding_window_view(close[start:end + mb], mb + 1)
    mat = windows[:, 1:] / windows[:, [0]] - 1

    up_hit = np.any(mat >= ur, axis=1)
    dn_hit = np.any(mat <= lr, axis=1)
    up_pos = np.argmax(mat >= ur, axis=1) + 1; up_pos[~up_hit] = mb + 1
    dn_pos = np.argmax(mat <= lr, axis=1) + 1; dn_pos[~dn_hit] = mb + 1

    is_up = up_hit & ((up_pos < dn_pos) | ~dn_hit)
    is_dn = dn_hit & ((dn_pos < up_pos) | ~up_hit)

    up_idx = np.where(is_up)[0]
    dn_idx = np.where(is_dn)[0]

    if len(up_idx) > 0:
        lbl[up_idx] = 1
        h_up = up_pos[up_idx] - 1
        pnl[up_idx] = mat[up_idx, h_up]
        hb[up_idx] = up_pos[up_idx]
    if len(dn_idx) > 0:
        lbl[dn_idx] = -1
        h_dn = dn_pos[dn_idx] - 1
        pnl[dn_idx] = -mat[dn_idx, h_dn]
        hb[dn_idx] = dn_pos[dn_idx]
    return lbl, pnl, hb


# ═══════════════════════════════════════════════════════════════
# CPCV per Lopez de Prado
# ═══════════════════════════════════════════════════════════════

def cpcv_eval(X, close, ret_1, dates, sigma=1.0, mb=96,
              n_blocks=6, k_train=None,
              n_est=40, depth=8, leaf=50, verbose=True,
              vwap=None, vwap_ret_1=None):
    """Full Combinatorial Purged Cross-Validation.

    If vwap is provided, auto-derives entry_offset=1, signal_return_offset=2.
    Otherwise entry_offset=0, signal_return_offset=1.
    """
    if k_train is None:
        k_train = max(1, n_blocks - 2)

    n = len(close)

    # -- Auto-derive timing offsets --
    if vwap is not None:
        price = vwap
        ret = vwap_ret_1
        entry_offset = 1
        signal_return_offset = 2
    else:
        price = close
        ret = ret_1
        entry_offset = 0
        signal_return_offset = 1

    # -- Precompute TB labels --
    train_vol = ret[:int(n * 0.70)].std() * np.sqrt(mb)
    labels, pnl, hbar = tb_labels(price, train_vol, sigma, sigma, mb, entry_offset=entry_offset)

    # -- eval_ret: VWAP returns for PnL (matches backtest) --
    eval_ret = np.zeros(n, np.float64)
    if signal_return_offset == 1:
        eval_ret[:-1] = price[1:] / price[:-1] - 1
    elif signal_return_offset == 2:
        eval_ret[:-2] = price[2:] / price[1:-1] - 1

    # -- Partition into N time-contiguous blocks --
    block_size = n // n_blocks
    block_bounds = []
    for b in range(n_blocks):
        lo = b * block_size
        hi = n if b == n_blocks - 1 else (b + 1) * block_size
        block_bounds.append((lo, hi))

    # -- C(N, k) combinatorial paths --
    from itertools import combinations
    paths = list(combinations(range(n_blocks), k_train))

    oos_sig = np.full(n, np.nan, np.float64)
    oos_pl = np.full(n, np.nan, np.float64)
    oos_ps = np.full(n, np.nan, np.float64)
    oos_cnt = np.zeros(n, np.int32)
    feat_imp = np.zeros(X.shape[1], np.float64)
    is_srs = np.zeros(len(paths), np.float64)
    oos_srs = np.zeros(len(paths), np.float64)

    if verbose:
        print(f'  CPCV: N={n_blocks}, k_train={k_train}, paths={len(paths)}')
        sizes = ', '.join([str(hi - lo) for lo, hi in block_bounds])
        print(f'  Block sizes: [{sizes}]')

    for path_idx, train_blks in enumerate(paths):
        test_blks = [b for b in range(n_blocks) if b not in train_blks]

        # Build train/test indices
        tr_idx = np.concatenate([np.arange(block_bounds[b][0], block_bounds[b][1]) for b in train_blks])
        te_idx = np.concatenate([np.arange(block_bounds[b][0], block_bounds[b][1]) for b in test_blks])
        tr_idx.sort(); te_idx.sort()

        # -- PURGE by index --
        min_te = te_idx[0]
        if tr_idx[-1] < min_te:
            cutoff = np.searchsorted(tr_idx, min_te - mb)
            tr_purged = tr_idx[:cutoff]
        elif tr_idx[0] > te_idx[-1]:
            tr_purged = tr_idx
        else:
            keep = np.ones(len(tr_idx), dtype=bool)
            for b in train_blks:
                lo, hi = block_bounds[b]
                te_after = [b2 for b2 in test_blks if b2 > b]
                if te_after:
                    next_test_start = block_bounds[min(te_after)][0]
                    mask = (tr_idx >= lo) & (tr_idx < hi) & (tr_idx >= next_test_start - mb)
                    keep[mask] = False
            tr_purged = tr_idx[keep]

        # -- EMBARGO: first mb bars of test blocks --
        embargo_mask = np.ones(len(te_idx), dtype=bool)
        for b in test_blks:
            lo, hi = block_bounds[b]
            mask = (te_idx >= lo) & (te_idx < lo + mb)
            embargo_mask[mask] = False
        te_purged = te_idx[embargo_mask]

        if len(tr_purged) < 10 or len(te_purged) < 5:
            is_srs[path_idx] = np.nan
            oos_srs[path_idx] = np.nan
            continue

        # -- Train RF (all 3 classes, including flat=0) --
        y_train = labels[tr_purged]
        if len(y_train) < 10:
            is_srs[path_idx] = np.nan
            oos_srs[path_idx] = np.nan
            continue

        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
            class_weight='balanced', random_state=42 + path_idx, n_jobs=-1)
        rf.fit(X[tr_purged], y_train.astype(int))

        # -- Predict (IS) --
        is_probs = rf.predict_proba(X[tr_purged])
        is_sig = _probs_to_signal(is_probs, rf.classes_)
        is_sig_1d = np.full(len(X), np.nan, np.float64)
        is_sig_1d[tr_purged] = is_sig

        # -- Predict (OOS) --
        oos_probs = rf.predict_proba(X[te_purged])
        pl, ps, _ = _probs_to_parts(oos_probs, rf.classes_)
        sig_vals = _probs_to_signal(oos_probs, rf.classes_)
        sig_1d = np.full(len(X), np.nan, np.float64)
        sig_1d[te_purged] = sig_vals

        # Store OOS (average across paths)
        for k, idx in enumerate(te_purged):
            if oos_cnt[idx] == 0:
                oos_sig[idx] = sig_vals[k]
                oos_pl[idx] = pl[k]
                oos_ps[idx] = ps[k]
                oos_cnt[idx] = 1
            else:
                oos_sig[idx] += sig_vals[k]
                oos_pl[idx] += pl[k]
                oos_ps[idx] += ps[k]
                oos_cnt[idx] += 1

        feat_imp += rf.feature_importances_

        # Path-level SR (IS and OOS, using VWAP returns)
        is_srs[path_idx] = _path_sr(is_sig_1d, eval_ret, hbar, labels, dates, tr_purged)
        oos_srs[path_idx] = _path_sr(sig_1d, eval_ret, hbar, labels, dates, te_purged)

    feat_imp /= max(len(paths), 1)

    # -- Average OOS predictions --
    valid = oos_cnt > 0
    oos_sig[valid] /= oos_cnt[valid]
    oos_pl[valid] /= oos_cnt[valid]
    oos_ps[valid] /= oos_cnt[valid]

    # -- PBO --
    pbo = _compute_pbo(is_srs, oos_srs)

    # -- Threshold evaluation --
    results = _eval_thresholds(oos_sig, labels, eval_ret, hbar, valid, pbo, dates)

    results.update(
        oos_sig=oos_sig, oos_cnt=oos_cnt, labels=labels, pnl=pnl, hbar=hbar,
        oos_pl=oos_pl, oos_ps=oos_ps,
        feat_imp=feat_imp, is_srs=is_srs, oos_srs=oos_srs, pbo=pbo,
        n_paths=len(paths), signal_return_offset=signal_return_offset,
        entry_offset=entry_offset
    )
    return results


# -- Signal conversion --

def _probs_to_signal(probs, classes):
    """Convert RF predict_proba to [-1, +1] signal."""
    sig = np.zeros(len(probs))
    for ic, c in enumerate(classes):
        if c == 1:
            sig += probs[:, ic]
        elif c == -1:
            sig -= probs[:, ic]
    return sig


def _probs_to_parts(probs, classes):
    pl = np.zeros(len(probs))
    ps = np.zeros(len(probs))
    pf = np.zeros(len(probs))
    for ic, c in enumerate(classes):
        if c == 1:
            pl = probs[:, ic]
        elif c == -1:
            ps = probs[:, ic]
        elif c == 0:
            pf = probs[:, ic]
    return pl, ps, pf


# -- Path-level SR (VWAP returns, not label_pnl) --

def _path_sr(sig, vwap_ret_arr, hbar_arr, labels_arr, dates_arr, idx):
    """Compute SR for a single path over specified indices using VWAP returns."""
    sid = sig[idx]
    m = (sid > 0.1) | (sid < -0.1)  # threshold=0.1
    if m.sum() < 3:
        return 0.0
    sm = np.where(sid[m] > 0.1, 1.0, -1.0)
    r = sm * vwap_ret_arr[idx][m]

    trade_dates = dates_arr[idx][m].astype('datetime64[D]')
    daily_df = pd.DataFrame({'date': trade_dates, 'pnl': r})
    daily_r = daily_df.groupby('date')['pnl'].sum()
    all_dates = pd.date_range(dates_arr[idx].min().astype('datetime64[D]'),
                              dates_arr[idx].max().astype('datetime64[D]'), freq='D')
    daily_r = daily_r.reindex(all_dates, fill_value=0.0)
    if daily_r.std() <= 1e-10:
        return 0.0
    return daily_r.mean() / daily_r.std() * np.sqrt(252)


# -- PBO: Probability of Backtest Overfitting --

def _compute_pbo(is_srs, oos_srs):
    """Rank-based PBO per Lopez de Prado."""
    valid = ~np.isnan(is_srs) & ~np.isnan(oos_srs)
    is_v, oos_v = is_srs[valid], oos_srs[valid]
    m = len(is_v)
    if m < 2:
        return 0.5
    order = np.argsort(is_v)[::-1]
    sorted_oos = oos_v[order]
    gaps = []
    for r in range(1, m + 1):
        lam = r / m
        omega = (sorted_oos[:r] > 0).mean()
        gaps.append(lam - omega)
    return max(0.0, min(gaps))


# -- Threshold Evaluation --

def _eval_thresholds(oos_sig, labels, vwap_ret, hbar, valid, pbo, dates):
    """Evaluate thresholds on bars with non-zero TB labels using VWAP returns."""
    ths = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    sr_by_th = {}; n_by_th = {}; acc_by_th = {}
    bp_by_th = {}; hold_by_th = {}

    for th in ths:
        v = valid & (labels != 0)
        sv = oos_sig[v]; yv = labels[v]; pv = vwap_ret[v]; hv = hbar[v]; dv = dates[v]
        m = (sv > th) | (sv < -th)
        if m.sum() < 3:
            sr_by_th[th] = 0; n_by_th[th] = 0
            continue
        sm = np.where(sv[m] > th, 1.0, -1.0)
        # PnL = position x VWAP return (NOT label_pnl!)
        r = sm * pv[m]

        trade_dates = dv[m].astype('datetime64[D]')
        daily_df = pd.DataFrame({'date': trade_dates, 'pnl': r})
        daily_r = daily_df.groupby('date')['pnl'].sum()
        all_dates = pd.date_range(dv.min().astype('datetime64[D]'),
                                  dv.max().astype('datetime64[D]'), freq='D')
        daily_r = daily_r.reindex(all_dates, fill_value=0.0)

        if daily_r.std() <= 1e-10:
            sr_by_th[th] = 0; n_by_th[th] = m.sum()
            continue

        sr = daily_r.mean() / daily_r.std() * np.sqrt(252)

        nzm = yv[m] != 0
        sr_by_th[th] = sr; n_by_th[th] = m.sum()
        acc_by_th[th] = (sm[nzm] == yv[m][nzm]).mean() if nzm.sum() > 0 else 0
        bp_by_th[th] = r.mean() * 10000
        hold_by_th[th] = hv[m].mean()

    return dict(sr_by_th=sr_by_th, n_by_th=n_by_th, acc_by_th=acc_by_th,
                bp_by_th=bp_by_th, hold_by_th=hold_by_th)


# -- Print Results --

def print_results(r, name):
    print(f'\n{name}:')
    print(f'  CPCV: {r["n_paths"]} paths, PBO={r["pbo"]:.0%}',
          f'({"Low" if r["pbo"] < 0.3 else "Medium" if r["pbo"] < 0.5 else "High"})')
    print(f'  {"th":>5s} {"SR":>8s} {"n_trades":>9s} {"TBacc":>7s} {"bp":>7s} {"hold":>5s}')
    print('  ' + '-' * 51)
    for th in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        sr = r['sr_by_th'].get(th, 0)
        if sr == 0 and r['n_by_th'].get(th, 0) == 0:
            continue
        print(f'  {th:5.2f} {sr:8.2f} {r["n_by_th"].get(th, 0):9,d}'
              f' {r["acc_by_th"].get(th, 0):7.1%}'
              f' {r["bp_by_th"].get(th, 0):7.1f}'
              f' {r["hold_by_th"].get(th, 0):5.1f}')


# ═══════════════════════════════════════════════════════════════
# Shared: Inference + PnL (used by both backtest and online)
# ═══════════════════════════════════════════════════════════════

def infer_signal(df, feat_names, model):
    """Compute signal from DataFrame + feature list + trained model.
    Returns (signal_array, p_long, p_short, timestamps, vwaps, closes) aligned to feat_names.
    """
    avail = [c for c in feat_names if c in df.columns]
    df_f = df[avail].dropna()
    if len(df_f) < 5:
        return None, None, None, None, None, None
    X = df_f.to_numpy(np.float32)
    probs = model['model'].predict_proba(X)
    sig = _probs_to_signal(probs, model['model'].classes_)
    pl, ps, _ = _probs_to_parts(probs, model['model'].classes_)
    idx = df_f.index
    dates = df['date'].iloc[idx].values
    vol = df['volume'].iloc[idx].values
    qv = df['quote_vol'].iloc[idx].values
    close = df['close'].iloc[idx].values
    vwap = np.where(vol > 0, qv / vol, close)
    return sig, pl, ps, dates, vwap, close


def ema_smooth(signals, alpha=0.50):
    """EMA smooth signal array in-place. alpha=0.50 -> halflife=1 bar."""
    s = np.asarray(signals, np.float64).copy()
    for t in range(1, len(s)):
        s[t] = alpha * s[t] + (1 - alpha) * s[t-1]
    return s


def backtest_pnl(signals, vwaps, threshold=0.20, fee=0.0004, ema_alpha=None):
    """Compute PnL from signals and VWAP prices.
    Uses VWAP timing (offset=2): signal[i] -> position[i+2], return[i+2] = V[i+2]/V[i+1]-1.
    If ema_alpha is set, applies EMA smoothing to signals before thresholding.
    Returns (pnl_array, positions_array, trades_count).
    """
    if ema_alpha is not None:
        signals = ema_smooth(signals, ema_alpha)
    n = len(signals)
    P = np.zeros(n)
    P[2:] = np.where(signals[:-2] > threshold, 1.0,
              np.where(signals[:-2] < -threshold, -1.0, 0.0))
    R = np.zeros(n)
    R[1:] = vwaps[1:] / vwaps[:-1] - 1
    dP = np.abs(np.diff(P))
    dP = np.concatenate([[0.0], dP])
    pnl = P * R - dP * fee
    trades = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], P]))) > 0))
    return pnl, P, trades
