#!/usr/bin/env python3
"""
Download Binance 15m kline data for all 9 portfolio coins.

Usage:
    python download_data.py              # Download all coins (default)
    python download_data.py --coins BTC  # Just BTC
    python download_data.py --year 2025  # Just 2025

Data source: https://data.binance.vision/
"""

import sys, argparse, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

DATA_DIR = Path(__file__).parent / 'data'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT',
           'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT']
INTERVAL = '15m'
BASE_URL = 'https://data.binance.vision/data/spot/monthly/klines'
# Binance data available from ~2017, but our coins start at different times
YEAR_RANGE = (2020, 2026)


def download_file(url, dest, retries=3):
    """Download with retries and basic UA to avoid being blocked."""
    req = Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; SylvaQuant/1.0)',
        'Accept': 'application/zip',
    })
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, 'wb') as f:
                f.write(data)
            return True, len(data)
        except HTTPError as e:
            if e.code == 404:
                return False, 0  # File doesn't exist (coin wasn't trading yet)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except (URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return False, 0


def main():
    parser = argparse.ArgumentParser(description='Download Binance 15m data')
    parser.add_argument('--coins', nargs='+', default=SYMBOLS,
                        help=f'Coins to download (default: all)')
    parser.add_argument('--year', type=int, default=None,
                        help='Specific year (default: all years)')
    args = parser.parse_args()

    coins = [c.upper() for c in args.coins]
    years = [args.year] if args.year else list(range(YEAR_RANGE[0], YEAR_RANGE[1] + 1))

    total, ok, skip = 0, 0, 0
    for sym in coins:
        for y in years:
            for m in range(1, 13):
                filename = f'{sym}-{INTERVAL}-{y}-{m:02d}.zip'
                dest = DATA_DIR / filename
                if dest.exists() and dest.stat().st_size > 1000:
                    print(f'  ✓ {filename}  (already exists, {dest.stat().st_size // 1024} KB)')
                    ok += 1
                    total += 1
                    continue

                url = f'{BASE_URL}/{sym}/{INTERVAL}/{filename}'
                success, size = download_file(url, dest)
                total += 1
                if success:
                    print(f'  ✓ {filename}  ({size // 1024} KB)')
                    ok += 1
                else:
                    skip += 1
                    # Quiet for 404s (coin not yet listed in that month)
                    if m == 1 or m == 12:
                        print(f'  - {filename}  (not available)')

    print(f'\nDone: {ok}/{total} files downloaded ({skip} skipped, not yet listed)')


if __name__ == '__main__':
    main()
