"""
test_phase1.py — Phase 1 connectivity test
Run from project root: python test_phase1.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
logger.remove()
logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

RESULTS = []

def check(name, fn):
    try:
        ok = fn()
        RESULTS.append((name, "✅ PASS" if ok else "❌ FAIL"))
        return ok
    except Exception as e:
        RESULTS.append((name, f"❌ ERROR — {e}"))
        return False

# ── 1. PostgreSQL ──────────────────────────────────────────────────────────────
print("\n─── Testing PostgreSQL ───")
from db.connection import test_connection
check("PostgreSQL", test_connection)

# ── 2. yfinance prices ────────────────────────────────────────────────────────
print("─── Testing yfinance prices ───")
from clients.yfinance_client import YFinanceClient
yf_prices = YFinanceClient()
check("yfinance ping", yf_prices.ping)

for asset in ["BTC", "ETH", "GOLD", "OIL"]:
    def test_price(a=asset):
        df = yf_prices.get_candles(a, interval="1d", days=7)
        assert len(df) > 0, f"Empty response for {a}"
        print(f"  {a} latest close: {df['close'].iloc[-1]:,.2f}")
        return True
    check(f"yfinance {asset} price", test_price)

# ── 3. yfinance news ──────────────────────────────────────────────────────────
print("─── Testing yfinance news ───")
from clients.yfinance_news import YFinanceNewsClient
yf_news = YFinanceNewsClient()
check("yfinance news ping", yf_news.ping)

def btc_news():
    df = yf_news.get_news("BTC")
    assert len(df) > 0, "No BTC news returned"
    print(f"  Latest: {df['title'].iloc[0][:80]}")
    return True

check("yfinance BTC news", btc_news)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 50)
print("  PHASE 1 CONNECTIVITY RESULTS")
print("═" * 50)
col_w = max(len(r[0]) for r in RESULTS) + 2
for name, status in RESULTS:
    print(f"  {name:<{col_w}} {status}")
print("═" * 50)
passes = sum(1 for _, s in RESULTS if s.startswith("✅"))
fails  = sum(1 for _, s in RESULTS if s.startswith("❌"))
print(f"  {passes} passed  |  {fails} failed\n")
