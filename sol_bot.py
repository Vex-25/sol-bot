"""
SOL/USD RSI mean-reversion bot (Alpaca PAPER trading)

Strategy (from the group's plan)
  - Long only, one position at a time, SOL/USD only
  - 4-hour candles, RSI(14), no trend filter
  - BUY : last CLOSED candle has RSI < 40 and we are flat -> spend 20% of cash
  - SELL: whichever comes first
        1) RSI > 60 on a closed candle
        2) stop-loss   : price <= entry * (1 - 8%)
        3) take-profit : price >= entry * (1 + 4%)
  - Buy/RSI-sell signals are judged once per closed candle.
    Stop-loss / take-profit are checked on every check (every 15 minutes).

Setup (Command Prompt):
    py -m pip install alpaca-py
    Keys: either the setx environment variables, or a .env file next to this
    script containing two lines:
        ALPACA_API_KEY=your_paper_key
        ALPACA_SECRET_KEY=your_paper_secret

Run:
    py sol_bot.py          (loops forever, checks every 15 minutes)
    py sol_bot.py --once   (one check, then exit - for schedulers)
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

# ----------------------------- CONFIG ---------------------------------------
SYMBOL = "SOL/USD"
POSITION_SYMBOL = "SOLUSD"      # how Alpaca names the position/close endpoint
CANDLE_HOURS = 4
RSI_PERIOD = 14
RSI_BUY_BELOW = 40
RSI_SELL_ABOVE = 60
STOP_LOSS_PCT = 0.08
TAKE_PROFIT_PCT = 0.04
POSITION_FRACTION = 0.20        # 20% of current cash per trade
MIN_ORDER_USD = 10              # safety floor, skip tiny orders
CHECK_SECONDS = 15 * 60
LOG_FILE = "trade_log.csv"
# -----------------------------------------------------------------------------


def load_env(path=".env"):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY (env vars or .env file).")

trading = TradingClient(API_KEY, SECRET_KEY, paper=True)  # PAPER ONLY
data = CryptoHistoricalDataClient()


def now_utc():
    return datetime.now(timezone.utc)


def log(action, price, detail, rsi_value):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "action", "symbol", "price", "detail", "rsi"])
        w.writerow([now_utc().isoformat(), action, SYMBOL, price, detail, round(rsi_value, 1)])
    print(f"  >>> {action} {SYMBOL} @ {price:.2f} | {detail} | RSI {rsi_value:.1f}")


def rsi(closes, n=14):
    """Wilder's RSI on a list of closes (same function as backtest.py)."""
    if len(closes) < n + 1:
        return None
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_g = sum(gains[:n]) / n
    avg_l = sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)


def get_closed_bars():
    """Return (list of closes, close time of the last CLOSED candle)."""
    start = now_utc() - timedelta(hours=CANDLE_HOURS * (RSI_PERIOD + 80))
    req = CryptoBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(CANDLE_HOURS, TimeFrameUnit.Hour),
        start=start,
    )
    bars = list(data.get_crypto_bars(req).data.get(SYMBOL, []))
    width = timedelta(hours=CANDLE_HOURS)
    # drop the candle that is still forming
    while bars and bars[-1].timestamp + width > now_utc():
        bars.pop()
    if not bars:
        return [], None
    return [b.close for b in bars], bars[-1].timestamp + width


def latest_price():
    trade = data.get_crypto_latest_trade(CryptoLatestTradeRequest(symbol_or_symbols=SYMBOL))
    return float(trade[SYMBOL].price)


def get_position():
    for p in trading.get_all_positions():
        if p.symbol.replace("/", "") == POSITION_SYMBOL:
            return p
    return None


def traded_since(t):
    """True if any order was placed after time t (enforces 'once per candle')."""
    try:
        orders = trading.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.ALL, symbols=[SYMBOL], after=t, limit=50))
        return len(orders) > 0
    except Exception as e:
        print(f"  (could not check recent orders: {e}) - skipping buy to be safe")
        return True


def step():
    closes, last_close_time = get_closed_bars()
    r = rsi(closes, RSI_PERIOD)
    if r is None:
        print("Not enough candle data yet.")
        return
    price = latest_price()
    pos = get_position()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    held = f"holding {float(pos.qty):.4f} SOL" if pos else "flat"
    print(f"[{stamp}] price={price:.2f} RSI(4h closed)={r:.1f} | {held}")

    if pos:
        entry = float(pos.avg_entry_price)
        if price <= entry * (1 - STOP_LOSS_PCT):
            trading.close_position(POSITION_SYMBOL)
            log("SELL", price, f"stop-loss {STOP_LOSS_PCT:.0%} (entry {entry:.2f})", r)
        elif price >= entry * (1 + TAKE_PROFIT_PCT):
            trading.close_position(POSITION_SYMBOL)
            log("SELL", price, f"take-profit {TAKE_PROFIT_PCT:.0%} (entry {entry:.2f})", r)
        elif r > RSI_SELL_ABOVE:
            trading.close_position(POSITION_SYMBOL)
            log("SELL", price, f"RSI > {RSI_SELL_ABOVE} (entry {entry:.2f})", r)
        return

    if r < RSI_BUY_BELOW and not traded_since(last_close_time):
        cash = float(trading.get_account().cash)
        notional = round(cash * POSITION_FRACTION, 2)
        if notional < MIN_ORDER_USD:
            print(f"  Order size ${notional} below minimum, skipping.")
            return
        trading.submit_order(MarketOrderRequest(
            symbol=SYMBOL, notional=notional, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
        log("BUY", price, f"RSI < {RSI_BUY_BELOW}, spent ${notional} (20% of cash)", r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run one check and exit")
    args = parser.parse_args()

    acct = trading.get_account()
    print(f"Connected to Alpaca PAPER account. Equity ${float(acct.equity):,.2f}, cash ${float(acct.cash):,.2f}")

    if args.once:
        step()
        return

    print(f"Trading {SYMBOL}. Checking every {CHECK_SECONDS // 60} min. Ctrl+C to stop.\n")
    while True:
        try:
            step()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error (will retry): {e}")
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped.")
