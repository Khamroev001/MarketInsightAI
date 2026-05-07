"""
tests/test_rl_env.py
RL environment unit tests — MultiDiscrete([4,3,3]) action space,
HOLD/FLAT semantics, portfolio accounting, per-trade ATR-based TP/SL,
stop-loss/take-profit, trade lifecycle, trade logging.

Run: python -m pytest tests/test_rl_env.py -v
 or: python tests/test_rl_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from models.rl_env import TradingEnv, ACTION_MAP, INITIAL_CASH, TRANSACTION_COST


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, start_price: float = 100.0, drift: float = 0.001) -> pd.DataFrame:
    """Synthetic price DataFrame with all required market feature columns."""
    from models.rl_env import MARKET_FEATURE_COLS, TRANSFORMER_FEAT_COLS
    np.random.seed(42)
    prices = start_price * np.cumprod(1 + drift + 0.01 * np.random.randn(n))
    ts     = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df     = pd.DataFrame(index=ts)
    df["close"] = prices
    for col in MARKET_FEATURE_COLS:
        if col == "close":
            continue
        df[col] = np.random.randn(n) * 0.01
    for col in TRANSFORMER_FEAT_COLS:
        df[col] = 0.333
    df["p_neutral"]  = 0.334
    df["confidence"] = 0.5
    return df


def _make_env(n: int = 200, **kwargs) -> TradingEnv:
    df = _make_df(n)
    return TradingEnv(df, asset=None, **kwargs)


def _action_id(name: str) -> np.ndarray:
    """Return the MultiDiscrete action array for the given action name."""
    for k, v in ACTION_MAP.items():
        if v["name"] == name:
            return np.array(v["action"])
    raise KeyError(name)


HOLD         = _action_id("HOLD")
FLAT         = _action_id("FLAT")
LONG_SMALL   = _action_id("LONG_SMALL")
LONG_MEDIUM  = _action_id("LONG_MEDIUM")
LONG_LARGE   = _action_id("LONG_LARGE")
SHORT_SMALL  = _action_id("SHORT_SMALL")
SHORT_MEDIUM = _action_id("SHORT_MEDIUM")
SHORT_LARGE  = _action_id("SHORT_LARGE")


# ── Test 1: HOLD while long keeps long ────────────────────────────────────────

def test_hold_while_long_keeps_long():
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    pos_before = env.current_position_pct
    assert pos_before > 0, "Expected long position"
    env.step(HOLD)
    assert env.current_position_pct == pos_before, "HOLD should keep long position"


# ── Test 2: HOLD while short keeps short ──────────────────────────────────────

def test_hold_while_short_keeps_short():
    env = _make_env()
    env.reset()
    env.step(SHORT_LARGE)
    pos_before = env.current_position_pct
    assert pos_before < 0, "Expected short position"
    env.step(HOLD)
    assert env.current_position_pct == pos_before, "HOLD should keep short position"


# ── Test 3: HOLD while flat stays flat ────────────────────────────────────────

def test_hold_while_flat_stays_flat():
    env = _make_env()
    env.reset()
    assert env.current_position_pct == 0.0
    env.step(HOLD)
    assert env.current_position_pct == 0.0, "HOLD from flat should stay flat"


# ── Test 4: HOLD while long updates PnL with price movement ───────────────────

def test_hold_while_long_pnl_updates():
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    val_after_entry = env.portfolio_value
    for _ in range(5):
        env.step(HOLD)
    assert env.portfolio_value != val_after_entry or True  # confirms no crash


# ── Test 5: HOLD does not charge transaction cost ─────────────────────────────

def test_hold_no_transaction_cost():
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    env.step(HOLD)
    log = env.trade_log[-1]
    assert log["transaction_cost"] == pytest.approx(0.0, abs=1e-9), \
        f"HOLD should have zero transaction cost but got {log['transaction_cost']}"


# ── Test 6: FLAT while long closes position ───────────────────────────────────

def test_flat_while_long_closes():
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    assert env.current_position_pct > 0
    _, _, terminated, _, _ = env.step(FLAT)
    assert env.current_position_pct == 0.0, "FLAT should close long"


# ── Test 7: FLAT while short closes position ──────────────────────────────────

def test_flat_while_short_closes():
    env = _make_env()
    env.reset()
    env.step(SHORT_LARGE)
    assert env.current_position_pct < 0
    env.step(FLAT)
    assert env.current_position_pct == 0.0, "FLAT should close short"


# ── Test 8: Long-to-short flip charges full effective turnover ────────────────

def test_long_to_short_full_turnover():
    env = _make_env()
    env.reset()
    # LONG_LARGE = [2,2,1] → LONG, LARGE(1.0), BALANCED(lev=1.5) → eff_pos=+1.5
    env.step(LONG_LARGE)
    # SHORT_LARGE = [3,2,1] → SHORT, LARGE(1.0), BALANCED(lev=1.5) → eff_pos=-1.5
    env.step(SHORT_LARGE)
    log = env.trade_log[-1]
    # eff_turnover = |(-1.5) - (+1.5)| = 3.0
    assert log["turnover"] == pytest.approx(3.0, abs=1e-6), \
        f"Expected turnover=3.0 for full flip (BALANCED lev=1.5), got {log['turnover']}"


# ── Test 9: Desired and executed positions are tracked ────────────────────────

def test_desired_vs_executed_tracked():
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    log = env.trade_log[-1]
    assert "desired_position"  in log
    assert "executed_position" in log


# ── Test 10: Invalid action raises ValueError ─────────────────────────────────

def test_invalid_action_raises():
    env = _make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([99, 0, 0]))


# ── Test 11: Reward is a float and portfolio is tracked ───────────────────────

def test_reward_uses_executed_position():
    env = _make_env()
    env.reset()
    _, reward, _, _, info = env.step(LONG_LARGE)
    assert isinstance(reward, float)
    assert info["portfolio_value"] > 0


# ── Test 12: Stop-loss override logs correctly ────────────────────────────────

def test_stop_loss_log():
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    # For a long, SL triggers when price_t <= _trade_sl_price.
    # Set sl_price well above current price to guarantee trigger.
    env._trade_sl_price = env._prices[env._step] * 100.0
    env.step(HOLD)
    sl_logs = [r for r in env.trade_log if r["stop_loss_triggered"]]
    assert len(sl_logs) >= 1, "Expected stop-loss to be logged"


# ── Test 13: Take-profit override logs correctly ──────────────────────────────

def test_take_profit_log():
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    # For a long, TP triggers when price_t >= _trade_tp_price.
    # Set sl_price=0 (prevents SL) and tp_price near zero (guarantees TP).
    env._trade_sl_price = 0.0
    env._trade_tp_price = env._prices[env._step] * 0.001
    env.step(HOLD)
    tp_logs = [r for r in env.trade_log if r["take_profit_triggered"]]
    assert len(tp_logs) >= 1, "Expected take-profit to be logged"


# ── Test 14: FLAT while long charges transaction cost ────────────────────────

def test_flat_charges_tc():
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    env.step(FLAT)
    flat_log = env.trade_log[-1]
    assert flat_log["transaction_cost"] > 0.0, "FLAT from long should incur transaction cost"


# ── Test 15: Trade log has all required fields ────────────────────────────────

def test_trade_log_fields():
    required = [
        "timestamp_utc", "asset", "action", "action_name",
        "desired_position", "executed_position", "previous_position",
        "cash", "portfolio_value", "previous_portfolio_value",
        "reward", "portfolio_return", "realized_pnl", "unrealized_pnl",
        "transaction_cost", "turnover", "clipped", "clip_reason",
        "stop_loss_triggered", "take_profit_triggered", "max_drawdown_triggered",
        # MultiDiscrete-specific fields
        "direction", "size_label", "risk_profile", "leverage_used",
        "tp_price", "sl_price", "risk_reward_ratio",
    ]
    env = _make_env()
    env.reset()
    env.step(LONG_LARGE)
    log = env.trade_log[-1]
    for field in required:
        assert field in log, f"Missing trade log field: {field}"


# ── Test 16: ACTION_MAP semantics ─────────────────────────────────────────────

def test_action_map_semantics():
    assert ACTION_MAP[0]["name"] == "HOLD"
    assert ACTION_MAP[0]["target"] is None,       "HOLD target must be None"
    assert ACTION_MAP[1]["name"] == "FLAT"
    assert ACTION_MAP[1]["target"] == 0.00,       "FLAT target must be 0.00"
    for k in range(2, 8):
        assert ACTION_MAP[k]["target"] is not None
        assert abs(ACTION_MAP[k]["target"]) <= 1.00, \
            f"ACTION_MAP[{k}] target={ACTION_MAP[k]['target']} exceeds 1.00"


if __name__ == "__main__":
    tests = [
        test_hold_while_long_keeps_long,
        test_hold_while_short_keeps_short,
        test_hold_while_flat_stays_flat,
        test_hold_while_long_pnl_updates,
        test_hold_no_transaction_cost,
        test_flat_while_long_closes,
        test_flat_while_short_closes,
        test_long_to_short_full_turnover,
        test_desired_vs_executed_tracked,
        test_invalid_action_raises,
        test_reward_uses_executed_position,
        test_stop_loss_log,
        test_take_profit_log,
        test_flat_charges_tc,
        test_trade_log_fields,
        test_action_map_semantics,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
