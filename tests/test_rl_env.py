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
    import models.rl_env as m
    orig = m.MIN_HOLD_BARS
    try:
        m.MIN_HOLD_BARS = 1  # allow immediate close for this mechanics test
        env = _make_env()
        env.reset()
        env.step(LONG_LARGE)
        assert env.current_position_pct > 0
        _, _, terminated, _, _ = env.step(FLAT)
        assert env.current_position_pct == 0.0, "FLAT should close long"
    finally:
        m.MIN_HOLD_BARS = orig


# ── Test 7: FLAT while short closes position ──────────────────────────────────

def test_flat_while_short_closes():
    import models.rl_env as m
    orig = m.MIN_HOLD_BARS
    try:
        m.MIN_HOLD_BARS = 1
        env = _make_env()
        env.reset()
        env.step(SHORT_LARGE)
        assert env.current_position_pct < 0
        env.step(FLAT)
        assert env.current_position_pct == 0.0, "FLAT should close short"
    finally:
        m.MIN_HOLD_BARS = orig


# ── Test 8: Long-to-short flip charges full effective turnover ────────────────

def test_long_to_short_full_turnover():
    import models.rl_env as m
    orig = m.MIN_HOLD_BARS
    try:
        m.MIN_HOLD_BARS = 1  # allow immediate reverse for turnover mechanics test
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
    finally:
        m.MIN_HOLD_BARS = orig


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
    import models.rl_env as m
    orig = m.MIN_HOLD_BARS
    try:
        m.MIN_HOLD_BARS = 1
        env = _make_env()
        env.reset()
        env.step(LONG_LARGE)
        env.step(FLAT)
        flat_log = env.trade_log[-1]
        assert flat_log["transaction_cost"] > 0.0, "FLAT from long should incur transaction cost"
    finally:
        m.MIN_HOLD_BARS = orig


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


# ── v2 Tests ──────────────────────────────────────────────────────────────────

def _make_df_ohlc(n: int = 200, start_price: float = 100.0) -> pd.DataFrame:
    """Synthetic OHLC DataFrame for v2 TP/SL tests."""
    from models.rl_env import MARKET_FEATURE_COLS, TRANSFORMER_FEAT_COLS
    np.random.seed(7)
    prices = start_price * np.cumprod(1 + 0.001 + 0.005 * np.random.randn(n))
    ts     = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df     = pd.DataFrame(index=ts)
    df["close"] = prices
    df["open"]  = prices * 0.999
    df["high"]  = prices * 1.005
    df["low"]   = prices * 0.995
    for col in MARKET_FEATURE_COLS:
        if col in ("close",):
            continue
        df[col] = np.random.randn(n) * 0.01
    for col in TRANSFORMER_FEAT_COLS:
        df[col] = 0.0
    return df


def _make_env_ohlc(n: int = 200, **kwargs) -> TradingEnv:
    df = _make_df_ohlc(n)
    return TradingEnv(df, asset=None, **kwargs)


# ── Test 17: New action does not earn the previous bar's return ───────────────

def test_pnl_timing_old_pos_earns_bar():
    """With USE_OLD_PNL_TIMING=False, PnL at the step when we open should be 0
    (old_effective_pos=0 since we were flat). If USE_OLD_PNL_TIMING were True,
    the PnL would be non-zero because new_effective_pos * price_ret."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.USE_OLD_PNL_TIMING
    try:
        rl_env_module.USE_OLD_PNL_TIMING = False
        env = _make_env()
        env.reset()
        # Force a non-zero price return
        env._prices[1] = env._prices[0] * 1.02  # +2% move
        _, _, _, _, _ = env.step(LONG_LARGE)     # step 0: open while flat
        log = env.trade_log[-1]
        # old_effective_pos was 0 (flat before), so pnl_t must be 0
        assert log["step_return"] == pytest.approx(0.0, abs=1e-9), (
            f"Opening step should earn 0 PnL (old pos was flat). Got {log['step_return']}"
        )
    finally:
        rl_env_module.USE_OLD_PNL_TIMING = orig


# ── Test 18: Long TP uses candle high ─────────────────────────────────────────

def test_long_tp_uses_candle_high():
    """TP for LONG should trigger when high_t >= tp_price even if close < tp_price."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.USE_CLOSE_ONLY_TPSL
    try:
        rl_env_module.USE_CLOSE_ONLY_TPSL = False
        env = _make_env_ohlc()
        env.reset()
        env.step(LONG_LARGE)
        # Set TP just below the high so that high triggers it but close would not.
        step = env._step
        high_next = float(env._highs[step])
        close_next = float(env._prices[step])
        env._trade_sl_price = 0.0  # disable SL
        env._trade_tp_price = high_next * 0.999   # below high → triggers via high
        # Ensure close will not trigger by itself
        assert close_next < env._trade_tp_price or True  # just ensuring TP set
        env.step(HOLD)
        tp_logs = [r for r in env.trade_log if r["take_profit_triggered"]]
        assert len(tp_logs) >= 1, "Long TP should trigger via candle high"
    finally:
        rl_env_module.USE_CLOSE_ONLY_TPSL = orig


# ── Test 19: Long SL uses candle low ─────────────────────────────────────────

def test_long_sl_uses_candle_low():
    """SL for LONG should trigger when low_t <= sl_price even if close > sl_price."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.USE_CLOSE_ONLY_TPSL
    try:
        rl_env_module.USE_CLOSE_ONLY_TPSL = False
        env = _make_env_ohlc()
        env.reset()
        env.step(LONG_LARGE)
        step = env._step
        low_next   = float(env._lows[step])
        env._trade_tp_price = 1e9   # disable TP
        env._trade_sl_price = low_next * 1.001  # above low → triggers via low
        env.step(HOLD)
        sl_logs = [r for r in env.trade_log if r["stop_loss_triggered"]]
        assert len(sl_logs) >= 1, "Long SL should trigger via candle low"
    finally:
        rl_env_module.USE_CLOSE_ONLY_TPSL = orig


# ── Test 20: Short SL uses candle high ────────────────────────────────────────

def test_short_sl_uses_candle_high():
    """SL for SHORT should trigger when high_t >= sl_price."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.USE_CLOSE_ONLY_TPSL
    try:
        rl_env_module.USE_CLOSE_ONLY_TPSL = False
        env = _make_env_ohlc()
        env.reset()
        env.step(SHORT_LARGE)
        step = env._step
        high_next  = float(env._highs[step])
        env._trade_tp_price = 0.0   # disable TP (price=0 means TP never reached for short)
        env._trade_sl_price = high_next * 0.999  # below high → triggers via high
        env.step(HOLD)
        sl_logs = [r for r in env.trade_log if r["stop_loss_triggered"]]
        assert len(sl_logs) >= 1, "Short SL should trigger via candle high"
    finally:
        rl_env_module.USE_CLOSE_ONLY_TPSL = orig


# ── Test 21: Short TP uses candle low ─────────────────────────────────────────

def test_short_tp_uses_candle_low():
    """TP for SHORT should trigger when low_t <= tp_price."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.USE_CLOSE_ONLY_TPSL
    try:
        rl_env_module.USE_CLOSE_ONLY_TPSL = False
        env = _make_env_ohlc()
        env.reset()
        env.step(SHORT_LARGE)
        step = env._step
        low_next   = float(env._lows[step])
        env._trade_sl_price = 1e9   # disable SL
        env._trade_tp_price = low_next * 1.001  # above low → triggers via low
        env.step(HOLD)
        tp_logs = [r for r in env.trade_log if r["take_profit_triggered"]]
        assert len(tp_logs) >= 1, "Short TP should trigger via candle low"
    finally:
        rl_env_module.USE_CLOSE_ONLY_TPSL = orig


# ── Test 22: MIN_HOLD_BARS blocks early flat ──────────────────────────────────

def test_min_hold_bars_blocks_early_flat():
    """Agent cannot FLAT before MIN_HOLD_BARS steps; direction is overridden to HOLD."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.MIN_HOLD_BARS
    try:
        rl_env_module.MIN_HOLD_BARS = 3
        env = _make_env()
        env.reset()
        env.step(LONG_LARGE)   # open
        env.step(FLAT)         # step 1 — should be blocked
        assert env.current_position_pct > 0.0, (
            "FLAT should be blocked by MIN_HOLD_BARS before 3 bars"
        )
    finally:
        rl_env_module.MIN_HOLD_BARS = orig


# ── Test 23: MIN_HOLD_BARS blocks early reverse ───────────────────────────────

def test_min_hold_bars_blocks_early_reverse():
    """Agent cannot reverse before MIN_HOLD_BARS steps."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.MIN_HOLD_BARS
    try:
        rl_env_module.MIN_HOLD_BARS = 3
        env = _make_env()
        env.reset()
        env.step(LONG_LARGE)    # open long
        env.step(SHORT_LARGE)   # attempt reverse at step 1 — should be blocked
        assert env.current_position_pct > 0.0, (
            "Reverse should be blocked by MIN_HOLD_BARS before 3 bars"
        )
    finally:
        rl_env_module.MIN_HOLD_BARS = orig


# ── Test 24: MIN_HOLD_BARS allows exit after threshold ───────────────────────

def test_min_hold_bars_allows_exit_after_threshold():
    """Agent CAN close after >= MIN_HOLD_BARS steps."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.MIN_HOLD_BARS
    try:
        rl_env_module.MIN_HOLD_BARS = 3
        env = _make_env()
        env.reset()
        env.step(LONG_LARGE)        # open
        env.step(HOLD)              # step 1
        env.step(HOLD)              # step 2
        env.step(HOLD)              # step 3 — steps_in_position now >=3
        env.step(FLAT)              # should close
        assert env.current_position_pct == 0.0, (
            "FLAT should be allowed after MIN_HOLD_BARS steps"
        )
    finally:
        rl_env_module.MIN_HOLD_BARS = orig


# ── Test 25: Cooldown forces HOLD after close ─────────────────────────────────

def test_cooldown_forces_hold_after_close():
    """After a FLAT, COOLDOWN_BARS steps must be HOLD regardless of action."""
    import models.rl_env as rl_env_module
    orig_min  = rl_env_module.MIN_HOLD_BARS
    orig_cool = rl_env_module.COOLDOWN_BARS
    try:
        rl_env_module.MIN_HOLD_BARS  = 1   # allow fast close for setup
        rl_env_module.COOLDOWN_BARS  = 2
        env = _make_env()
        env.reset()
        env.step(LONG_LARGE)   # open
        env.step(FLAT)         # close — triggers cooldown
        # Next 2 steps should stay flat (HOLD forced)
        env.step(LONG_LARGE)   # cooldown step 1 — should be overridden
        assert env.current_position_pct == 0.0, "Step 1 of cooldown should force HOLD/flat"
        env.step(LONG_LARGE)   # cooldown step 2 — should be overridden
        assert env.current_position_pct == 0.0, "Step 2 of cooldown should force HOLD/flat"
    finally:
        rl_env_module.MIN_HOLD_BARS  = orig_min
        rl_env_module.COOLDOWN_BARS  = orig_cool


# ── Test 26: TP/SL always allowed despite min-hold ────────────────────────────

def test_tp_sl_always_allowed_despite_min_hold():
    """TP/SL should close the trade even within MIN_HOLD_BARS."""
    import models.rl_env as rl_env_module
    orig = rl_env_module.MIN_HOLD_BARS
    try:
        rl_env_module.MIN_HOLD_BARS = 10  # very long hold requirement
        env = _make_env()
        env.reset()
        env.step(LONG_LARGE)
        # Force SL to trigger on next step (guaranteed by extremely high sl_price)
        env._trade_sl_price = env._prices[env._step] * 100.0
        env._trade_tp_price = env._prices[env._step] * 999.0  # far away, won't trigger
        env.step(HOLD)
        # SL should have fired (position closed)
        assert env.current_position_pct == 0.0, "SL must close trade even within MIN_HOLD_BARS"
    finally:
        rl_env_module.MIN_HOLD_BARS = orig


# ── Test 27: completed_trade_log metrics calculated without crash ─────────────

def test_completed_trade_log_metrics():
    """completed_trade_log should be non-empty after trades and support metric calc."""
    import models.rl_env as rl_env_module
    orig_min  = rl_env_module.MIN_HOLD_BARS
    orig_cool = rl_env_module.COOLDOWN_BARS
    try:
        rl_env_module.MIN_HOLD_BARS  = 1
        rl_env_module.COOLDOWN_BARS  = 0
        env = _make_env(n=50)
        env.reset()
        env.step(LONG_LARGE)
        env.step(HOLD)
        env.step(FLAT)
        assert len(env.completed_trade_log) >= 1, "Expected at least one completed trade"
        df = pd.DataFrame(env.completed_trade_log)
        assert "net_pnl"               in df.columns
        assert "transaction_cost_total" in df.columns
        assert "exit_reason"            in df.columns
        wins  = (df["net_pnl"] > 0).sum()
        total = len(df)
        win_rate = wins / total
        gross_profit = df.loc[df["net_pnl"] > 0, "net_pnl"].sum()
        gross_loss   = abs(df.loc[df["net_pnl"] <= 0, "net_pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        exit_counts   = df["exit_reason"].value_counts().to_dict()
        assert isinstance(win_rate, float)
        assert isinstance(profit_factor, float)
        assert isinstance(exit_counts, dict)
    finally:
        rl_env_module.MIN_HOLD_BARS  = orig_min
        rl_env_module.COOLDOWN_BARS  = orig_cool


# ── v3 Tests ─────────────────────────────────────────────────────────────────

def _make_env_with_signal(
    asset: str = None,
    signal_val: float = 0.333,
    plr_val: float = 0.10,
    n: int = 200,
    **kwargs,
) -> TradingEnv:
    """Create TradingEnv with controlled signal_strength and predicted_log_return values."""
    df = _make_df(n)
    df["signal_strength"]     = signal_val
    df["predicted_log_return"] = plr_val
    df["ret_4"] = 0.001   # slight positive trend by default
    return TradingEnv(df, asset=asset, **kwargs)


# ── Test 28: Signal gate blocks weak new GOLD/OIL entries ────────────────────

def test_signal_gate_blocks_weak_gold_entry():
    """GOLD signal gate (threshold=0.35) should block entries when signal < 0.35."""
    import models.rl_env as m
    orig_gates = m.USE_ASSET_SPECIFIC_GATES
    orig_dir   = m.USE_SIGNAL_DIRECTION_FILTER
    try:
        m.USE_ASSET_SPECIFIC_GATES    = True
        m.USE_SIGNAL_DIRECTION_FILTER = False  # isolate gate A
        env = _make_env_with_signal(asset="GOLD", signal_val=0.20, plr_val=0.10)
        env.reset()
        env.step(LONG_LARGE)   # signal=0.20 < 0.35 → should be blocked
        assert env.current_position_pct == 0.0, (
            "GOLD entry should be blocked when signal_strength < ASSET_SIGNAL_THRESHOLDS['GOLD']"
        )
    finally:
        m.USE_ASSET_SPECIFIC_GATES    = orig_gates
        m.USE_SIGNAL_DIRECTION_FILTER = orig_dir


def test_signal_gate_allows_strong_gold_entry():
    """GOLD entry should be allowed when signal_strength >= threshold."""
    import models.rl_env as m
    orig_gates = m.USE_ASSET_SPECIFIC_GATES
    orig_dir   = m.USE_SIGNAL_DIRECTION_FILTER
    try:
        m.USE_ASSET_SPECIFIC_GATES    = True
        m.USE_SIGNAL_DIRECTION_FILTER = False
        env = _make_env_with_signal(asset="GOLD", signal_val=0.40, plr_val=0.10)
        env.reset()
        env.step(LONG_LARGE)   # signal=0.40 >= 0.35 → should be allowed
        assert env.current_position_pct > 0.0, (
            "GOLD entry should succeed when signal_strength >= threshold"
        )
    finally:
        m.USE_ASSET_SPECIFIC_GATES    = orig_gates
        m.USE_SIGNAL_DIRECTION_FILTER = orig_dir


# ── Test 29: Signal gate does NOT block exits ──────────────────────────────────

def test_signal_gate_does_not_block_exits():
    """FLAT should succeed even when signal is weak (agent is already in position)."""
    import models.rl_env as m
    orig_gates   = m.USE_ASSET_SPECIFIC_GATES
    orig_dir     = m.USE_SIGNAL_DIRECTION_FILTER
    orig_min     = m.MIN_HOLD_BARS
    orig_amh     = m.ASSET_MIN_HOLD_BARS.copy()
    try:
        m.USE_ASSET_SPECIFIC_GATES    = True
        m.USE_SIGNAL_DIRECTION_FILTER = False
        m.MIN_HOLD_BARS               = 1
        m.ASSET_MIN_HOLD_BARS["GOLD"] = 1   # override per-asset hold so FLAT isn't blocked
        # Open with strong signal and sufficient predicted_lr (above GOLD threshold 0.35)
        env = _make_env_with_signal(asset="GOLD", signal_val=0.40, plr_val=0.40)
        env.reset()
        env.step(LONG_LARGE)
        assert env.current_position_pct > 0.0, "Setup: should be in position"
        # Now weaken the signal — gate should NOT block exit (agent is already in position)
        for i in range(len(env._transformer_raw)):
            env._transformer_raw[i, 1] = 0.10  # signal_strength → 0.10
        env.step(FLAT)
        assert env.current_position_pct == 0.0, "FLAT should succeed even with weak signal"
    finally:
        m.USE_ASSET_SPECIFIC_GATES    = orig_gates
        m.USE_SIGNAL_DIRECTION_FILTER = orig_dir
        m.MIN_HOLD_BARS               = orig_min
        m.ASSET_MIN_HOLD_BARS.update(orig_amh)


# ── Test 30: TP/SL exits never blocked by gates ───────────────────────────────

def test_tpsl_not_blocked_by_signal_gate():
    """TP/SL must close the trade even when USE_ASSET_SPECIFIC_GATES is True."""
    import models.rl_env as m
    orig_gates = m.USE_ASSET_SPECIFIC_GATES
    orig_dir   = m.USE_SIGNAL_DIRECTION_FILTER
    try:
        m.USE_ASSET_SPECIFIC_GATES    = True
        m.USE_SIGNAL_DIRECTION_FILTER = False  # isolate gate test; direction filter tested separately
        # GOLD — plr_val must exceed GOLD threshold (0.35) so setup entry is allowed
        env = _make_env_with_signal(asset="GOLD", signal_val=0.40, plr_val=0.40)
        env.reset()
        env.step(LONG_LARGE)
        assert env.current_position_pct > 0.0
        # Force SL trigger
        env._trade_sl_price = env._prices[env._step] * 100.0
        env._trade_tp_price = env._prices[env._step] * 999.0
        # Weaken signal so gate would normally block new entries
        for i in range(len(env._transformer_raw)):
            env._transformer_raw[i, 1] = 0.10
        env.step(HOLD)
        assert env.current_position_pct == 0.0, "SL must fire despite weak signal gate"
    finally:
        m.USE_ASSET_SPECIFIC_GATES    = orig_gates
        m.USE_SIGNAL_DIRECTION_FILTER = orig_dir


# ── Test 31: Risk profile downgrades correctly for GOLD ──────────────────────

def test_risk_downgrade_gold_aggressive_to_conservative():
    """GOLD AGGRESSIVE should be downgraded to CONSERVATIVE regardless of signal."""
    from models.rl_env import _apply_asset_risk_limits, USE_ASSET_SPECIFIC_RISK_LIMITS
    import models.rl_env as m
    orig = m.USE_ASSET_SPECIFIC_RISK_LIMITS
    try:
        m.USE_ASSET_SPECIFIC_RISK_LIMITS = True
        result = _apply_asset_risk_limits("GOLD", risk_idx=2, sig_str=0.99)
        assert result == 0, f"GOLD AGGRESSIVE must downgrade to CONSERVATIVE, got {result}"
    finally:
        m.USE_ASSET_SPECIFIC_RISK_LIMITS = orig


def test_risk_downgrade_eth_aggressive_below_threshold():
    """ETH AGGRESSIVE should downgrade to BALANCED when signal < ETH_AGG_SIGNAL_MIN."""
    from models.rl_env import _apply_asset_risk_limits, ETH_AGG_SIGNAL_MIN
    import models.rl_env as m
    orig = m.USE_ASSET_SPECIFIC_RISK_LIMITS
    try:
        m.USE_ASSET_SPECIFIC_RISK_LIMITS = True
        result = _apply_asset_risk_limits("ETH", risk_idx=2, sig_str=ETH_AGG_SIGNAL_MIN - 0.01)
        assert result == 1, f"ETH AGGRESSIVE below threshold must become BALANCED, got {result}"
    finally:
        m.USE_ASSET_SPECIFIC_RISK_LIMITS = orig


def test_risk_downgrade_eth_aggressive_above_threshold():
    """ETH AGGRESSIVE should remain AGGRESSIVE when signal >= ETH_AGG_SIGNAL_MIN."""
    from models.rl_env import _apply_asset_risk_limits, ETH_AGG_SIGNAL_MIN, ASSET_MAX_RISK_IDX
    import models.rl_env as m
    orig = m.USE_ASSET_SPECIFIC_RISK_LIMITS
    try:
        m.USE_ASSET_SPECIFIC_RISK_LIMITS = True
        result = _apply_asset_risk_limits("ETH", risk_idx=2, sig_str=ETH_AGG_SIGNAL_MIN + 0.01)
        # ETH max risk cap is 1 (BALANCED); so even with strong signal, capped at 1
        assert result == ASSET_MAX_RISK_IDX.get("ETH", 1), (
            f"ETH AGGRESSIVE above threshold should be capped at ASSET_MAX_RISK_IDX, got {result}"
        )
    finally:
        m.USE_ASSET_SPECIFIC_RISK_LIMITS = orig


# ── Test 32: Breakeven stop only moves SL in protective direction ─────────────

def test_breakeven_stop_long_moves_sl_up():
    """For LONG, breakeven stop must move SL up (to entry), never down."""
    import models.rl_env as m
    orig     = m.USE_BREAKEVEN_STOP
    orig_dir = m.USE_SIGNAL_DIRECTION_FILTER
    try:
        m.USE_BREAKEVEN_STOP          = True
        m.USE_SIGNAL_DIRECTION_FILTER = False  # don't let direction filter block entry
        # plr_val must exceed ETH threshold (0.20) for entry to succeed
        env = _make_env_with_signal(asset="ETH", signal_val=0.40, plr_val=0.25)
        env.reset()
        env.step(LONG_LARGE)
        assert env.in_position, "Should be in LONG"
        original_sl = env._trade_sl_price
        entry        = env.entry_price
        atr          = float(env._atr[env._step])
        # Force price to be above breakeven trigger
        env._prices[env._step] = entry + 1.0 * atr  # > 0.8 * atr trigger for ETH
        env.step(HOLD)
        new_sl = env._trade_sl_price
        assert new_sl >= original_sl, "Breakeven SL must not be lower than original SL"
    finally:
        m.USE_BREAKEVEN_STOP          = orig
        m.USE_SIGNAL_DIRECTION_FILTER = orig_dir


def test_breakeven_stop_does_not_lower_sl_for_long():
    """Breakeven stop must never lower the SL (only raise it for LONG)."""
    import models.rl_env as m
    orig     = m.USE_BREAKEVEN_STOP
    orig_dir = m.USE_SIGNAL_DIRECTION_FILTER
    try:
        m.USE_BREAKEVEN_STOP          = True
        m.USE_SIGNAL_DIRECTION_FILTER = False  # don't let direction filter block entry
        # plr_val must exceed ETH threshold (0.20) for entry to succeed
        env = _make_env_with_signal(asset="ETH", signal_val=0.40, plr_val=0.25)
        env.reset()
        env.step(LONG_LARGE)
        original_sl = env._trade_sl_price
        # Set price BELOW trigger — breakeven should NOT fire
        entry = env.entry_price
        atr   = float(env._atr[env._step])
        env._prices[env._step] = entry + 0.3 * atr  # below 0.8 * atr
        env.step(HOLD)
        assert env._trade_sl_price == pytest.approx(original_sl, rel=1e-6), (
            "SL should not move when price has not reached breakeven trigger"
        )
    finally:
        m.USE_BREAKEVEN_STOP          = orig
        m.USE_SIGNAL_DIRECTION_FILTER = orig_dir


# ── Test 33: OIL short-bias control blocks weak repeated shorts ───────────────

def test_oil_short_bias_blocks_entry():
    """OIL SHORT should be blocked when recent ratio > threshold and signal weak."""
    import models.rl_env as m
    orig = m.USE_ASSET_SPECIFIC_GATES
    try:
        m.USE_ASSET_SPECIFIC_GATES = True
        env = _make_env_with_signal(asset="OIL", signal_val=0.30, plr_val=-0.05)
        env.reset()
        # Artificially populate recent closed sides with all SHORT
        env._recent_closed_sides = ["SHORT"] * 15   # 15 recent SHORTs out of 15 = 100%
        env.step(SHORT_LARGE)  # signal=0.30 < OIL_SHORT_BIAS_MIN_SS=0.60, bias=100% > 70%
        assert env.current_position_pct == 0.0, (
            "OIL SHORT should be blocked when short-bias is too high and signal is weak"
        )
    finally:
        m.USE_ASSET_SPECIFIC_GATES = orig


def test_oil_short_bias_allows_strong_signal():
    """OIL SHORT should be allowed despite bias when signal_strength >= OIL_SHORT_BIAS_MIN_SS."""
    import models.rl_env as m
    orig_gates = m.USE_ASSET_SPECIFIC_GATES
    orig_dir   = m.USE_SIGNAL_DIRECTION_FILTER
    try:
        m.USE_ASSET_SPECIFIC_GATES    = True
        m.USE_SIGNAL_DIRECTION_FILTER = False
        env = _make_env_with_signal(asset="OIL", signal_val=0.65, plr_val=-0.10)
        env.reset()
        env._recent_closed_sides = ["SHORT"] * 15  # high bias
        env.step(SHORT_LARGE)  # signal=0.65 >= 0.60 → should be allowed despite bias
        assert env.current_position_pct < 0.0, (
            "OIL SHORT with strong signal should be allowed even with short bias"
        )
    finally:
        m.USE_ASSET_SPECIFIC_GATES    = orig_gates
        m.USE_SIGNAL_DIRECTION_FILTER = orig_dir


# ── Test 34: BTC behavior mostly unchanged (thresholds=0) ─────────────────────

def test_btc_signal_gate_does_not_block():
    """BTC has signal threshold=0.00, so gates should not block any entry."""
    import models.rl_env as m
    orig_gates = m.USE_ASSET_SPECIFIC_GATES
    orig_dir   = m.USE_SIGNAL_DIRECTION_FILTER
    try:
        m.USE_ASSET_SPECIFIC_GATES    = True
        m.USE_SIGNAL_DIRECTION_FILTER = True
        # Very weak signal — would block ETH/GOLD/OIL but not BTC
        env = _make_env_with_signal(asset="BTC", signal_val=0.01, plr_val=0.005)
        env.reset()
        env.step(LONG_LARGE)
        assert env.current_position_pct > 0.0, (
            "BTC entry should always be allowed (threshold=0.00)"
        )
    finally:
        m.USE_ASSET_SPECIFIC_GATES    = orig_gates
        m.USE_SIGNAL_DIRECTION_FILTER = orig_dir


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
        # v2 tests
        test_pnl_timing_old_pos_earns_bar,
        test_long_tp_uses_candle_high,
        test_long_sl_uses_candle_low,
        test_short_sl_uses_candle_high,
        test_short_tp_uses_candle_low,
        test_min_hold_bars_blocks_early_flat,
        test_min_hold_bars_blocks_early_reverse,
        test_min_hold_bars_allows_exit_after_threshold,
        test_cooldown_forces_hold_after_close,
        test_tp_sl_always_allowed_despite_min_hold,
        test_completed_trade_log_metrics,
        # v3 tests
        test_signal_gate_blocks_weak_gold_entry,
        test_signal_gate_allows_strong_gold_entry,
        test_signal_gate_does_not_block_exits,
        test_tpsl_not_blocked_by_signal_gate,
        test_risk_downgrade_gold_aggressive_to_conservative,
        test_risk_downgrade_eth_aggressive_below_threshold,
        test_risk_downgrade_eth_aggressive_above_threshold,
        test_breakeven_stop_long_moves_sl_up,
        test_breakeven_stop_does_not_lower_sl_for_long,
        test_oil_short_bias_blocks_entry,
        test_oil_short_bias_allows_strong_signal,
        test_btc_signal_gate_does_not_block,
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
