"""
models/rl_env.py
Custom Gymnasium trading environment — MultiDiscrete([4,3,3]) action space.

Action dimensions:
    direction [4]: 0=HOLD  1=FLAT  2=LONG  3=SHORT
    size      [3]: 0=SMALL(0.25)  1=MEDIUM(0.50)  2=LARGE(1.00)
    risk      [3]: 0=CONSERVATIVE  1=BALANCED  2=AGGRESSIVE

Risk profiles:
    CONSERVATIVE : leverage=1.0  sl_mult=1.0  tp_mult=1.5
    BALANCED     : leverage=1.5  sl_mult=1.5  tp_mult=2.5
    AGGRESSIVE   : leverage=2.0  sl_mult=2.0  tp_mult=4.0

For HOLD/FLAT the size and risk dims are ignored (logged only).
TP/SL prices are fixed at trade entry from entry_price ± mult*ATR.

Outputs (written by save_trade_log_csv / save_trade_history_csv):
    data/dashboard/latest_rl_trades.csv          — one row per env step
    data/dashboard/latest_rl_trade_history.csv   — one row per completed trade
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import psycopg2.extras
from loguru import logger
from pathlib import Path

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        _GYM = True
        logger.warning("Using legacy 'gym'; consider upgrading to 'gymnasium'.")
    except ImportError:
        _GYM = False
        logger.error("Neither 'gymnasium' nor 'gym' found: pip install gymnasium")
        class gym:           # type: ignore
            class Env: pass
        class spaces:        # type: ignore
            @staticmethod
            def Box(*a, **kw): return None
            @staticmethod
            def Discrete(*a, **kw): return None
            @staticmethod
            def MultiDiscrete(*a, **kw): return None

try:
    import joblib
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_CASH     = 10_000.0
TRANSACTION_COST = 0.001
MAX_DRAWDOWN     = 0.15

DEFAULT_MAX_HOLDING_BARS      = 48
DEFAULT_SIGNAL_THRESHOLD      = 0.001
DEFAULT_OPPORTUNITY_THRESHOLD = 0.002
DEFAULT_MISSED_OPP_COEF       = 0.05

# Kept for backward-compat imports (per-trade TP/SL comes from risk profile)
DEFAULT_SL_MULT = 1.5
DEFAULT_TP_MULT = 2.5

DRAWDOWN_PENALTY_COEF   = 2.0
WRONG_SIDE_PENALTY_COEF = 0.05
OVERTRADE_PENALTY_COEF  = 0.02
TP_BONUS_VALUE          = 0.002
SL_PENALTY_VALUE        = 0.005

SAVED_DIR = Path(__file__).parent / "saved"

# ── MultiDiscrete action decoders ─────────────────────────────────────────────

DIRECTION_NAMES = {0: "HOLD", 1: "FLAT", 2: "LONG", 3: "SHORT"}
SIZE_NAMES      = {0: "SMALL", 1: "MEDIUM", 2: "LARGE"}
RISK_NAMES      = {0: "CONSERVATIVE", 1: "BALANCED", 2: "AGGRESSIVE"}

SIZE_FRACTIONS  = {0: 0.25, 1: 0.50, 2: 1.00}

RISK_PROFILES = {
    0: {"name": "CONSERVATIVE", "leverage": 1.0, "sl_mult": 1.0, "tp_mult": 1.5},
    1: {"name": "BALANCED",     "leverage": 1.5, "sl_mult": 1.5, "tp_mult": 2.5},
    2: {"name": "AGGRESSIVE",   "leverage": 2.0, "sl_mult": 2.0, "tp_mult": 4.0},
}

# Backward-compat legacy map — tests reference ACTION_MAP[k]["name"] / ["target"]
ACTION_MAP = {
    0: {"name": "HOLD",         "target": None,  "action": [0, 1, 1]},
    1: {"name": "FLAT",         "target":  0.00, "action": [1, 1, 1]},
    2: {"name": "LONG_SMALL",   "target":  0.25, "action": [2, 0, 1]},
    3: {"name": "LONG_MEDIUM",  "target":  0.50, "action": [2, 1, 1]},
    4: {"name": "LONG_LARGE",   "target":  1.00, "action": [2, 2, 1]},
    5: {"name": "SHORT_SMALL",  "target": -0.25, "action": [3, 0, 1]},
    6: {"name": "SHORT_MEDIUM", "target": -0.50, "action": [3, 1, 1]},
    7: {"name": "SHORT_LARGE",  "target": -1.00, "action": [3, 2, 1]},
}

# ── Market feature columns ────────────────────────────────────────────────────

MARKET_FEATURE_COLS = [
    "close",
    "ret_1",  "ret_4",  "ret_8",  "ret_16",
    "mean_4", "mean_8", "mean_16",
    "std_4",  "std_8",  "std_16",
    "mom_4",  "mom_16",
    "vol_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct", "atr_14",
    "btc_ret_lag_1", "btc_ret_lag_4", "oil_vol_lag_1",
    "dxy", "vix", "spy",
]

TRANSFORMER_FEAT_COLS   = ["predicted_log_return", "signal_strength", "derived_direction"]
_TRANSFORMER_SIGNAL_DIM = len(TRANSFORMER_FEAT_COLS)
FEATURE_COLS            = MARKET_FEATURE_COLS + TRANSFORMER_FEAT_COLS
_EXTRA_DIM              = 7
STATE_DIM               = len(FEATURE_COLS) + _EXTRA_DIM


# ── Load Transformer predictions ──────────────────────────────────────────────

def _load_transformer_preds(
    asset: str,
    prediction_horizon: int = 4,
    target_mode: str = "vol_norm",
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=TRANSFORMER_FEAT_COLS)
    try:
        from db.connection import get_conn
        sql = """
            SELECT ts,
                   COALESCE(predicted_log_return, 0.0) AS predicted_log_return,
                   COALESCE(signal_strength,      0.0) AS signal_strength,
                   COALESCE(predicted_direction,  0)   AS derived_direction
            FROM   transformer_predictions
            WHERE  asset = %s
              AND  prediction_horizon = %s
              AND  target_mode = %s
            ORDER  BY ts
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (asset, int(prediction_horizon), target_mode))
                rows = cur.fetchall()

        if not rows:
            raise ValueError(
                f"No transformer_predictions for {asset} "
                f"horizon={prediction_horizon} target_mode={target_mode}."
            )

        preds = pd.DataFrame(rows)
        preds["ts"] = pd.to_datetime(preds["ts"], utc=True)
        preds = preds.sort_values("ts").drop_duplicates("ts", keep="last")
        for col in TRANSFORMER_FEAT_COLS:
            preds[col] = pd.to_numeric(preds[col], errors="coerce").fillna(0.0)
        preds = preds.set_index("ts")[TRANSFORMER_FEAT_COLS]

        assert not preds.index.duplicated().any()
        logger.info(
            f"[RL/{asset}] Loaded {len(preds)} transformer predictions "
            f"(horizon={prediction_horizon} target_mode={target_mode})"
        )
        return preds

    except (ValueError, AssertionError):
        raise
    except Exception as e:
        logger.warning(f"Could not load transformer_predictions for {asset}: {e}")
        return empty


# ── Environment ───────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Single-asset futures-style environment with MultiDiscrete([4,3,3]) actions.

    direction: 0=HOLD 1=FLAT 2=LONG 3=SHORT
    size:      0=SMALL(0.25) 1=MEDIUM(0.50) 2=LARGE(1.00)
    risk:      0=CONSERVATIVE 1=BALANCED 2=AGGRESSIVE

    Per-trade leverage, TP/SL prices, and risk-reward ratio are
    determined by the risk dim at entry.

    Trade lifecycle:
        flat → open (LONG/SHORT)
             → resize (same direction, different size) → AGENT_RESIZE close + reopen
             → reverse (opposite direction)            → AGENT_REVERSE close + reopen
             → close (FLAT / TP / SL / MAX_HOLD / END_OF_EPISODE)

    Outputs written by save_trade_log_csv() and save_trade_history_csv().
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 df: pd.DataFrame,
                 asset: str = None,
                 feature_cols: list = None,
                 initial_cash: float = INITIAL_CASH,
                 transaction_cost: float = TRANSACTION_COST,
                 scaler=None,
                 market_feature_cols: list = None,
                 prediction_horizon: int = 4,
                 target_mode: str = "vol_norm",
                 leverage: float = 1.0,          # kept for compat; per-trade lev from risk
                 sl_mult: float = DEFAULT_SL_MULT,  # kept for compat
                 tp_mult: float = DEFAULT_TP_MULT,
                 max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
                 signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
                 opportunity_threshold: float = DEFAULT_OPPORTUNITY_THRESHOLD,
                 missed_opp_coef: float = DEFAULT_MISSED_OPP_COEF):
        super().__init__()

        self._mkt_cols              = market_feature_cols or MARKET_FEATURE_COLS
        self.feature_cols           = feature_cols or (self._mkt_cols + TRANSFORMER_FEAT_COLS)
        self.initial_capital        = initial_cash
        self.initial_cash           = initial_cash
        self.transaction_cost       = transaction_cost
        self._scaler                = scaler
        self._asset                 = asset
        self._prediction_horizon    = prediction_horizon
        self._target_mode           = target_mode
        self._leverage              = float(leverage)
        self._max_holding_bars      = int(max_holding_bars)
        self._signal_threshold      = float(signal_threshold)
        self._opportunity_threshold = float(opportunity_threshold)
        self._missed_opp_coef       = float(missed_opp_coef)

        if self._scaler is None and asset is not None and _JOBLIB:
            scaler_path = SAVED_DIR / f"feature_scaler_{asset.lower()}.pkl"
            if scaler_path.exists():
                try:
                    self._scaler = joblib.load(scaler_path)
                    logger.debug(f"{asset}: loaded scaler from {scaler_path}")
                except Exception as e:
                    logger.warning(f"{asset}: could not load scaler — {e}")

        df = df.copy()
        dup_before = int(df.index.duplicated().sum())
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        missing_trans = [c for c in TRANSFORMER_FEAT_COLS if c not in df.columns]
        preds = pd.DataFrame(columns=TRANSFORMER_FEAT_COLS)
        if missing_trans:
            if asset is not None:
                preds = _load_transformer_preds(asset, prediction_horizon, target_mode)
                for col in TRANSFORMER_FEAT_COLS:
                    df[col] = preds[col].reindex(df.index).fillna(0.0)
            else:
                for col in TRANSFORMER_FEAT_COLS:
                    df[col] = 0.0

        plr = df.get("predicted_log_return", pd.Series(np.zeros(len(df)))).values
        ss  = df.get("signal_strength",      pd.Series(np.zeros(len(df)))).values
        pred_ts_min = preds.index[0]  if not preds.empty else "N/A"
        pred_ts_max = preds.index[-1] if not preds.empty else "N/A"
        logger.info(
            f"[{asset or 'ENV'}] RL data diagnostics:\n"
            f"  asset={asset}  horizon={prediction_horizon}  target_mode={target_mode}\n"
            f"  market rows={len(df)}  prediction rows={len(preds)}\n"
            f"  market ts=[{df.index.min()}...{df.index.max()}]\n"
            f"  prediction ts=[{pred_ts_min}...{pred_ts_max}]\n"
            f"  dup market ts (before dedup)={dup_before}\n"
            f"  predicted_log_return: mean={float(plr.mean()):.6f}  std={float(plr.std()):.6f}\n"
            f"  signal_strength:      mean={float(ss.mean()):.6f}  std={float(ss.std()):.6f}"
        )

        cols_needed = list(dict.fromkeys(self.feature_cols + ["close"]))
        self.df = df[[c for c in cols_needed if c in df.columns]].copy()
        # Preserve OHLC passthrough columns if present
        for _ohlc in ("open", "high", "low"):
            if _ohlc in df.columns and _ohlc not in self.df.columns:
                self.df[_ohlc] = df[_ohlc]
        self.df = self.df.dropna(subset=["close"])

        market_present = [c for c in self._mkt_cols if c in self.df.columns]
        self.df[market_present] = self.df[market_present].fillna(
            self.df[market_present].median()
        )
        trans_present = [c for c in TRANSFORMER_FEAT_COLS if c in self.df.columns]
        self.df[trans_present] = self.df[trans_present].fillna(0.0)

        if self.df.empty:
            raise ValueError("TradingEnv: DataFrame is empty after dropping NaN rows.")

        self._prices     = self.df["close"].values.astype(np.float32)
        self._timestamps = self.df.index
        self.n_steps     = len(self.df)

        # OHLC arrays — loaded from df if available, otherwise NaN with loud warning
# OHLC arrays for dashboard candlesticks.
# Raw OHLC is used where available. Missing bars are reconstructed from the
# RL step-level close/price series so dashboard candles are always available.
        _ohlc_cols = ("open", "high", "low")
        self._has_ohlc = all(c in self.df.columns for c in _ohlc_cols)

        close_s = pd.Series(self._prices.astype(float), index=self.df.index)

        if self._has_ohlc:
            open_s = pd.to_numeric(self.df["open"], errors="coerce").astype(float)
            high_s = pd.to_numeric(self.df["high"], errors="coerce").astype(float)
            low_s  = pd.to_numeric(self.df["low"],  errors="coerce").astype(float)
        else:
            _missing = [c for c in _ohlc_cols if c not in self.df.columns]
            logger.warning(
                f"[{asset or 'ENV'}] OHLC columns missing from DataFrame: {_missing}. "
                "Reconstructing step-level OHLC from close prices for dashboard visualization."
            )
            open_s = pd.Series(np.nan, index=self.df.index, dtype=float)
            high_s = pd.Series(np.nan, index=self.df.index, dtype=float)
            low_s  = pd.Series(np.nan, index=self.df.index, dtype=float)

        raw_mask = open_s.notna() & high_s.notna() & low_s.notna()

        # Reconstruct missing step-level candles:
        # open = previous close; high/low = max/min(open, close).
        recon_open = close_s.shift(1).fillna(close_s)
        open_s = open_s.where(open_s.notna(), recon_open)
        high_s = high_s.where(high_s.notna(), pd.concat([open_s, close_s], axis=1).max(axis=1))
        low_s  = low_s.where(low_s.notna(),  pd.concat([open_s, close_s], axis=1).min(axis=1))

        self._opens = open_s.values.astype(np.float32)
        self._highs = high_s.values.astype(np.float32)
        self._lows  = low_s.values.astype(np.float32)

        # Track which candles are true raw OHLC vs reconstructed step candles.
        self._ohlc_source = np.where(raw_mask.values, "raw", "reconstructed")
        recon_pct = float((~raw_mask).mean() * 100.0)

        if recon_pct > 0:
            logger.warning(
                f"[{asset or 'ENV'}] Reconstructed {recon_pct:.1f}% of OHLC candles "
                "from step-level close prices. These candles are for RL dashboard "
                "visualization, not raw exchange OHLC."
            )
        else:
            logger.info(f"[{asset or 'ENV'}] All OHLC candles loaded from raw_price_bars.")

        self._hours = np.zeros(self.n_steps, dtype=np.float32)
        for i, idx in enumerate(self.df.index):
            if hasattr(idx, "hour"):
                self._hours[i] = float(idx.hour)

        raw_market = self.df[
            [c for c in self._mkt_cols if c in self.df.columns]
        ].values.astype(np.float32)

        if self._scaler is not None:
            n_market = raw_market.shape[1]
            if (hasattr(self._scaler, "n_features_in_") and
                    self._scaler.n_features_in_ != n_market):
                logger.error(
                    f"SCALER MISMATCH: scaler expects {self._scaler.n_features_in_} "
                    f"features, env has {n_market}. Using fallback normalisation."
                )
                normed_market = self._normalise(raw_market)
            else:
                normed_market = np.nan_to_num(self._scaler.transform(raw_market), nan=0.0)
        else:
            normed_market = self._normalise(raw_market)

        self._transformer_raw = self.df[
            [c for c in TRANSFORMER_FEAT_COLS if c in self.df.columns]
        ].values.astype(np.float32)

        normed_trans = self._normalise_transformer(self._transformer_raw)
        self._features = np.concatenate([normed_market, normed_trans], axis=1)

        # ATR array (used to fix TP/SL prices at entry)
        if "atr_14" in self.df.columns:
            raw_atr = self.df["atr_14"].values.astype(np.float64)
            raw_atr = np.where(raw_atr > 0, raw_atr, np.nan)
            self._atr = pd.Series(raw_atr).ffill().fillna(1.0).values.astype(np.float32)
        else:
            log_rets = np.zeros(self.n_steps, dtype=np.float64)
            prices_f = self._prices.astype(np.float64)
            log_rets[1:] = np.where(
                prices_f[:-1] > 0,
                np.log(np.maximum(prices_f[1:], 1e-10) / np.maximum(prices_f[:-1], 1e-10)),
                0.0,
            )
            roll_vol = pd.Series(log_rets).rolling(14, min_periods=1).std().fillna(0.001).values
            self._atr = (prices_f * roll_vol).astype(np.float32)

        obs_dim = self._features.shape[1] + _EXTRA_DIM

        logger.info(
            f"[{asset or 'ENV'}] TradingEnv started (MultiDiscrete([4,3,3])):\n"
            f"  asset={asset}  obs_dim={obs_dim}  n_steps={self.n_steps}\n"
            f"  direction: 0=HOLD 1=FLAT 2=LONG 3=SHORT\n"
            f"  size:      0=SMALL(0.25) 1=MEDIUM(0.50) 2=LARGE(1.00)\n"
            f"  risk:      0=CONSERVATIVE(lev=1.0 sl=1.0 tp=1.5)\n"
            f"             1=BALANCED    (lev=1.5 sl=1.5 tp=2.5)\n"
            f"             2=AGGRESSIVE  (lev=2.0 sl=2.0 tp=4.0)\n"
            f"  fee_rate={transaction_cost:.4f}  max_holding_bars={max_holding_bars}\n"
            f"  signal_threshold={signal_threshold}  "
            f"opportunity_threshold={opportunity_threshold}\n"
            f"  ohlc_available={self._has_ohlc}"
        )

        self.action_space = spaces.MultiDiscrete([4, 3, 3])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._init_episode_state()

    def _init_episode_state(self):
        self.portfolio_value         = self.initial_capital
        self.current_position_pct    = 0.0
        self.in_position             = False
        self.trade_side              = 0
        self.entry_price             = None
        self.entry_time              = None
        self.steps_in_position       = 0
        self.unrealized_pnl          = 0.0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion   = 0.0
        # Per-trade risk state
        self._trade_leverage         = 1.0
        self._trade_sl_price         = None
        self._trade_tp_price         = None
        self._trade_risk_profile     = ""
        self._trade_rr_ratio         = 0.0
        # Per-trade accumulators (reset at open, emitted at close)
        self._trade_id               = 0
        self._tacc_high              = -np.inf
        self._tacc_low               =  np.inf
        self._tacc_tc                = 0.0
        self._tacc_reward            = 0.0
        self._tacc_rpnl              = 0.0
        self._tacc_rtc               = 0.0
        self._tacc_rdd               = 0.0
        self._tacc_rot               = 0.0
        self._tacc_rmiss             = 0.0
        self._tacc_rws               = 0.0
        self._tacc_rtp               = 0.0
        self._tacc_rsl               = 0.0
        # Entry-time snapshots
        self._trade_entry_plr        = 0.0
        self._trade_entry_ss         = 0.0
        self._trade_entry_dir_act    = ""
        self._trade_entry_size       = ""
        self._trade_entry_risk       = ""
        self._trade_entry_pos_frac   = 0.0
        self._trade_pv_before        = 0.0
        # Episode state
        self.peak_portfolio          = self.initial_capital
        self.current_drawdown        = 0.0
        self.max_drawdown_breached   = False
        self.episode_returns         = []
        self.rolling_vol_20          = 0.0
        self._log_returns_buf        = []
        self.metrics                 = self._init_metrics()
        self.closed_trades           = []
        self._hold_bars_list         = []
        self.trade_log               = []
        self.completed_trade_log     = []
        self._step                   = 0
        self.done                    = False
        self._tacc_portfolio_delta = 0.0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._init_episode_state()
        return self._obs(), {}

    def step(self, action):
        # ── Decode and validate MultiDiscrete action ──────────────────────────
        action   = np.asarray(action, dtype=int).ravel()
        if len(action) != 3:
            raise ValueError(
                f"action must have 3 elements [direction, size, risk], got {len(action)}"
            )
        dir_idx  = int(action[0])
        size_idx = int(action[1])
        risk_idx = int(action[2])
        if dir_idx not in DIRECTION_NAMES:
            raise ValueError(f"direction index {dir_idx} out of range [0,3]")
        if size_idx not in SIZE_NAMES:
            raise ValueError(f"size index {size_idx} out of range [0,2]")
        if risk_idx not in RISK_NAMES:
            raise ValueError(f"risk index {risk_idx} out of range [0,2]")

        idx        = self._step
        price_t    = float(self._prices[idx])
        price_prev = float(self._prices[idx - 1]) if idx > 0 else price_t
        ts_utc     = self._timestamps[idx]
        prev_pv    = self.portfolio_value
        open_t     = float(self._opens[idx])
        high_t     = float(self._highs[idx])
        low_t      = float(self._lows[idx])

        # Rolling vol
        if idx > 0 and price_prev > 0 and price_t > 0:
            self._log_returns_buf.append(float(np.log(price_t / price_prev)))
            if len(self._log_returns_buf) > 20:
                self._log_returns_buf.pop(0)
            if len(self._log_returns_buf) >= 2:
                self.rolling_vol_20 = float(np.std(self._log_returns_buf))

        # Transformer signal
        trans        = self._transformer_raw[idx]
        predicted_lr = float(trans[0])
        signal_str   = float(trans[1])
        derived_dir  = int(trans[2])

        # ATR for TP/SL fixing at entry
        atr_t = float(self._atr[idx])
        if atr_t <= 0 or np.isnan(atr_t):
            atr_t = max(self.rolling_vol_20 * price_t, 1e-8)

        # ── TP/SL check using pre-computed entry prices ───────────────────────
        tp_triggered       = False
        sl_triggered       = False
        max_hold_triggered = False

        if self.in_position and self._trade_sl_price is not None:
            if self.trade_side == 1:
                if price_t <= self._trade_sl_price:
                    dir_idx, sl_triggered = 1, True
                    self.metrics["n_stop_losses_hit"] += 1
                elif price_t >= self._trade_tp_price:
                    dir_idx, tp_triggered = 1, True
                    self.metrics["n_take_profits_hit"] += 1
            elif self.trade_side == -1:
                if price_t >= self._trade_sl_price:
                    dir_idx, sl_triggered = 1, True
                    self.metrics["n_stop_losses_hit"] += 1
                elif price_t <= self._trade_tp_price:
                    dir_idx, tp_triggered = 1, True
                    self.metrics["n_take_profits_hit"] += 1

            if (not sl_triggered and not tp_triggered
                    and self.steps_in_position >= self._max_holding_bars):
                dir_idx, max_hold_triggered = 1, True
                self.metrics["n_max_hold_exits"] += 1

        direction  = DIRECTION_NAMES[dir_idx]
        size_label = SIZE_NAMES[size_idx]
        risk_prof  = RISK_PROFILES[risk_idx]
        risk_name  = risk_prof["name"]

        old_position_pct = self.current_position_pct
        old_leverage     = self._trade_leverage
        clipped          = False
        clip_reason      = ""

        # ── Map to desired position fraction ──────────────────────────────────
        if direction == "HOLD":
            desired_pct      = old_position_pct
            executed_pct     = old_position_pct
            applied_leverage = old_leverage
            applied_sl_mult  = 0.0
            applied_tp_mult  = 0.0
            applied_risk     = self._trade_risk_profile
            action_name      = "HOLD"
        elif direction == "FLAT":
            desired_pct      = 0.0
            executed_pct     = 0.0
            applied_leverage = old_leverage
            applied_sl_mult  = 0.0
            applied_tp_mult  = 0.0
            applied_risk     = ""
            action_name      = "FLAT"
        else:
            sign             = 1.0 if direction == "LONG" else -1.0
            desired_pct      = sign * SIZE_FRACTIONS[size_idx]
            applied_leverage = risk_prof["leverage"]
            applied_sl_mult  = risk_prof["sl_mult"]
            applied_tp_mult  = risk_prof["tp_mult"]
            applied_risk     = risk_name
            action_name      = f"{direction}_{size_label}_{risk_name}"
            if self.portfolio_value < 0.05 * self.initial_capital:
                executed_pct = 0.0
                clipped      = True
                clip_reason  = "insufficient_capital"
                self.metrics["n_clipped_actions"] += 1
            else:
                executed_pct = desired_pct

        new_position_pct = float(executed_pct)

        # ── Transition detection ──────────────────────────────────────────────
        was_flat    = (old_position_pct == 0.0)
        is_flat_now = (new_position_pct == 0.0)
        dir_changed = (
            np.sign(old_position_pct) != np.sign(new_position_pct)
            and old_position_pct != 0.0 and new_position_pct != 0.0
        )
        # Resize: same direction, different magnitude (Option A: close + reopen)
        resized = (
            not was_flat and not is_flat_now and not dir_changed
            and new_position_pct != old_position_pct
        )

        # ── exit_reason for trades that close this step ───────────────────────
        realized_pnl = 0.0
        exit_reason  = ""
        closing_now  = not was_flat and (is_flat_now or dir_changed or resized)

        if closing_now:
            if tp_triggered:
                exit_reason = "TP"
            elif sl_triggered:
                exit_reason = "SL"
            elif max_hold_triggered:
                exit_reason = "MAX_HOLD"
            elif dir_changed:
                exit_reason = "AGENT_REVERSE"
            elif resized:
                exit_reason = "AGENT_RESIZE"
            else:
                exit_reason = "AGENT_CLOSE"

            if self.entry_price and self.entry_price > 0:
                ps           = 1.0 if old_position_pct > 0 else -1.0
                realized_pnl = ps * (price_t - self.entry_price) / self.entry_price
                self.closed_trades.append(realized_pnl)
                self._hold_bars_list.append(self.steps_in_position)
            self.metrics["n_trades"] += 1

        # ── P&L with per-trade leverage ───────────────────────────────────────
        price_ret         = (price_t - price_prev) / price_prev if price_prev > 0 else 0.0
        effective_pos     = new_position_pct * applied_leverage
        old_effective_pos = old_position_pct * old_leverage
        pnl_t             = effective_pos * price_ret
        eff_turnover      = abs(effective_pos - old_effective_pos)
        tc_t              = eff_turnover * self.transaction_cost

        pnl_dollars = pnl_t * self.initial_capital
        tc_dollars  = eff_turnover * self.portfolio_value * self.transaction_cost

        raw_pv = self.portfolio_value + pnl_dollars - tc_dollars
        if applied_leverage > 1.0 and raw_pv <= 0:
            logger.warning(
                f"[RL/{self._asset}] Liquidation at step {idx} "
                f"(leverage={applied_leverage:.1f})"
            )
            self.portfolio_value = 0.0
            self.done = True
        else:
            self.portfolio_value = max(0.0, raw_pv)

        if self.portfolio_value > self.peak_portfolio:
            self.peak_portfolio = self.portfolio_value
        self.current_drawdown = (
            (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
            if self.peak_portfolio > 0 else 0.0
        )
        max_drawdown_triggered = self.current_drawdown > MAX_DRAWDOWN

        # ── Reward components ─────────────────────────────────────────────────
        reward_pnl                        = float(pnl_t)
        reward_transaction_cost           = float(tc_t)
        reward_drawdown_penalty           = float(
            DRAWDOWN_PENALTY_COEF * max(0.0, self.current_drawdown - 0.05)
        )
        reward_overtrading_penalty        = float(
            OVERTRADE_PENALTY_COEF * eff_turnover if eff_turnover > 0 else 0.0
        )
        reward_missed_opportunity_penalty = 0.0
        reward_wrong_side_penalty         = 0.0
        reward_tp_bonus                   = float(TP_BONUS_VALUE  if tp_triggered  else 0.0)
        reward_sl_penalty                 = float(SL_PENALTY_VALUE if sl_triggered else 0.0)

        if old_position_pct == 0.0 and direction in ("HOLD", "FLAT"):
            if (abs(predicted_lr) > self._signal_threshold
                    and abs(price_ret) > self._opportunity_threshold):
                reward_missed_opportunity_penalty = self._missed_opp_coef * abs(price_ret)

        old_side = float(np.sign(old_position_pct))
        if old_side != 0:
            if old_side > 0 and price_ret < -self._opportunity_threshold:
                reward_wrong_side_penalty = WRONG_SIDE_PENALTY_COEF * abs(price_ret)
            elif old_side < 0 and price_ret > self._opportunity_threshold:
                reward_wrong_side_penalty = WRONG_SIDE_PENALTY_COEF * abs(price_ret)

        reward = (
            reward_pnl
            - reward_transaction_cost
            - reward_drawdown_penalty
            - reward_overtrading_penalty
            - reward_missed_opportunity_penalty
            - reward_wrong_side_penalty
            + reward_tp_bonus
            - reward_sl_penalty
        )

        # ── Accumulate per-trade data ─────────────────────────────────────────
        # Accumulate whenever any position touched this step (open, hold, or close)
        if old_position_pct != 0.0 or new_position_pct != 0.0:
            self._tacc_tc    += tc_dollars
            self._tacc_portfolio_delta += float(self.portfolio_value - prev_pv)
            self._tacc_reward += reward
            self._tacc_rpnl   += reward_pnl
            self._tacc_rtc    += reward_transaction_cost
            self._tacc_rdd    += reward_drawdown_penalty
            self._tacc_rot    += reward_overtrading_penalty
            self._tacc_rmiss  += reward_missed_opportunity_penalty
            self._tacc_rws    += reward_wrong_side_penalty
            self._tacc_rtp    += reward_tp_bonus
            self._tacc_rsl    += reward_sl_penalty
            # OHLC high/low tracking during trade (NaN-safe)
            if not np.isnan(high_t):
                self._tacc_high = max(self._tacc_high, high_t)
            else:
                self._tacc_high = max(self._tacc_high, price_t)
            if not np.isnan(low_t):
                self._tacc_low = min(self._tacc_low, low_t)
            else:
                self._tacc_low = min(self._tacc_low, price_t)

        # ── Update unrealized PnL and MFE/MAE before emit ────────────────────
        if self.in_position and self.entry_price and self.entry_price > 0:
            upnl = float(self.trade_side) * (price_t - self.entry_price) / self.entry_price
            self.unrealized_pnl = upnl
            if upnl > self.max_favorable_excursion:
                self.max_favorable_excursion = upnl
            if upnl < self.max_adverse_excursion:
                self.max_adverse_excursion = upnl

        # ── Emit completed trade + lifecycle transitions ──────────────────────
        def _reset_trade_accumulators():
            self._tacc_high  = -np.inf
            self._tacc_low   =  np.inf
            self._tacc_tc    = 0.0
            self._tacc_reward = 0.0
            self._tacc_rpnl   = 0.0
            self._tacc_rtc    = 0.0
            self._tacc_rdd    = 0.0
            self._tacc_rot    = 0.0
            self._tacc_rmiss  = 0.0
            self._tacc_rws    = 0.0
            self._tacc_rtp    = 0.0
            self._tacc_rsl    = 0.0
            self._tacc_portfolio_delta = 0.0

        def _reset_trade_state():
            self.in_position             = False
            self.trade_side              = 0
            self.entry_price             = None
            self.entry_time              = None
            self.steps_in_position       = 0
            self.max_favorable_excursion = 0.0
            self.max_adverse_excursion   = 0.0
            self._trade_leverage         = 1.0
            self._trade_sl_price         = None
            self._trade_tp_price         = None
            self._trade_risk_profile     = ""
            self._trade_rr_ratio         = 0.0
            _reset_trade_accumulators()

        def _open_trade_state(pos_pct):
            self.in_position             = True
            self.trade_side              = 1 if pos_pct > 0 else -1
            self.entry_price             = price_t
            self.entry_time              = ts_utc
            self.steps_in_position       = 0
            self.max_favorable_excursion = 0.0
            self.max_adverse_excursion   = 0.0
            self._trade_leverage         = applied_leverage
            self._trade_risk_profile     = applied_risk
            if pos_pct > 0:
                self._trade_sl_price = price_t - applied_sl_mult * atr_t
                self._trade_tp_price = price_t + applied_tp_mult * atr_t
            else:
                self._trade_sl_price = price_t + applied_sl_mult * atr_t
                self._trade_tp_price = price_t - applied_tp_mult * atr_t
            self._trade_rr_ratio = (
                applied_tp_mult / applied_sl_mult if applied_sl_mult > 0 else 0.0
            )
            self._trade_id            += 1
            self._trade_entry_plr      = predicted_lr
            self._trade_entry_ss       = signal_str
            self._trade_entry_dir_act  = direction
            self._trade_entry_size     = size_label
            self._trade_entry_risk     = applied_risk
            self._trade_entry_pos_frac = abs(pos_pct)
            self._trade_pv_before = float(self.portfolio_value - self._tacc_portfolio_delta)

        def _emit_completed_trade(er: str):
            """Append one row to completed_trade_log for the trade that just closed."""
            if not self.entry_price or self.entry_price <= 0:
                return
            ep       = float(self.entry_price)
            xp       = float(price_t)
            side     = self.trade_side
            pf       = self._trade_entry_pos_frac
            lev      = self._trade_leverage
            # Price-move estimate is kept only for explanation/debugging.
            # Main trade PnL must reconcile with portfolio accounting.
            price_move_pnl_estimate = side * (xp - ep) / ep * pf * lev * self.initial_capital

            notional = pf * lev * self.initial_capital

            portfolio_before = float(self._trade_pv_before)
            net_pnl = float(self._tacc_portfolio_delta)
            portfolio_after = portfolio_before + net_pnl

            # gross_pnl here means portfolio-attributed PnL before transaction costs.
            gross_pnl = net_pnl + float(self._tacc_tc)

            # return_pct = net_pnl / notional_value, leveraged notional basis.
            ret_pct = net_pnl / notional if notional > 0 else 0.0
            high_dt   = float(self._tacc_high) if np.isfinite(self._tacc_high) else xp
            low_dt    = float(self._tacc_low)  if np.isfinite(self._tacc_low)  else xp
            self.completed_trade_log.append({
                "trade_id":                               self._trade_id,
                "asset":                                  self._asset or "",
                "side":                                   "LONG" if side == 1 else "SHORT",
                "entry_time":                             str(self.entry_time),
                "exit_time":                              str(ts_utc),
                "holding_bars":                           self.steps_in_position,
                "entry_price":                            ep,
                "exit_price":                             xp,
                "high_during_trade":                      high_dt,
                "low_during_trade":                       low_dt,
                "position_fraction":                      pf,
                "leverage_used":                          lev,
                "effective_position":                     pf * lev,
                "notional_value":                         notional,
                "margin_used":                            pf * self.initial_capital,
                "tp_price":                               float(self._trade_tp_price) if self._trade_tp_price is not None else None,
                "sl_price":                               float(self._trade_sl_price) if self._trade_sl_price is not None else None,
                "risk_reward_ratio":                      float(self._trade_rr_ratio),
                "exit_reason":                            er,
                "gross_pnl":                                 gross_pnl,
                "price_move_pnl_estimate":                  price_move_pnl_estimate,
                "transaction_cost_total":                   float(self._tacc_tc),
                "net_pnl":                                  net_pnl,
                "return_pct":                                ret_pct,
                "reward_total":                           float(self._tacc_reward),
                "reward_pnl_total":                       float(self._tacc_rpnl),
                "reward_transaction_cost_total":          float(self._tacc_rtc),
                "reward_drawdown_penalty_total":          float(self._tacc_rdd),
                "reward_overtrading_penalty_total":       float(self._tacc_rot),
                "reward_missed_opportunity_penalty_total": float(self._tacc_rmiss),
                "reward_wrong_side_penalty_total":        float(self._tacc_rws),
                "reward_tp_bonus_total":                  float(self._tacc_rtp),
                "reward_sl_penalty_total":                float(self._tacc_rsl),
                "portfolio_value_before":                 portfolio_before,
                "portfolio_value_after":                  portfolio_after,
                "max_favorable_excursion":                float(self.max_favorable_excursion),
                "max_adverse_excursion":                  float(self.max_adverse_excursion),
                "predicted_log_return_at_entry":          float(self._trade_entry_plr),
                "signal_strength_at_entry":               float(self._trade_entry_ss),
                "direction_action_at_entry":              self._trade_entry_dir_act,
                "size_label_at_entry":                    self._trade_entry_size,
                "risk_profile_at_entry":                  self._trade_entry_risk,
            })

        # Execute lifecycle transitions
        if closing_now:
            if is_flat_now and not dir_changed and not resized:
                _emit_completed_trade(exit_reason)
                _reset_trade_state()
            elif dir_changed:
                _emit_completed_trade("AGENT_REVERSE")
                _reset_trade_state()
                _open_trade_state(new_position_pct)
            elif resized:
                _emit_completed_trade("AGENT_RESIZE")
                _reset_trade_state()
                _open_trade_state(new_position_pct)
        elif was_flat and not is_flat_now:
            _open_trade_state(new_position_pct)

        self.current_position_pct = new_position_pct

        # Update unrealized PnL and MFE/MAE for new trade state
        if self.in_position and self.entry_price and self.entry_price > 0:
            upnl = float(self.trade_side) * (price_t - self.entry_price) / self.entry_price
            self.unrealized_pnl = upnl
            if upnl > self.max_favorable_excursion:
                self.max_favorable_excursion = upnl
            if upnl < self.max_adverse_excursion:
                self.max_adverse_excursion = upnl
        else:
            self.unrealized_pnl = 0.0

        if self.current_position_pct != 0.0:
            self.steps_in_position += 1

        # ── Metrics ───────────────────────────────────────────────────────────
        self.metrics["total_pnl"] = (
            (self.portfolio_value - self.initial_capital) / self.initial_capital
        )
        self.metrics["total_transaction_cost"] += float(tc_t)
        self.metrics["sum_position"] = (
            self.metrics.get("sum_position", 0.0) + abs(self.current_position_pct)
        )

        self.episode_returns.append(reward_pnl - reward_transaction_cost)
        self.metrics["total_reward"]                       += reward
        self.metrics["total_reward_pnl"]                   += reward_pnl
        self.metrics["total_transaction_cost_penalty"]     += reward_transaction_cost
        self.metrics["total_drawdown_penalty"]             += reward_drawdown_penalty
        self.metrics["total_overtrading_penalty"]          += reward_overtrading_penalty
        self.metrics["total_missed_opportunity_penalty"]   += reward_missed_opportunity_penalty
        self.metrics["total_wrong_side_penalty"]           += reward_wrong_side_penalty
        self.metrics["total_tp_bonus"]                     += reward_tp_bonus
        self.metrics["total_sl_penalty"]                   += reward_sl_penalty
        self.metrics["n_steps"]                            += 1
        self.metrics["direction_counts"][dir_idx]          += 1

        if max_drawdown_triggered:
            self.max_drawdown_breached = True
            self.done = True

        # Portfolio accounting helpers for step log
        cash = max(
            0.0,
            self.portfolio_value - abs(self.current_position_pct) * self.initial_capital,
        )
        portfolio_return = (
            (self.portfolio_value - prev_pv) / prev_pv if prev_pv > 0 else 0.0
        )
        margin_used = abs(new_position_pct) * self.initial_capital

        # ── Per-step trade log ────────────────────────────────────────────────
        self.trade_log.append({
            # identity
            "asset":                          self._asset or "",
            "step":                           idx,
            "timestamp_utc":                  str(ts_utc),
            # OHLC (NaN if not loaded from raw_price_bars)
            "open":                           open_t,
            "high":                           high_t,
            "low":                            low_t,
            "close":                          price_t,
            "price":                          price_t,
            "ohlc_source":                    str(self._ohlc_source[idx]) if hasattr(self, "_ohlc_source") else "",
            # action (multi-dim)
            "action":                         f"[{dir_idx},{size_idx},{risk_idx}]",
            "action_name":                    action_name,
            "direction":                      direction,
            "size_label":                     size_label,
            "risk_profile":                   applied_risk,
            "leverage_used":                  float(applied_leverage),
            "tp_price":                       float(self._trade_tp_price) if self._trade_tp_price is not None else None,
            "sl_price":                       float(self._trade_sl_price) if self._trade_sl_price is not None else None,
            "risk_reward_ratio":              float(self._trade_rr_ratio),
            # position
            "desired_position":               float(desired_pct),
            "executed_position":              float(new_position_pct),
            "position_fraction":              float(new_position_pct),
            "previous_position":              float(old_position_pct),
            "effective_position":             float(effective_pos),
            "margin_used":                    float(margin_used),
            "clipped":                        clipped,
            "clip_reason":                    clip_reason,
            # trade lifecycle
            "in_position":                    self.in_position,
            "trade_side":                     self.trade_side,
            "trade_id":                       self._trade_id,
            "entry_price":                    float(self.entry_price) if self.entry_price is not None else None,
            "entry_time_utc":                 str(self.entry_time) if self.entry_time else "",
            "holding_bars":                   self.steps_in_position,
            "unrealized_pnl":                 float(self.unrealized_pnl),
            "max_favorable_excursion":        float(self.max_favorable_excursion),
            "max_adverse_excursion":          float(self.max_adverse_excursion),
            "exit_reason":                    exit_reason,
            "realized_pnl":                   float(realized_pnl),
            # market
            "actual_return":                  float(price_ret),
            "predicted_log_return":           predicted_lr,
            "signal_strength":                signal_str,
            "derived_direction":              derived_dir,
            # portfolio
            "portfolio_value":                float(self.portfolio_value),
            "previous_portfolio_value":       float(prev_pv),
            "portfolio_return":               float(portfolio_return),
            "cash":                           float(cash),
            "cash_balance":                   float(cash),
            "drawdown":                       float(self.current_drawdown),
            # P&L / cost
            "step_return":                    float(pnl_t),
            "reward_total":                   float(reward),
            "transaction_cost":               float(tc_t),
            "turnover":                       float(eff_turnover),
            # event flags
            "stop_loss_triggered":            sl_triggered,
            "take_profit_triggered":          tp_triggered,
            "max_hold_triggered":             max_hold_triggered,
            "max_drawdown_triggered":         max_drawdown_triggered,
            # reward components
            "reward_pnl":                     reward_pnl,
            "reward_transaction_cost":        reward_transaction_cost,
            "reward_drawdown_penalty":        reward_drawdown_penalty,
            "reward_overtrading_penalty":     reward_overtrading_penalty,
            "reward_missed_opportunity_penalty": reward_missed_opportunity_penalty,
            "reward_wrong_side_penalty":      reward_wrong_side_penalty,
            "reward_tp_bonus":                reward_tp_bonus,
            "reward_sl_penalty":              reward_sl_penalty,
            "reward":                         float(reward),
        })

        self._step += 1
        terminated = (self._step >= self.n_steps - 1) or self.done
        truncated  = False

        episode_sharpe = 0.0
        if terminated:
            # Emit any open trade at episode end
            if self.in_position:
                self.trade_log[-1]["exit_reason"] = "END_OF_EPISODE"
                _emit_completed_trade("END_OF_EPISODE")
                _reset_trade_state()

            rets = self.episode_returns
            if len(rets) > 1 and np.std(rets) > 0:
                episode_sharpe  = (np.mean(rets) / np.std(rets)) * np.sqrt(8760)
                terminal_reward = 0.1 * episode_sharpe
            else:
                terminal_reward = 0.0
            if self.max_drawdown_breached:
                terminal_reward -= 1.0
            reward += terminal_reward

            self.metrics["max_drawdown"] = float(self.current_drawdown)
            self.metrics["sharpe"]       = float(episode_sharpe)
            if self.closed_trades:
                self.metrics["win_rate"] = float(
                    sum(1 for p in self.closed_trades if p > 0) / len(self.closed_trades)
                )
            if self._hold_bars_list:
                self.metrics["avg_hold_bars"] = float(np.mean(self._hold_bars_list))

            n_s      = self.metrics["n_steps"]
            avg_pos  = self.metrics.get("sum_position", 0.0) / n_s if n_s > 0 else 0.0
            avg_rew  = self.metrics["total_reward"] / n_s if n_s > 0 else 0.0
            dir_dist = self.metrics["direction_counts"]
            logger.info(
                f"[{self._asset or 'ENV'}] Episode end:\n"
                f"  total_pnl={self.metrics['total_pnl']:.2%}  "
                f"num_trades={self.metrics['n_trades']}  "
                f"completed_trades={len(self.completed_trade_log)}  "
                f"win_rate={self.metrics['win_rate']:.1%}\n"
                f"  sl_hits={self.metrics['n_stop_losses_hit']}  "
                f"tp_hits={self.metrics['n_take_profits_hit']}  "
                f"max_hold_exits={self.metrics['n_max_hold_exits']}\n"
                f"  max_drawdown={self.metrics['max_drawdown']:.2%}  "
                f"sharpe={self.metrics['sharpe']:.2f}  "
                f"transaction_cost_total={self.metrics['total_transaction_cost']:.4f}\n"
                f"  avg_position={avg_pos:.3f}  avg_reward={avg_rew:.5f}\n"
                f"  reward_breakdown: pnl={self.metrics['total_reward_pnl']:.4f}  "
                f"tc={self.metrics['total_transaction_cost_penalty']:.4f}  "
                f"dd_pen={self.metrics['total_drawdown_penalty']:.4f}  "
                f"ot_pen={self.metrics['total_overtrading_penalty']:.4f}  "
                f"miss={self.metrics['total_missed_opportunity_penalty']:.4f}  "
                f"ws={self.metrics['total_wrong_side_penalty']:.4f}  "
                f"tp_bon={self.metrics['total_tp_bonus']:.4f}  "
                f"sl_pen={self.metrics['total_sl_penalty']:.4f}\n"
                f"  direction_distribution: "
                f"HOLD={dir_dist[0]} FLAT={dir_dist[1]} "
                f"LONG={dir_dist[2]} SHORT={dir_dist[3]}"
            )

        obs  = self._obs()
        info = {
            "price":            price_t,
            "position_pct":     self.current_position_pct,
            "portfolio_value":  self.portfolio_value,
            "total_value":      self.portfolio_value,
            "current_drawdown": self.current_drawdown,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        price = float(self._prices[min(self._step, self.n_steps - 1)])
        print(
            f"step={self._step:5d}  price={price:10.2f}  "
            f"pos={self.current_position_pct:+.2f}  "
            f"lev={self._trade_leverage:.1f}  "
            f"eff={self.current_position_pct * self._trade_leverage:+.2f}  "
            f"value={self.portfolio_value:10.2f}  dd={self.current_drawdown:.2%}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        idx   = min(self._step, self.n_steps - 1)
        feat  = self._features[idx].copy()
        price = float(self._prices[idx])

        if self.current_position_pct != 0.0 and self.entry_price and self.entry_price != 0:
            ps             = 1.0 if self.current_position_pct > 0 else -1.0
            unrealised_pnl = ps * (price - self.entry_price) / self.entry_price
        else:
            unrealised_pnl = 0.0

        cash_amount = max(
            0.0,
            self.portfolio_value - abs(self.current_position_pct) * self.initial_capital,
        )
        cash_ratio = cash_amount / self.initial_capital

        portfolio_state = np.array([
            float(self.current_position_pct),
            float(unrealised_pnl),
            float(cash_ratio),
            float(self.portfolio_value / self.initial_capital),
            float(self.current_drawdown),
            float(self.steps_in_position / 100.0),
            float(self._hours[idx] / 24.0),
        ], dtype=np.float32)

        return np.concatenate([feat, portfolio_state])

    @staticmethod
    def _normalise(X: np.ndarray) -> np.ndarray:
        mean = np.nanmean(X, axis=0)
        std  = np.nanstd(X, axis=0)
        std  = np.where(std == 0, 1.0, std)
        return np.nan_to_num((X - mean) / std, nan=0.0)

    @staticmethod
    def _normalise_transformer(X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X, dtype=np.float32)
        if X.shape[1] >= 1:
            out[:, 0] = np.clip(X[:, 0], -0.05, 0.05) / 0.05
        if X.shape[1] >= 2:
            out[:, 1] = np.clip(X[:, 1], 0.0, 0.05) / 0.05
        if X.shape[1] >= 3:
            out[:, 2] = X[:, 2].astype(np.float32)
        return np.nan_to_num(out, nan=0.0)

    @staticmethod
    def _init_metrics() -> dict:
        return {
            "total_pnl":                        0.0,
            "n_trades":                         0,
            "n_stop_losses_hit":                0,
            "n_take_profits_hit":               0,
            "n_max_hold_exits":                 0,
            "n_clipped_actions":                0,
            "max_drawdown":                     0.0,
            "win_rate":                         0.0,
            "avg_hold_bars":                    0.0,
            "sharpe":                           0.0,
            "total_transaction_cost":           0.0,
            "total_reward":                     0.0,
            "total_reward_pnl":                 0.0,
            "total_transaction_cost_penalty":   0.0,
            "total_drawdown_penalty":           0.0,
            "total_overtrading_penalty":        0.0,
            "total_missed_opportunity_penalty": 0.0,
            "total_wrong_side_penalty":         0.0,
            "total_tp_bonus":                   0.0,
            "total_sl_penalty":                 0.0,
            "n_steps":                          0,
            "sum_position":                     0.0,
            "direction_counts":                 {i: 0 for i in range(4)},
        }


# ── CSV export helpers ────────────────────────────────────────────────────────

def _csv_name_parts(asset: str, target_horizon: int, target_mode: str) -> str:
    """Return the suffix used in asset-specific CSV filenames."""
    mode_safe = target_mode.replace("_", "")
    return f"{asset.lower()}_h{target_horizon}_{mode_safe}"


def save_trade_log_csv(
    env: "TradingEnv",
    asset: str,
    target_horizon: int,
    target_mode: str,
    leverage: float,
    output_dir: Path = None,
) -> Path:
    """Write step-level trade log to CSV. Returns path to specific file."""
    if not env.trade_log:
        logger.warning(f"[RL/{asset}] trade_log is empty, nothing to save")
        return None

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(env.trade_log)
    df["asset"] = asset

    csv_cols = [
        "asset", "timestamp_utc",
        "open", "high", "low", "close", "price","ohlc_source",
        "action", "action_name",
        "direction", "size_label", "risk_profile",
        "position_fraction", "leverage_used", "effective_position",
        "margin_used", "cash_balance", "cash",
        "desired_position", "executed_position", "previous_position",
        "portfolio_value", "previous_portfolio_value", "portfolio_return",
        "drawdown",
        "in_position", "trade_side", "trade_id",
        "entry_price", "tp_price", "sl_price", "risk_reward_ratio",
        "entry_time_utc", "holding_bars", "exit_reason",
        "unrealized_pnl", "realized_pnl",
        "step_return", "reward_total",
        "transaction_cost", "turnover",
        "stop_loss_triggered", "take_profit_triggered",
        "max_hold_triggered", "max_drawdown_triggered",
        "max_favorable_excursion", "max_adverse_excursion",
        "predicted_log_return", "signal_strength",
        "reward_pnl", "reward_transaction_cost",
        "reward_drawdown_penalty", "reward_overtrading_penalty",
        "reward_missed_opportunity_penalty", "reward_wrong_side_penalty",
        "reward_tp_bonus", "reward_sl_penalty",
        "reward",
    ]
    missing = [c for c in csv_cols if c not in df.columns]
    if missing:
        logger.warning(f"[RL/{asset}] step-log CSV missing columns: {missing}")

    out_df = df[[c for c in csv_cols if c in df.columns]]
    suffix  = _csv_name_parts(asset, target_horizon, target_mode)
    specific = output_dir / f"latest_rl_trades_{suffix}.csv"
    generic  = output_dir / "latest_rl_trades.csv"
    out_df.to_csv(specific, index=False)
    out_df.to_csv(generic,  index=False)
    logger.info(
        f"[RL/{asset}] step-log → {specific} ({len(out_df)} rows)"
    )
    return specific


def save_trade_history_csv(
    env: "TradingEnv",
    asset: str,
    target_horizon: int,
    target_mode: str,
    leverage: float,
    output_dir: Path = None,
) -> Path:
    """Write completed-trade history to CSV. Returns path to specific file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not env.completed_trade_log:
        logger.warning(
            f"[RL/{asset}] *** No completed trades were recorded; "
            "trade history CSV will be empty. ***"
        )
        pd.DataFrame().to_csv(output_dir / "latest_rl_trade_history.csv", index=False)
        suffix = _csv_name_parts(asset, target_horizon, target_mode)
        pd.DataFrame().to_csv(
            output_dir / f"latest_rl_trade_history_{suffix}.csv", index=False
        )
        return None

    df = pd.DataFrame(env.completed_trade_log)
    df["asset"] = asset

    col_order = [
        "trade_id", "asset", "side",
        "entry_time", "exit_time", "holding_bars",
        "entry_price", "exit_price",
        "high_during_trade", "low_during_trade",
        "position_fraction", "leverage_used", "effective_position",
        "notional_value", "margin_used",
        "tp_price", "sl_price", "risk_reward_ratio",
        "exit_reason",
        "gross_pnl", "transaction_cost_total", "net_pnl", "return_pct",
        "reward_total",
        "reward_pnl_total", "reward_transaction_cost_total",
        "reward_drawdown_penalty_total", "reward_overtrading_penalty_total",
        "reward_missed_opportunity_penalty_total", "reward_wrong_side_penalty_total",
        "reward_tp_bonus_total", "reward_sl_penalty_total",
        "portfolio_value_before", "portfolio_value_after",
        "max_favorable_excursion", "max_adverse_excursion",
        "predicted_log_return_at_entry", "signal_strength_at_entry",
        "direction_action_at_entry", "size_label_at_entry", "risk_profile_at_entry",
    ]
    missing = [c for c in col_order if c not in df.columns]
    if missing:
        logger.warning(f"[RL/{asset}] trade-history CSV missing columns: {missing}")

    out_df  = df[[c for c in col_order if c in df.columns]]
    suffix  = _csv_name_parts(asset, target_horizon, target_mode)
    specific = output_dir / f"latest_rl_trade_history_{suffix}.csv"
    generic  = output_dir / "latest_rl_trade_history.csv"
    out_df.to_csv(specific, index=False)
    out_df.to_csv(generic,  index=False)
    logger.info(
        f"[RL/{asset}] trade-history → {specific} ({len(out_df)} completed trades)"
    )
    return specific


# ── Convenience factory ───────────────────────────────────────────────────────

def make_env(df: pd.DataFrame, asset: str = None, **kwargs) -> TradingEnv:
    return TradingEnv(df, asset=asset, **kwargs)
