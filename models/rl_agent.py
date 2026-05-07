"""
models/rl_agent.py
Phase 7b — Train and evaluate a PPO reinforcement-learning trading agent.

Workflow:
  1. Load feature + close data from PostgreSQL.
  2. Train PPO on BTC (or specified asset).
  3. Evaluate on held-out test set: Sharpe, cumulative return, win rate, max drawdown.
  4. Compare RL agent vs. buy-and-hold benchmark.
  5. Save trained agent to models/saved/.

Usage:
    python -m models.rl_agent --asset BTC --target-horizon 4 --target-mode vol_norm --check-data
    python -m models.rl_agent --asset BTC --target-horizon 4 --target-mode vol_norm --env-smoke-test
    python -m models.rl_agent --asset BTC --debug-rl --retrain --target-horizon 4 --target-mode vol_norm --leverage 1.0
    python -m models.rl_agent --asset BTC --timesteps 200000 --retrain --target-horizon 4 --target-mode vol_norm
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import psycopg2.extras
from loguru import logger
from pathlib import Path

from db.connection import get_conn
from config import ALL_ASSETS
from models.rl_env import (TradingEnv, FEATURE_COLS, MARKET_FEATURE_COLS,
                           INITIAL_CASH, SAVED_DIR, STATE_DIM,
                           save_trade_log_csv, save_trade_history_csv)
from models.ridge_baseline import chronological_split

try:
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import joblib
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util   import make_vec_env
    from stable_baselines3.common.callbacks  import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.monitor    import Monitor
    _SB3 = True
except ImportError:
    logger.error("stable-baselines3 not found: pip install stable-baselines3")
    _SB3 = False

SAVED_DIR      = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(exist_ok=True)
DEFAULT_STEPS  = 200_000
DEBUG_STEPS    = 10_000


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(conn, asset: str) -> pd.DataFrame:
    """Load market feature columns + close price for one asset.

    Also LEFT JOINs open/high/low from raw_price_bars so the env can
    use real OHLC for candlestick charts and trade high/low tracking.
    Columns are NaN for any bar that has no raw_price_bars entry.
    """
    feat_sel = ", ".join(f"f.{c}" for c in MARKET_FEATURE_COLS)
    sql = f"""
        SELECT f.ts,
               {feat_sel},
               rpb.open  AS open,
               rpb.high  AS high,
               rpb.low   AS low
        FROM   features f
        LEFT JOIN raw_price_bars rpb
               ON  rpb.asset = f.asset
               AND rpb.ts    = f.ts
        WHERE  f.asset = %s
          AND  f.close IS NOT NULL
        ORDER  BY f.ts
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (asset,))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ohlc_present = df[["open", "high", "low"]].notna().any().any()
    if not ohlc_present:
        logger.warning(
            f"[RL/{asset}] raw_price_bars has no rows matching features timestamps — "
            "open/high/low will be NaN in the env. "
            "Populate raw_price_bars to enable OHLC step-log and trade-history."
        )
    return df


# ── One-time cleanup ──────────────────────────────────────────────────────────

def _cleanup_test_predictions(asset: str) -> int:
    """Delete any transformer_predictions rows where target_mode ends with '_test'."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM transformer_predictions "
                "WHERE asset = %s AND target_mode LIKE %s",
                (asset, "%_test"),
            )
            deleted = cur.rowcount
    if deleted:
        logger.info(f"[RL/{asset}] Cleaned up {deleted} test prediction rows")
    return deleted


# ── Check-data mode (Task 6) ──────────────────────────────────────────────────

def check_data(asset: str, target_horizon: int, target_mode: str) -> None:
    """
    Print comprehensive diagnostics about prediction and market data for the
    requested asset/horizon/mode.  Does not train.  Exits after printing.
    """
    print(f"\n{'='*60}")
    print(f"check-data: asset={asset}  horizon={target_horizon}  target_mode={target_mode}")
    print(f"{'='*60}")

    with get_conn() as conn:
        # 1. Prediction groups
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT prediction_horizon, target_mode, COUNT(*)
                FROM transformer_predictions
                WHERE asset = %s
                GROUP BY 1, 2
                ORDER BY 1, 2
                """,
                (asset,),
            )
            groups = cur.fetchall()

        print(f"\n1. Prediction groups for {asset}:")
        found_requested = False
        for h, m, cnt in groups:
            marker = "  <-- requested" if h == target_horizon and m == target_mode else ""
            print(f"   horizon={h}  target_mode={m:20s}  count={cnt}{marker}")
            if h == target_horizon and m == target_mode:
                found_requested = True
        if not found_requested:
            print(f"   [MISSING] (horizon={target_horizon}, target_mode={target_mode}) — run transformer first")

        # 2. Duplicate timestamps for requested horizon/mode
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT ts FROM transformer_predictions
                    WHERE asset = %s AND prediction_horizon = %s AND target_mode = %s
                    GROUP BY ts HAVING COUNT(*) > 1
                ) sub
                """,
                (asset, target_horizon, target_mode),
            )
            dup_count = cur.fetchone()[0]
        print(f"\n2. Duplicate timestamps in ({target_horizon}, {target_mode}): {dup_count}")

        # 3. Market feature rows
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM features WHERE asset = %s", (asset,))
            mkt_count = cur.fetchone()[0]
        print(f"\n3. Market feature rows: {mkt_count}")

        # 4. Prediction rows for requested horizon/mode
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM transformer_predictions
                WHERE asset = %s AND prediction_horizon = %s AND target_mode = %s
                """,
                (asset, target_horizon, target_mode),
            )
            pred_count = cur.fetchone()[0]
        print(f"\n4. Prediction rows ({target_horizon}, {target_mode}): {pred_count}")

        # 5. Overlap between market and prediction timestamps
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM features f
                JOIN transformer_predictions tp
                  ON f.asset = tp.asset AND f.ts = tp.ts
                WHERE f.asset = %s
                  AND tp.prediction_horizon = %s
                  AND tp.target_mode = %s
                """,
                (asset, target_horizon, target_mode),
            )
            overlap = cur.fetchone()[0]
        print(f"\n5. Overlap rows (market x prediction): {overlap}")

        # 6-7. Timestamp ranges
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(ts), MAX(ts) FROM features WHERE asset = %s", (asset,)
            )
            mkt_min, mkt_max = cur.fetchone()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MIN(ts), MAX(ts) FROM transformer_predictions
                WHERE asset = %s AND prediction_horizon = %s AND target_mode = %s
                """,
                (asset, target_horizon, target_mode),
            )
            pred_min, pred_max = cur.fetchone()

        print(f"\n6. Market ts range:     {mkt_min} to {mkt_max}")
        print(f"\n7. Prediction ts range: {pred_min} to {pred_max}")

        # 8-9. Signal statistics
        if pred_count > 0:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        AVG(predicted_log_return), STDDEV(predicted_log_return),
                        MIN(predicted_log_return), MAX(predicted_log_return),
                        AVG(signal_strength),      STDDEV(signal_strength),
                        MIN(signal_strength),      MAX(signal_strength)
                    FROM transformer_predictions
                    WHERE asset = %s AND prediction_horizon = %s AND target_mode = %s
                    """,
                    (asset, target_horizon, target_mode),
                )
                row = cur.fetchone()
            print(
                f"\n8. predicted_log_return: "
                f"mean={row[0]:.6f}  std={row[1]:.6f}  "
                f"min={row[2]:.6f}  max={row[3]:.6f}"
            )
            print(
                f"\n9. signal_strength:      "
                f"mean={row[4]:.6f}  std={row[5]:.6f}  "
                f"min={row[6]:.6f}  max={row[7]:.6f}"
            )
        else:
            print("\n8-9. No predictions to compute signal stats.")

    print(f"\n{'='*60}\n")


# ── Env smoke test (Task 7) ───────────────────────────────────────────────────

def env_smoke_test(
    asset: str,
    target_horizon: int,
    target_mode: str,
    leverage: float = 1.0,
    sl_mult: float = None,
    tp_mult: float = None,
    max_holding_bars: int = None,
    signal_threshold: float = None,
    opportunity_threshold: float = None,
    seed: int = 42,
) -> None:
    """
    Load data, create TradingEnv, reset, take 10 random valid actions.
    Prints obs shape, reward, done/truncated.  Does not train.
    """
    from models.rl_env import (DEFAULT_SL_MULT, DEFAULT_TP_MULT,
                                DEFAULT_MAX_HOLDING_BARS,
                                DEFAULT_SIGNAL_THRESHOLD,
                                DEFAULT_OPPORTUNITY_THRESHOLD)
    env_extra = {
        "sl_mult":             sl_mult            or DEFAULT_SL_MULT,
        "tp_mult":             tp_mult            or DEFAULT_TP_MULT,
        "max_holding_bars":    max_holding_bars   or DEFAULT_MAX_HOLDING_BARS,
        "signal_threshold":    signal_threshold   or DEFAULT_SIGNAL_THRESHOLD,
        "opportunity_threshold": opportunity_threshold or DEFAULT_OPPORTUNITY_THRESHOLD,
    }
    logger.info(
        f"[smoke-test/{asset}] Loading market data "
        f"(horizon={target_horizon} target_mode={target_mode})"
    )
    with get_conn() as conn:
        df = load_features(conn, asset)

    if df.empty:
        logger.error(f"[smoke-test/{asset}] No market data found")
        return

    logger.info(f"[smoke-test/{asset}] market rows={len(df)}  Creating TradingEnv...")
    try:
        env = TradingEnv(
            df, asset=asset,
            prediction_horizon=target_horizon,
            target_mode=target_mode,
            leverage=leverage,
            **env_extra,
        )
    except Exception as e:
        logger.error(f"[smoke-test/{asset}] TradingEnv creation failed: {e}")
        raise

    obs, info = env.reset(seed=seed)
    print(f"\n[smoke-test/{asset}] obs_dim={obs.shape[0]}  env.n_steps={env.n_steps}")

    rng = np.random.default_rng(seed)
    for step_i in range(10):
        action = rng.integers(0, [4, 3, 3])
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  step={step_i+1:2d}  action={action}  "
            f"reward={reward:.6f}  done={terminated}  truncated={truncated}  "
            f"portfolio={info['portfolio_value']:.2f}"
        )
        if terminated or truncated:
            print("  Episode ended early.")
            break

    print(f"\n[smoke-test/{asset}] PASS — environment created and stepped successfully\n")


# ── Evaluation metrics ────────────────────────────────────────────────────────

def evaluate_agent(model, env: TradingEnv) -> dict:
    obs, _ = env.reset()
    done   = False

    portfolio_values = [INITIAL_CASH]
    step_returns     = []
    trades           = 0
    wins             = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        val = info["total_value"]
        portfolio_values.append(val)

        if len(portfolio_values) >= 2:
            r = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            step_returns.append(r)
            if r != 0:
                trades += 1
                if r > 0:
                    wins += 1

    vals = np.array(portfolio_values, dtype=float)
    rets = np.array(step_returns,     dtype=float)

    cum_return = (vals[-1] - vals[0]) / vals[0]
    sharpe     = _sharpe(rets)
    max_dd     = _max_drawdown(vals)
    win_rate   = wins / trades if trades > 0 else 0.0

    result = {
        "cumulative_return": float(cum_return),
        "sharpe_ratio":      float(sharpe),
        "win_rate":          float(win_rate),
        "max_drawdown":      float(max_dd),
        "n_trades":          trades,
        "final_value":       float(vals[-1]),
    }

    # ── Completed-trade metrics (primary result summary) ──────────────────────
    ct = env.completed_trade_log
    if ct:
        df_ct = pd.DataFrame(ct)
        ct_n           = len(df_ct)
        ct_net_pnl     = float(df_ct["net_pnl"].sum())
        ct_wins        = int((df_ct["net_pnl"] > 0).sum())
        ct_win_rate    = ct_wins / ct_n if ct_n > 0 else 0.0
        gross_profit   = float(df_ct.loc[df_ct["net_pnl"] > 0, "net_pnl"].sum())
        gross_loss     = abs(float(df_ct.loc[df_ct["net_pnl"] <= 0, "net_pnl"].sum()))
        profit_factor  = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_tc       = float(df_ct["transaction_cost_total"].sum())
        ct_return_pct  = ct_net_pnl / INITIAL_CASH * 100.0
        exit_counts    = df_ct["exit_reason"].value_counts().to_dict()

        result.update({
            "completed_trades":       ct_n,
            "closed_trade_net_pnl":   ct_net_pnl,
            "closed_trade_return_pct": ct_return_pct,
            "closed_trade_win_rate":  ct_win_rate,
            "profit_factor":          profit_factor,
            "total_transaction_costs": total_tc,
            "exit_reason_counts":     exit_counts,
        })
    else:
        result.update({
            "completed_trades":       0,
            "closed_trade_net_pnl":   0.0,
            "closed_trade_return_pct": 0.0,
            "closed_trade_win_rate":  0.0,
            "profit_factor":          0.0,
            "total_transaction_costs": 0.0,
            "exit_reason_counts":     {},
        })

    return result


def buy_and_hold(df: pd.DataFrame) -> dict:
    prices = df["close"].dropna().values
    if len(prices) < 2:
        return {}
    n_units    = INITIAL_CASH / prices[0]
    values     = n_units * prices
    rets       = np.diff(values) / values[:-1]
    cum_return = (values[-1] - values[0]) / values[0]
    return {
        "cumulative_return": float(cum_return),
        "sharpe_ratio":      float(_sharpe(rets)),
        "max_drawdown":      float(_max_drawdown(values)),
    }


# ── Finance helpers ───────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, periods_per_year: int = 365 * 24) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _max_drawdown(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    dd   = (values - peak) / np.where(peak == 0, 1.0, peak)
    return float(-dd.min())


# ── Training ──────────────────────────────────────────────────────────────────

def train_asset(
    asset: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    total_timesteps: int = DEFAULT_STEPS,
    scaler=None,
    market_feature_cols: list = None,
    target_horizon: int = 4,
    target_mode: str = "vol_norm",
    leverage: float = 1.0,
    sl_mult: float = None,
    tp_mult: float = None,
    max_holding_bars: int = None,
    signal_threshold: float = None,
    opportunity_threshold: float = None,
) -> "PPO | None":
    if not _SB3:
        return None

    from models.rl_env import (DEFAULT_SL_MULT, DEFAULT_TP_MULT,
                                DEFAULT_MAX_HOLDING_BARS,
                                DEFAULT_SIGNAL_THRESHOLD,
                                DEFAULT_OPPORTUNITY_THRESHOLD)
    env_kwargs = dict(
        asset=asset, scaler=scaler,
        market_feature_cols=market_feature_cols,
        prediction_horizon=target_horizon,
        target_mode=target_mode,
        leverage=leverage,
        sl_mult=sl_mult or DEFAULT_SL_MULT,
        tp_mult=tp_mult or DEFAULT_TP_MULT,
        max_holding_bars=max_holding_bars or DEFAULT_MAX_HOLDING_BARS,
        signal_threshold=signal_threshold or DEFAULT_SIGNAL_THRESHOLD,
        opportunity_threshold=opportunity_threshold or DEFAULT_OPPORTUNITY_THRESHOLD,
    )

    try:
        train_env = Monitor(TradingEnv(df_train, **env_kwargs))
        eval_env  = Monitor(TradingEnv(df_val,   **env_kwargs))
    except ValueError as e:
        logger.error(f"{asset}: environment creation failed — {e}")
        raise
    except Exception as e:
        logger.error(f"{asset}: environment creation failed — {e}")
        return None

    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=20, verbose=0
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(SAVED_DIR),
        log_path=str(SAVED_DIR),
        eval_freq=max(1, total_timesteps // 20),
        n_eval_episodes=3,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=0,
    )

    # v2: more stable PPO hyperparameters to reduce overtrading / exploration noise
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.003,
        verbose=0,
    )

    logger.info(f"{asset}: training PPO for {total_timesteps:,} steps ...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_cb,
                    progress_bar=False)
    except Exception as e:
        logger.error(f"{asset}: PPO training failed — {e}")
        return None

    save_path = SAVED_DIR / f"ppo_{asset.lower()}"
    model.save(str(save_path))
    logger.success(f"{asset}: PPO agent saved → {save_path}.zip")
    return model


# ── Per-asset pipeline ────────────────────────────────────────────────────────

def _fit_scaler(df_train: pd.DataFrame, asset: str,
                feat_cols: list = None) -> "StandardScaler | None":
    if not _SKLEARN:
        return None
    if feat_cols is None:
        feat_cols = MARKET_FEATURE_COLS
    feat_cols = [c for c in feat_cols if c in df_train.columns]
    X = df_train[feat_cols].fillna(df_train[feat_cols].median()).values.astype(float)
    scaler = StandardScaler()
    scaler.fit(X)
    if _JOBLIB:
        SAVED_DIR.mkdir(exist_ok=True)
        scaler_path = SAVED_DIR / f"feature_scaler_{asset.lower()}.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"{asset}: scaler saved → {scaler_path}")
    return scaler


def run_asset(
    asset: str,
    total_timesteps: int = DEFAULT_STEPS,
    force_retrain: bool = False,
    use_gdelt_sentiment: bool = False,
    target_horizon: int = 4,
    target_mode: str = "vol_norm",
    leverage: float = 1.0,
    sl_mult: float = None,
    tp_mult: float = None,
    max_holding_bars: int = None,
    signal_threshold: float = None,
    opportunity_threshold: float = None,
    seed: int = None,
    output_suffix: str = "",
) -> dict:
    """Full Phase 7 pipeline for one asset."""
    from models.feature_schema import get_feature_columns

    logger.info(
        f"\n[RL/{asset}] Startup config:\n"
        f"  asset={asset}  timesteps={total_timesteps:,}  retrain={force_retrain}\n"
        f"  target_horizon={target_horizon}  target_mode={target_mode}  "
        f"leverage={leverage}  seed={seed}"
    )

    # Clean up any test prediction rows
    _cleanup_test_predictions(asset)

    save_path = SAVED_DIR / f"ppo_{asset.lower()}.zip"
    if save_path.exists() and not force_retrain:
        logger.info(f"[RL/{asset}] Agent exists — loading for evaluation. Use --retrain to retrain.")
    else:
        force_retrain = True  # ensure training happens

    with get_conn() as conn:
        df = load_features(conn, asset)

    if df.empty or len(df) < 500:
        logger.warning(f"{asset}: not enough data ({len(df)} rows) — skipping")
        return {}

    if use_gdelt_sentiment:
        from models.gdelt_sentiment import merge_gdelt_features
        df = merge_gdelt_features(df, asset)

    market_feat_cols = get_feature_columns(
        asset, model="rl", use_gdelt_sentiment=use_gdelt_sentiment
    )
    market_feat_cols = [c for c in market_feat_cols if c in df.columns]
    logger.info(f"{asset}: {len(market_feat_cols)} market feature cols for RL")

    df_train, df_val, df_test = chronological_split(df)
    logger.info(
        f"{asset}: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test bars"
    )

    scaler = _fit_scaler(df_train, asset, feat_cols=market_feat_cols)

    from models.rl_env import (DEFAULT_SL_MULT, DEFAULT_TP_MULT,
                                DEFAULT_MAX_HOLDING_BARS,
                                DEFAULT_SIGNAL_THRESHOLD,
                                DEFAULT_OPPORTUNITY_THRESHOLD)
    _sl_mult             = sl_mult            or DEFAULT_SL_MULT
    _tp_mult             = tp_mult            or DEFAULT_TP_MULT
    _max_holding_bars    = max_holding_bars   or DEFAULT_MAX_HOLDING_BARS
    _signal_threshold    = signal_threshold   or DEFAULT_SIGNAL_THRESHOLD
    _opp_threshold       = opportunity_threshold or DEFAULT_OPPORTUNITY_THRESHOLD

    env_kwargs = dict(
        scaler=scaler,
        market_feature_cols=market_feat_cols,
        prediction_horizon=target_horizon,
        target_mode=target_mode,
        leverage=leverage,
        sl_mult=_sl_mult,
        tp_mult=_tp_mult,
        max_holding_bars=_max_holding_bars,
        signal_threshold=_signal_threshold,
        opportunity_threshold=_opp_threshold,
    )

    if force_retrain:
        model = train_asset(
            asset, df_train, df_val, total_timesteps,
            target_horizon=target_horizon, target_mode=target_mode,
            leverage=leverage, scaler=scaler, market_feature_cols=market_feat_cols,
            sl_mult=_sl_mult, tp_mult=_tp_mult,
            max_holding_bars=_max_holding_bars,
            signal_threshold=_signal_threshold,
            opportunity_threshold=_opp_threshold,
        )
        if model is None:
            return {}
    else:
        try:
            model = PPO.load(str(save_path))
            logger.info(f"[RL/{asset}] Loaded existing agent from {save_path}")
        except Exception as e:
            logger.error(f"[RL/{asset}] Failed to load agent: {e}")
            return {}

    # Evaluate on test set
    try:
        test_env    = TradingEnv(df_test, asset=asset, **env_kwargs)
        rl_metrics  = evaluate_agent(model, test_env)
        bnh_metrics = buy_and_hold(df_test)
    except Exception as e:
        logger.error(f"{asset}: evaluation failed — {e}")
        return {}

    # Save step-level and trade-history CSVs
    _base_dir = Path(__file__).parent.parent / "data" / "dashboard"
    _out_dir  = (_base_dir / output_suffix) if output_suffix else _base_dir
    save_trade_log_csv(test_env, asset, target_horizon, target_mode, leverage, output_dir=_out_dir)
    save_trade_history_csv(test_env, asset, target_horizon, target_mode, leverage, output_dir=_out_dir)

    # Summary log
    _log_trade_summary(test_env, asset, rl_metrics)

    # Log RL vs B&H comparison
    logger.info(f"\n{'─'*55}")
    logger.info(f"{asset} — RL agent vs. Buy-and-Hold (test set)")
    logger.info(f"{'─'*55}")
    _log_comparison(rl_metrics, bnh_metrics)

    return {"rl": rl_metrics, "buy_and_hold": bnh_metrics}


def _log_trade_summary(env: "TradingEnv", asset: str, rl_metrics: dict) -> None:
    """Log a concise post-evaluation trade summary."""
    trades = env.completed_trade_log
    n      = len(trades)
    if n == 0:
        logger.warning(
            f"[RL/{asset}] *** No completed trades were recorded; "
            "trade history CSV will be empty. ***"
        )
        return

    df = pd.DataFrame(trades)

    wins          = (df["net_pnl"] > 0).sum()
    losses        = (df["net_pnl"] <= 0).sum()
    win_rate      = wins / n if n > 0 else 0.0
    gross_profit  = df.loc[df["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss    = abs(df.loc[df["net_pnl"] <= 0, "net_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_rr        = df["risk_reward_ratio"].mean()
    avg_hold      = df["holding_bars"].mean()
    exit_counts   = df["exit_reason"].value_counts().to_dict()

    final_pv    = rl_metrics.get("final_value", env.portfolio_value)
    net_ret_pct = (final_pv - INITIAL_CASH) / INITIAL_CASH * 100
    max_dd      = rl_metrics.get("max_drawdown", env.metrics.get("max_drawdown", 0.0))

    logger.info(
        f"\n[RL/{asset}] ── Trade Log Summary ──────────────────────────\n"
        f"  Step rows         : {len(env.trade_log)}\n"
        f"  Completed trades  : {n}\n"
        f"  Final portfolio   : ${final_pv:,.2f}\n"
        f"  Net return        : {net_ret_pct:+.2f}%\n"
        f"  Max drawdown      : {max_dd:.2%}\n"
        f"  Win rate          : {win_rate:.1%}  ({wins}W / {losses}L)\n"
        f"  Profit factor     : {profit_factor:.2f}\n"
        f"  Avg risk/reward   : {avg_rr:.2f}\n"
        f"  Avg holding bars  : {avg_hold:.1f}\n"
        f"  Exit reasons      : {exit_counts}\n"
        f"────────────────────────────────────────────────────────────"
    )


def check_rl_logs(
    asset: str,
    target_horizon: int,
    target_mode: str,
) -> None:
    """Read latest_rl_trades.csv and latest_rl_trade_history.csv and print diagnostics."""
    from pathlib import Path
    output_dir = Path(__file__).parent.parent / "data" / "dashboard"

    step_required = [
        "asset", "timestamp_utc", "open", "high", "low", "close","ohlc_source",
        "action", "direction", "size_label", "risk_profile",
        "position_fraction", "leverage_used", "effective_position",
        "portfolio_value", "cash_balance", "margin_used", "unrealized_pnl",
        "step_return", "reward_total",
        "reward_pnl", "reward_transaction_cost", "reward_drawdown_penalty",
        "reward_overtrading_penalty", "reward_missed_opportunity_penalty",
        "reward_wrong_side_penalty", "reward_tp_bonus", "reward_sl_penalty",
        "transaction_cost", "drawdown", "holding_bars",
        "in_position", "trade_side", "entry_price", "tp_price", "sl_price",
        "risk_reward_ratio", "exit_reason",
        "predicted_log_return", "signal_strength",
    ]
    hist_required = [
        "trade_id", "asset", "side", "entry_time", "exit_time", "holding_bars",
        "entry_price", "exit_price", "high_during_trade", "low_during_trade",
        "position_fraction", "leverage_used", "effective_position",
        "notional_value", "margin_used", "tp_price", "sl_price",
        "risk_reward_ratio", "exit_reason",
        "gross_pnl", "transaction_cost_total", "net_pnl", "return_pct",
        "reward_total", "reward_pnl_total", "reward_transaction_cost_total",
        "reward_drawdown_penalty_total", "reward_overtrading_penalty_total",
        "reward_missed_opportunity_penalty_total", "reward_wrong_side_penalty_total",
        "reward_tp_bonus_total", "reward_sl_penalty_total",
        "portfolio_value_before", "portfolio_value_after",
        "max_favorable_excursion", "max_adverse_excursion",
        "predicted_log_return_at_entry", "signal_strength_at_entry",
        "direction_action_at_entry", "size_label_at_entry", "risk_profile_at_entry",
    ]

    def _check_file(path: Path, required_cols: list, label: str) -> None:
        print(f"\n--- {label} ---")
        if not path.exists():
            print(f"  MISSING: {path}")
            return
        df = pd.read_csv(path)
        print(f"  Path  : {path}")
        print(f"  Shape : {df.shape[0]} rows x {df.shape[1]} cols")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  MISSING COLUMNS ({len(missing)}): {missing}")
        else:
            print(f"  Required columns : ALL PRESENT ({len(required_cols)})")
        nan_cols = [c for c in df.columns if df[c].isna().all()]
        if nan_cols:
            print(f"  All-NaN columns  : {nan_cols}")
        if "timestamp_utc" in df.columns and df["timestamp_utc"].notna().any():
            print(f"  Timestamp range  : {df['timestamp_utc'].iloc[0]}  to  {df['timestamp_utc'].iloc[-1]}")
        elif "entry_time" in df.columns and df["entry_time"].notna().any():
            print(f"  Entry time range : {df['entry_time'].iloc[0]}  to  {df['entry_time'].iloc[-1]}")

        if "portfolio_value" in df.columns:
            final_pv = df["portfolio_value"].dropna().iloc[-1] if not df["portfolio_value"].dropna().empty else None
            if final_pv:
                net = (final_pv - INITIAL_CASH) / INITIAL_CASH * 100
                print(f"  Final portfolio  : ${final_pv:,.2f}  ({net:+.2f}%)")

        if "exit_reason" in df.columns:
            print(f"  exit_reason counts :\n{df['exit_reason'].value_counts().to_string()}")
        if "risk_profile" in df.columns:
            print(f"  risk_profile counts:\n{df['risk_profile'].value_counts().to_string()}")
        if "risk_profile_at_entry" in df.columns:
            print(f"  risk_profile_at_entry:\n{df['risk_profile_at_entry'].value_counts().to_string()}")

        if "net_pnl" in df.columns:
            n   = len(df)
            wins = (df["net_pnl"] > 0).sum()
            print(f"  Completed trades : {n}   win_rate={wins/n:.1%}" if n > 0 else "  Completed trades : 0")

    suffix     = f"{asset.lower()}_h{target_horizon}_{target_mode.replace('_', '')}"
    step_path  = output_dir / f"latest_rl_trades_{suffix}.csv"
    hist_path  = output_dir / f"latest_rl_trade_history_{suffix}.csv"
    # Fall back to generic if specific not found
    if not step_path.exists():
        step_path = output_dir / "latest_rl_trades.csv"
    if not hist_path.exists():
        hist_path = output_dir / "latest_rl_trade_history.csv"

    print(f"\n{'='*60}")
    print(f"check-rl-logs: asset={asset}  horizon={target_horizon}  mode={target_mode}")
    print(f"{'='*60}")
    _check_file(step_path, step_required,  "Step-level log  (latest_rl_trades)")
    _check_file(hist_path, hist_required,  "Trade history   (latest_rl_trade_history)")
    print(f"\n{'='*60}\n")


def _log_comparison(rl: dict, bnh: dict) -> None:
    rows = [
        ("Cumulative return", rl.get("cumulative_return"), bnh.get("cumulative_return"), ".2%"),
        ("Sharpe ratio",      rl.get("sharpe_ratio"),      bnh.get("sharpe_ratio"),      ".3f"),
        ("Max drawdown",      rl.get("max_drawdown"),      bnh.get("max_drawdown"),      ".2%"),
        ("Win rate",          rl.get("win_rate"),           None,                         ".2%"),
        ("# Trades",          rl.get("n_trades"),           None,                         "d"),
    ]
    for name, rl_val, bnh_val, fmt in rows:
        rl_str  = format(rl_val,  fmt) if rl_val  is not None else "—"
        bnh_str = format(bnh_val, fmt) if bnh_val is not None else "—"
        logger.info(f"  {name:<22}  RL: {rl_str:>10}   B&H: {bnh_str:>10}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(
    assets: list = None,
    total_timesteps: int = DEFAULT_STEPS,
    force_retrain: bool = False,
    use_gdelt_sentiment: bool = False,
    target_horizon: int = 4,
    target_mode: str = "vol_norm",
    leverage: float = 1.0,
    sl_mult: float = None,
    tp_mult: float = None,
    max_holding_bars: int = None,
    signal_threshold: float = None,
    opportunity_threshold: float = None,
    seed: int = None,
    output_suffix: str = "",
) -> dict:
    assets = assets or ALL_ASSETS
    ordered = (["BTC"] + [a for a in assets if a != "BTC"]
               if "BTC" in assets else assets)

    results: dict = {}
    for asset in ordered:
        try:
            results[asset] = run_asset(
                asset,
                total_timesteps=total_timesteps,
                force_retrain=force_retrain,
                use_gdelt_sentiment=use_gdelt_sentiment,
                target_horizon=target_horizon,
                target_mode=target_mode,
                leverage=leverage,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                max_holding_bars=max_holding_bars,
                signal_threshold=signal_threshold,
                opportunity_threshold=opportunity_threshold,
                seed=seed,
                output_suffix=output_suffix,
            )
        except Exception as e:
            logger.error(f"{asset}: RL pipeline failed — {e}")
            results[asset] = {}
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 7: PPO RL trading agent")
    parser.add_argument("--asset",          default=None,
                        help="BTC|ETH|GOLD|OIL (default: all, BTC first)")
    parser.add_argument("--timesteps",      type=int, default=None,
                        help=f"PPO training steps (default: {DEFAULT_STEPS:,}; "
                             f"--debug-rl sets {DEBUG_STEPS:,})")
    parser.add_argument("--retrain",        action="store_true",
                        help="Force new PPO training even if checkpoint exists")
    parser.add_argument("--target-horizon", type=int, default=4, choices=[1, 4],
                        help="Transformer prediction horizon (default: 4)")
    parser.add_argument("--target-mode",    default="vol_norm",
                        choices=["raw", "vol_norm"],
                        help="Transformer target mode (default: vol_norm)")
    parser.add_argument("--leverage",       type=float, default=1.0,
                        help="Leverage multiplier (default: 1.0)")
    parser.add_argument("--sl-mult",        type=float, default=None,
                        help="Stop-loss ATR multiplier (default: 1.5)")
    parser.add_argument("--tp-mult",        type=float, default=None,
                        help="Take-profit ATR multiplier (default: 2.5)")
    parser.add_argument("--max-holding-bars", type=int, default=None,
                        help="Max bars to hold a position before forced FLAT (default: 48)")
    parser.add_argument("--signal-threshold", type=float, default=None,
                        help="Min |predicted_lr| to flag strong signal (default: 0.001)")
    parser.add_argument("--opportunity-threshold", type=float, default=None,
                        help="Min |price_ret| for missed-opportunity check (default: 0.002)")
    parser.add_argument("--debug-rl",       action="store_true",
                        help=f"Fast smoke run: {DEBUG_STEPS:,} timesteps (overridden by --timesteps)")
    parser.add_argument("--seed",           type=int, default=None,
                        help="Random seed")
    parser.add_argument("--check-data",     action="store_true",
                        help="Print data diagnostics and exit without training")
    parser.add_argument("--env-smoke-test", action="store_true",
                        help="Create env, take 10 random steps, exit without training")
    parser.add_argument("--check-rl-logs", action="store_true",
                        help="Read saved CSV logs and print diagnostics; no training")
    parser.add_argument("--use-gdelt-sentiment", action="store_true")
    parser.add_argument(
        "--output-suffix", default="",
        help="Subfolder name under data/dashboard/ for output CSVs "
             "(e.g. 'rl_test_v2' saves to data/dashboard/rl_test_v2/). "
             "Empty string writes to data/dashboard/ (default, overwrites old files)."
    )
    args = parser.parse_args()

    # Resolve timesteps
    if args.timesteps is not None:
        timesteps = args.timesteps
    elif args.debug_rl:
        timesteps = DEBUG_STEPS
    else:
        timesteps = DEFAULT_STEPS

    assets = [args.asset] if args.asset else ALL_ASSETS

    # --check-data mode
    if args.check_data:
        for asset in assets:
            check_data(asset, args.target_horizon, args.target_mode)
        return

    # --check-rl-logs mode
    if args.check_rl_logs:
        for asset in assets:
            check_rl_logs(asset, args.target_horizon, args.target_mode)
        return

    # --env-smoke-test mode
    if args.env_smoke_test:
        for asset in assets:
            env_smoke_test(
                asset, args.target_horizon, args.target_mode,
                leverage=args.leverage,
                sl_mult=args.sl_mult,
                tp_mult=args.tp_mult,
                max_holding_bars=args.max_holding_bars,
                signal_threshold=args.signal_threshold,
                opportunity_threshold=args.opportunity_threshold,
                seed=args.seed or 42,
            )
        return

    # Normal training / evaluation
    logger.info(
        f"RL startup: assets={assets}  timesteps={timesteps:,}  retrain={args.retrain}\n"
        f"  target_horizon={args.target_horizon}  target_mode={args.target_mode}  "
        f"leverage={args.leverage}  seed={args.seed}"
    )

    results = run(
        assets=assets,
        total_timesteps=timesteps,
        force_retrain=args.retrain,
        use_gdelt_sentiment=args.use_gdelt_sentiment,
        target_horizon=args.target_horizon,
        target_mode=args.target_mode,
        leverage=args.leverage,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        max_holding_bars=args.max_holding_bars,
        signal_threshold=args.signal_threshold,
        opportunity_threshold=args.opportunity_threshold,
        seed=args.seed,
        output_suffix=args.output_suffix,
    )

    logger.info("\nFinal summary:")
    for asset, res in results.items():
        rl = res.get("rl", {})
        if rl:
            logger.info(
                f"  {asset}: ret={rl.get('cumulative_return', float('nan')):.2%}  "
                f"sharpe={rl.get('sharpe_ratio', float('nan')):.3f}  "
                f"dd={rl.get('max_drawdown', float('nan')):.2%}  "
                f"completed_trades={rl.get('completed_trades', '?')}  "
                f"net_pnl={rl.get('closed_trade_net_pnl', float('nan')):+.2f}  "
                f"ct_ret={rl.get('closed_trade_return_pct', float('nan')):+.2f}%  "
                f"win_rate={rl.get('closed_trade_win_rate', float('nan')):.1%}  "
                f"pf={rl.get('profit_factor', float('nan')):.2f}  "
                f"exits={rl.get('exit_reason_counts', {})}"
            )


if __name__ == "__main__":
    main()
