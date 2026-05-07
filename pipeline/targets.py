"""
pipeline/targets.py
Phase 5 — Generate forward-looking prediction targets from the feature table.

Targets per bar (strictly no look-ahead leakage):
  ret_1h        — log return over next 1 bar  (1 hour  at hourly cadence)
  ret_4h        — log return over next 4 bars (4 hours at hourly cadence)
  direction_1h  — 3-class label: 1 up / 0 neutral / -1 down  (±0.2% threshold)
  direction_4h  — same for ret_4h
  norm_ret_1h   — ret_1h / vol_target_20  clipped to [-5, 5]
  norm_ret_4h   — ret_4h / vol_target_20  clipped to [-5, 5]
  vol_target_20 — trailing 20-bar log-return std (past only, no leakage)

Downstream consumers (transformer_model.py) query:
  raw  mode : t.ret_1h       AS future_log_return
  vol_norm  : t.norm_ret_1h  AS future_log_return, t.ret_1h, t.vol_target_20

Usage:
    python -m pipeline.targets
    python -m pipeline.targets --asset BTC
    python -m pipeline.targets --start 2024-01-01 --end 2024-06-01
    python -m pipeline.targets --audit --asset BTC --audit-n 10
    python -m pipeline.targets --migrate-precision
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import psycopg2.extras
from loguru import logger

from db.connection import get_conn, log_ingestion
from config import ALL_ASSETS

HORIZON_1H   = 1
HORIZON_4H   = 4
THRESHOLD    = 0.002       # ±0.2 % neutral boundary
VOL_WINDOW   = 20          # bars for rolling volatility
NORM_CLIP    = 5.0         # clip normalized returns to ±5 sigma
VOL_FLOOR    = 1e-6        # minimum vol to avoid division by zero

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS targets (
    id            BIGSERIAL    PRIMARY KEY,
    asset         VARCHAR(10)  NOT NULL,
    ts            TIMESTAMPTZ  NOT NULL,
    ret_1h        DOUBLE PRECISION,
    ret_4h        DOUBLE PRECISION,
    direction_1h  SMALLINT,
    direction_4h  SMALLINT,
    norm_ret_1h   DOUBLE PRECISION,
    norm_ret_4h   DOUBLE PRECISION,
    vol_target_20 DOUBLE PRECISION,
    inserted_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (asset, ts)
);
CREATE INDEX IF NOT EXISTS idx_targets_asset_ts ON targets (asset, ts DESC);
"""

# Migrations that add new columns to an existing targets table safely.
_ALTER_STMTS = [
    "ALTER TABLE targets ALTER COLUMN ret_1h TYPE DOUBLE PRECISION USING ret_1h::DOUBLE PRECISION",
    "ALTER TABLE targets ALTER COLUMN ret_4h TYPE DOUBLE PRECISION USING ret_4h::DOUBLE PRECISION",
    "ALTER TABLE targets ADD COLUMN IF NOT EXISTS norm_ret_1h   DOUBLE PRECISION",
    "ALTER TABLE targets ADD COLUMN IF NOT EXISTS norm_ret_4h   DOUBLE PRECISION",
    "ALTER TABLE targets ADD COLUMN IF NOT EXISTS vol_target_20 DOUBLE PRECISION",
]


# ── Schema ────────────────────────────────────────────────────────────────────

def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE)
        for stmt in _ALTER_STMTS:
            try:
                cur.execute("SAVEPOINT sp_mig")
                cur.execute(stmt)
                cur.execute("RELEASE SAVEPOINT sp_mig")
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT sp_mig")
                cur.execute("RELEASE SAVEPOINT sp_mig")


# ── Database I/O ──────────────────────────────────────────────────────────────

def load_close_series(conn, asset: str,
                      start: str = None, end: str = None) -> pd.Series:
    """Load close prices from features table. Returns UTC-indexed Series."""
    conditions = ["asset = %s"]
    params     = [asset]
    if start:
        conditions.append("ts >= %s"); params.append(start)
    if end:
        conditions.append("ts < %s");  params.append(end)
    sql = f"""
        SELECT ts, close FROM features
        WHERE  {' AND '.join(conditions)} AND close IS NOT NULL
        ORDER  BY ts
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        logger.warning(f"{asset}: no close prices in features table")
        return pd.Series(dtype=float, name="close")
    df = pd.DataFrame(rows)
    df["ts"]    = pd.to_datetime(df["ts"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.set_index("ts")["close"].sort_index()


def upsert_targets(conn, asset: str, df: pd.DataFrame) -> int:
    """Upsert target rows (index = ts) into the targets table."""
    if df.empty:
        return 0
    df = df.reset_index()

    def _f(v): return float(v) if pd.notna(v) else None
    def _i(v): return int(v)   if pd.notna(v) else None

    records = [
        (asset, row.ts,
         _f(row.ret_1h),  _f(row.ret_4h),
         _i(row.direction_1h), _i(row.direction_4h),
         _f(row.norm_ret_1h), _f(row.norm_ret_4h), _f(row.vol_target_20))
        for row in df.itertuples(index=False)
    ]
    sql = """
        INSERT INTO targets
               (asset, ts, ret_1h, ret_4h, direction_1h, direction_4h,
                norm_ret_1h, norm_ret_4h, vol_target_20)
        VALUES %s
        ON CONFLICT (asset, ts) DO UPDATE SET
            ret_1h        = EXCLUDED.ret_1h,
            ret_4h        = EXCLUDED.ret_4h,
            direction_1h  = EXCLUDED.direction_1h,
            direction_4h  = EXCLUDED.direction_4h,
            norm_ret_1h   = EXCLUDED.norm_ret_1h,
            norm_ret_4h   = EXCLUDED.norm_ret_4h,
            vol_target_20 = EXCLUDED.vol_target_20
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, records, page_size=500)
    return len(records)


# ── Target construction ───────────────────────────────────────────────────────

def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """
    ret[t] = log( close[t + horizon] / close[t] )
    Last `horizon` bars have no future: NaN.
    """
    return np.log(close.shift(-horizon) / close)


def direction_label(ret: pd.Series, threshold: float = THRESHOLD) -> pd.Series:
    """3-class labels: 1 up / 0 neutral / -1 down."""
    labels = pd.Series(np.where(
        ret.isna(),         np.nan,
        np.where(ret >  threshold,  1,
        np.where(ret < -threshold, -1, 0))
    ), index=ret.index, dtype="Int64")
    return labels


def build_targets(close: pd.Series) -> pd.DataFrame:
    """Compute all targets from a close-price Series. All outputs are causal."""
    ret_1h = forward_log_return(close, HORIZON_1H)
    ret_4h = forward_log_return(close, HORIZON_4H)

    # Trailing past-return volatility — uses log(close[t]/close[t-1]), fully causal.
    log_ret_past   = np.log(close / close.shift(1))
    rolling_vol_20 = log_ret_past.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    rolling_vol_20 = rolling_vol_20.clip(lower=VOL_FLOOR)

    norm_ret_1h = (ret_1h / rolling_vol_20).clip(-NORM_CLIP, NORM_CLIP)
    norm_ret_4h = (ret_4h / rolling_vol_20).clip(-NORM_CLIP, NORM_CLIP)

    return pd.DataFrame({
        "ret_1h":        ret_1h,
        "ret_4h":        ret_4h,
        "direction_1h":  direction_label(ret_1h),
        "direction_4h":  direction_label(ret_4h),
        "norm_ret_1h":   norm_ret_1h,
        "norm_ret_4h":   norm_ret_4h,
        "vol_target_20": rolling_vol_20,
    })


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(assets: list = None, start: str = None, end: str = None) -> dict[str, int]:
    """Phase 5: build and save forward-looking targets for all assets."""
    assets = assets or ALL_ASSETS

    with get_conn() as conn:
        ensure_table(conn)

    results: dict[str, int] = {}
    for asset in assets:
        try:
            with get_conn() as conn:
                close = load_close_series(conn, asset, start=start, end=end)
            if close.empty:
                logger.warning(f"{asset}: empty close series — skipping")
                results[asset] = 0
                continue

            tgt = build_targets(close)
            tgt = tgt.dropna(how="all")

            with get_conn() as conn:
                n = upsert_targets(conn, asset, tgt)

            logger.success(f"{asset}: saved {n} target rows")
            log_ingestion("targets", asset, "success", rows_saved=n)
            results[asset] = n

            # Log vol_norm coverage
            valid_norm = tgt["norm_ret_1h"].notna().sum()
            logger.info(
                f"  {asset}: vol_target_20 coverage={valid_norm}/{len(tgt)}  "
                f"norm_1h std={tgt['norm_ret_1h'].std():.3f}  "
                f"norm_1h range=[{tgt['norm_ret_1h'].min():.3f}, {tgt['norm_ret_1h'].max():.3f}]"
            )

        except Exception as e:
            logger.error(f"{asset}: target build failed — {e}")
            log_ingestion("targets", asset, "error", error_msg=str(e))
            results[asset] = 0

    return results


# ── Audit ─────────────────────────────────────────────────────────────────────

def audit_targets(assets: list = None, n_samples: int = 10) -> None:
    """Print n_samples random rows per asset comparing stored ret_1h vs manual."""
    assets = assets or ALL_ASSETS
    ok_assets = 0
    for asset in assets:
        print(f"\n{'='*72}")
        print(f" TARGET AUDIT: {asset}   (n={n_samples} random rows)")
        print(f"{'='*72}")
        try:
            with get_conn() as conn:
                sql = """
                    WITH price_lead AS (
                        SELECT ts, close,
                               LEAD(close, 1) OVER (PARTITION BY asset ORDER BY ts) AS close_next
                        FROM   features WHERE asset = %s
                    )
                    SELECT pl.ts,
                           pl.close                           AS close_t,
                           pl.close_next                      AS close_t_plus_1,
                           CAST(t.ret_1h AS DOUBLE PRECISION) AS stored_ret_1h,
                           CAST(t.norm_ret_1h AS DOUBLE PRECISION) AS stored_norm_1h,
                           CAST(t.vol_target_20 AS DOUBLE PRECISION) AS stored_vol
                    FROM   price_lead pl
                    JOIN   targets t ON t.ts = pl.ts
                    WHERE  t.asset = %s AND t.ret_1h IS NOT NULL AND pl.close_next IS NOT NULL
                    ORDER  BY RANDOM() LIMIT %s
                """
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (asset, asset, n_samples))
                    rows = cur.fetchall()
        except Exception as e:
            logger.error(f"  {asset}: query failed — {e}")
            continue

        if not rows:
            logger.warning(f"  {asset}: no rows returned")
            continue

        df = pd.DataFrame(rows).sort_values("ts")
        for c in ["close_t", "close_t_plus_1", "stored_ret_1h", "stored_norm_1h", "stored_vol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["manual"]     = np.log(df["close_t_plus_1"] / df["close_t"])
        df["difference"] = df["stored_ret_1h"] - df["manual"]

        print(f"  {'timestamp':26s}  {'stored':>12s}  {'manual':>12s}  {'diff':>10s}  "
              f"{'norm_1h':>8s}  {'vol_20':>10s}")
        print("  " + "-" * 84)
        for _, row in df.iterrows():
            print(
                f"  {str(row['ts'])[:26]:26s}  "
                f"{row['stored_ret_1h']:12.8f}  "
                f"{row['manual']:12.8f}  "
                f"{row['difference']:10.2e}  "
                f"{row['stored_norm_1h']:8.4f}  "
                f"{row['stored_vol']:10.8f}"
            )
        max_diff = df["difference"].abs().max()
        status   = "PASS" if max_diff < 1e-6 else "FAIL"
        print(f"\n  Max |difference| = {max_diff:.2e}  ->  {status}")
        if max_diff < 1e-6:
            ok_assets += 1
        else:
            print(f"  WARNING: ret_1h != log(close[t+1]/close[t]) — TARGET FORMULA BUG")

    print(f"\n{'='*72}")
    print(f" AUDIT SUMMARY: {ok_assets}/{len(assets)} assets PASS")
    print(f"{'='*72}\n")


# ── Precision migration ───────────────────────────────────────────────────────

def migrate_target_precision() -> None:
    """Upgrade ret_1h / ret_4h from NUMERIC(12,6) to DOUBLE PRECISION."""
    stmts = [
        "ALTER TABLE targets ALTER COLUMN ret_1h TYPE DOUBLE PRECISION USING ret_1h::DOUBLE PRECISION",
        "ALTER TABLE targets ALTER COLUMN ret_4h TYPE DOUBLE PRECISION USING ret_4h::DOUBLE PRECISION",
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            for stmt in stmts:
                logger.info(f"Running: {stmt}")
                cur.execute(stmt)
    logger.success("targets.ret_1h and targets.ret_4h migrated to DOUBLE PRECISION")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: build prediction targets")
    parser.add_argument("--asset",   default=None, help="BTC|ETH|GOLD|OIL (default: all)")
    parser.add_argument("--start",   default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end",     default=None, help="YYYY-MM-DD exclusive")
    parser.add_argument("--audit",   action="store_true",
                        help="Audit target formula for each asset")
    parser.add_argument("--audit-n", type=int, default=10,
                        help="Number of random rows to show per asset (default 10)")
    parser.add_argument("--migrate-precision", action="store_true",
                        help="Upgrade ret_1h/ret_4h to DOUBLE PRECISION")
    args = parser.parse_args()

    assets = [args.asset] if args.asset else ALL_ASSETS

    if args.migrate_precision:
        migrate_target_precision()
        return

    if args.audit:
        audit_targets(assets=assets, n_samples=args.audit_n)
        return

    results = run(assets=assets, start=args.start, end=args.end)
    total   = sum(results.values())
    logger.info(f"Done. Total target rows saved: {total}")
    for asset, n in results.items():
        logger.info(f"  {asset}: {n}")


if __name__ == "__main__":
    main()
