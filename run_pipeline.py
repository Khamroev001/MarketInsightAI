"""
run_pipeline.py
MarketInsight AI — end-to-end pipeline runner.

Execution order:
  Step 1:  python -m ingestion.ingest_prices          # fetch yfinance OHLCV
  Step 2:  python -m ingestion.ingest_news_finnhub    # fetch Finnhub news
  Step 3:  python -m pipeline.features                # compute features + GDELT sentiment
  Step 4:  python -m pipeline.targets                 # compute labels with ±0.2% threshold
  Step 5:  python -m models.ridge_baseline            # train and evaluate Ridge
  Step 6:  python -m models.transformer_model         # train Transformer, save predictions
  Step 7:  python -m models.rl_agent                  # train PPO RL agent

Usage:
    python run_pipeline.py                  # full run (steps 1–7)
    python run_pipeline.py --from 3         # start from step 3 (skip ingest)
    python run_pipeline.py --only 6         # run only step 6 (Transformer)
    python run_pipeline.py --no-rl          # skip step 7 (RL is slow)
    python run_pipeline.py --no-ingest      # skip steps 1 and 2 (already ingested)
"""

import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger

# ── Required environment variables ────────────────────────────────────────────
_REQUIRED_ENV_VARS = [
    "DB_HOST",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
]

# Also accept Postgres-prefixed variants used in config.py
_ENV_ALIASES = {
    "DB_HOST":     "POSTGRES_HOST",
    "DB_NAME":     "POSTGRES_DB",
    "DB_USER":     "POSTGRES_USER",
    "DB_PASSWORD": "POSTGRES_PASSWORD",
}


def check_env_vars() -> None:
    """
    Verify required environment variables are set.
    Exits with code 1 and lists missing vars if any are absent.
    """
    from dotenv import load_dotenv
    load_dotenv()

    missing = []
    for var in _REQUIRED_ENV_VARS:
        alias = _ENV_ALIASES.get(var)
        if not os.getenv(var) and not (alias and os.getenv(alias)):
            missing.append(var)

    if missing:
        print("ERROR: The following required environment variables are not set:")
        for v in missing:
            alias = _ENV_ALIASES.get(v)
            if alias:
                print(f"  {v}  (or {alias})")
            else:
                print(f"  {v}")
        print("\nSet them in your .env file or shell environment before running.")
        sys.exit(1)


PHASES = {
    1: "Ingest OHLCV prices              (ingestion.ingest_prices)",
    2: "News sentiment source            (GDELT default, Finnhub optional)",
    3: "Feature engineering              (pipeline.features)",
    4: "Build prediction targets         (pipeline.targets)",
    5: "Ridge baseline                   (models.ridge_baseline)",
    6: "Transformer regressor            (models.transformer_model)",
    7: "PPO RL trading agents            (models.rl_agent)",
}


# ── Phase runners ──────────────────────────────────────────────────────────────

def _banner(n: int) -> None:
    logger.info("")
    logger.info("=" * 65)
    logger.info(f"  STEP {n} — {PHASES[n]}")
    logger.info("=" * 65)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s // 60)}m {int(s % 60)}s"


def run_phase(n: int, fn) -> tuple[bool, object]:
    _banner(n)
    t0 = time.time()
    try:
        result = fn()
        logger.success(f"Step {n} finished in {_elapsed(t0)}")
        return True, result
    except Exception as e:
        logger.error(f"Step {n} FAILED after {_elapsed(t0)}: {e}")
        return False, None


# ── Individual step launchers ──────────────────────────────────────────────────

def step1():
    from ingestion.ingest_prices import run
    results = run()
    total   = sum(results.values()) if isinstance(results, dict) else 0
    logger.info(f"Price bars saved: {total:,}")
    return results


def step2(days: int = 365, use_finnhub_news: bool = False):
    if not use_finnhub_news:
        logger.info("Finnhub skipped by default. Using GDELT sentiment features instead.")
        return {}

    from ingestion.ingest_news_finnhub import run
    results = run(days=days)
    total = sum(results.values()) if isinstance(results, dict) else 0
    logger.info(f"Finnhub news articles submitted: {total:,}")
    return results


def step3(use_gdelt_sentiment: bool = False):
    from pipeline.features import run
    results = run(use_gdelt_sentiment=use_gdelt_sentiment)
    total   = sum(results.values())
    logger.info(f"Feature rows saved: {total:,}")
    return results


def step4():
    from pipeline.targets import run
    results = run()
    total   = sum(results.values())
    logger.info(f"Target rows saved: {total:,}")
    return results


def step5(force_retrain: bool = False):
    from models.ridge_baseline import run
    results = run(force_retrain=force_retrain)
    for asset, res in results.items():
        r = res.get("ridge", {})
        if r and not r.get("skipped"):
            t = r.get("test", {})
            if t:
                logger.info(
                    f"  {asset}: MAE={t.get('mae', float('nan')):.6f}  "
                    f"DirAcc={t.get('directional_accuracy', float('nan')):.3f}"
                )
    return results


def step6(force_retrain: bool = False, use_gdelt_sentiment: bool = False,
          debug: bool = False):
    from models.transformer_model import run
    results = run(
        force_retrain=force_retrain,
        use_gdelt_sentiment=use_gdelt_sentiment,
        debug=debug,
    )
    for asset, res in results.items():
        t = res.get("test", {})
        if t:
            logger.info(
                f"  {asset} Transformer: "
                f"MAE={t.get('mae', float('nan')):.6f}  "
                f"RMSE={t.get('rmse', float('nan')):.6f}  "
                f"R2={t.get('r2', float('nan')):.4f}  "
                f"Corr={t.get('pearson_corr', float('nan')):.4f}  "
                f"DirAcc={t.get('directional_accuracy', float('nan')):.3f}  "
                f"DerivedAcc={t.get('derived_accuracy', float('nan')):.3f}  "
                f"DerivedF1={t.get('derived_f1_macro', float('nan')):.3f}"
            )
    return results


def step7(timesteps: int = 200_000, force_retrain: bool = False,
          use_gdelt_sentiment: bool = False, assets: list = None,
          target_horizon: int = 4, target_mode: str = "vol_norm",
          leverage: float = 1.0):
    from models.rl_agent import run
    results = run(
        assets=assets,
        total_timesteps=timesteps,
        force_retrain=force_retrain,
        use_gdelt_sentiment=use_gdelt_sentiment,
        target_horizon=target_horizon,
        target_mode=target_mode,
        leverage=leverage,
    )
    for asset, res in results.items():
        rl  = res.get("rl", {})
        bnh = res.get("buy_and_hold", {})
        if rl:
            logger.info(
                f"  {asset} RL: "
                f"ret={rl.get('cumulative_return', float('nan')):.2%}  "
                f"Sharpe={rl.get('sharpe_ratio', float('nan')):.3f}  "
                f"MaxDD={rl.get('max_drawdown', float('nan')):.2%}  "
                f"WinRate={rl.get('win_rate', float('nan')):.2%}  "
                f"Trades={rl.get('n_trades', 'N/A')}  "
                f"B&H={bnh.get('cumulative_return', float('nan')):.2%}"
            )
    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(phase_results: dict) -> None:
    logger.info("")
    logger.info("=" * 65)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 65)
    for n, (ok, _) in sorted(phase_results.items()):
        status = "OK" if ok else "FAILED"
        logger.info(f"  Step {n}  {PHASES[n][:48]:<48}  {status}")

    from pathlib import Path
    saved = Path(__file__).parent / "models" / "saved"
    if saved.exists():
        artefacts = sorted(saved.iterdir())
        if artefacts:
            logger.info("")
            logger.info("Saved artefacts:")
            for f in artefacts:
                logger.info(f"  {f.name}")
    logger.info("=" * 65)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Verify required env vars before doing anything
    check_env_vars()

    parser = argparse.ArgumentParser(description="MarketInsight AI — full pipeline")
    parser.add_argument("--from",       dest="from_step", type=int, default=1,
                        help="Start from this step number (1–7, default 1)")
    parser.add_argument("--only",       dest="only_step", type=int, default=None,
                        help="Run only this step number")
    parser.add_argument("--no-rl",      action="store_true",
                        help="Skip step 7 (RL training is slow)")
    parser.add_argument("--no-ingest",  action="store_true",
                        help="Skip steps 1 and 2 (price + news already ingested)")
    parser.add_argument("--news-days",  type=int, default=365,
                        help="Days of Finnhub history to backfill (default 365)")
    parser.add_argument("--timesteps",  type=int, default=200_000,
                        help="PPO timesteps per asset (default 200,000)")
    parser.add_argument("--retrain",    nargs="*", default=[],
                        metavar="MODEL",
                        help="Which models to retrain: ridge transformer rl all")
    parser.add_argument("--retrain-all", action="store_true",
                        help="Retrain every model from scratch")
    parser.add_argument("--use-gdelt-sentiment", action="store_true",
                        help="Include GDELT FinBERT sentiment features")
    parser.add_argument("--debug-transformer", action="store_true",
                        help="Fast Transformer smoke test: 2 epochs, small rows, small window")
    parser.add_argument(
        "--use-finnhub-news",
        action="store_true",
        help="Run old Finnhub news ingestion. Disabled by default.",
    )
    parser.add_argument("--asset",          default=None,
                        help="BTC|ETH|GOLD|OIL — run step 7 for this asset only")
    parser.add_argument("--target-horizon", type=int, default=4, choices=[1, 4],
                        help="Transformer prediction horizon for RL (default: 4)")
    parser.add_argument("--target-mode",    default="vol_norm",
                        choices=["raw", "vol_norm"],
                        help="Transformer target mode for RL (default: vol_norm)")
    parser.add_argument("--leverage",       type=float, default=1.0,
                        help="RL leverage multiplier (default: 1.0)")
    parser.add_argument("--debug-rl",       action="store_true",
                        help="Pass --debug-rl to RL agent (10,000 timesteps unless --timesteps set)")

    args = parser.parse_args()


    retrain = args.retrain or []
    force_ridge       = "all" in retrain or "ridge"       in retrain or args.retrain_all
    force_transformer = "all" in retrain or "transformer" in retrain or args.retrain_all
    force_rl          = "all" in retrain or "rl"          in retrain or args.retrain_all
    use_gdelt         = args.use_gdelt_sentiment

    if use_gdelt and (force_transformer or force_rl):
        logger.info(
            "GDELT sentiment enabled — run smoke tests first:\n"
            "  python -m models.gdelt_sentiment --finbert-test\n"
            "  python -m models.gdelt_sentiment --mock-test\n"
            "  python -m models.gdelt_sentiment --smoke-test --assets BTC GOLD "
            "--start-date 2026-04-20 --end-date 2026-04-22 --max-records-per-query 5 --no-full-text"
        )

    def should_run(n: int) -> bool:
        if args.only_step:
            return n == args.only_step
        return n >= args.from_step

    t_total       = time.time()
    phase_results: dict[int, tuple[bool, object]] = {}

    rl_assets = [args.asset] if args.asset else None
    rl_timesteps = args.timesteps
    if args.debug_rl and args.timesteps == 200_000:
        rl_timesteps = 10_000

    runners = {
        1: step1,
        2: lambda: step2(days=args.news_days, use_finnhub_news=args.use_finnhub_news),
        3: lambda: step3(use_gdelt_sentiment=use_gdelt),
        4: step4,
        5: lambda: step5(force_retrain=force_ridge),
        6: lambda: step6(force_retrain=force_transformer, use_gdelt_sentiment=use_gdelt,
                         debug=args.debug_transformer),
        7: lambda: step7(
                rl_timesteps, force_retrain=force_rl,
                use_gdelt_sentiment=use_gdelt,
                assets=rl_assets,
                target_horizon=args.target_horizon,
                target_mode=args.target_mode,
                leverage=args.leverage,
           ),
    }

    skip = set()
    if args.no_ingest:
        skip.add(1)
        skip.add(2)
    if args.no_rl:
        skip.add(7)

    for n, fn in runners.items():
        if not should_run(n):
            continue
        if n in skip:
            logger.info(f"Skipping step {n} ({PHASES[n].split('(')[0].strip()})")
            continue
        ok, result = run_phase(n, fn)
        phase_results[n] = (ok, result)
        # If --asset is set and this step fails, stop immediately
        if not ok and args.asset and n == 7:
            logger.error(
                f"Step {n} failed for --asset {args.asset}. Stopping pipeline."
            )
            print_summary(phase_results)
            sys.exit(1)

    print_summary(phase_results)
    logger.info(f"Total wall time: {_elapsed(t_total)}")



if __name__ == "__main__":
    main()
