"""
models/model_status.py
Run this anytime to see which models are trained and when.
Usage: python -m models.model_status
"""

import os
import json
from datetime import datetime
from pathlib import Path

ASSETS    = ["BTC", "ETH", "GOLD", "OIL"]
SAVED_DIR = Path(__file__).parent / "saved"


def check_status() -> None:
    print("\n" + "=" * 65)
    print("  MarketInsight AI - Model Status")
    print("=" * 65)

    for asset in ASSETS:
        print(f"\n  {asset}")
        print(f"  {'-' * 40}")

        # Ridge
        p = SAVED_DIR / f"ridge_{asset.lower()}.pkl"
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            print(f"  Ridge:       OK  saved {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"  Ridge:       --  not trained")

        # Transformer
        p = SAVED_DIR / f"transformer_{asset.lower()}.pt"
        if p.exists():
            try:
                import torch
                ckpt = torch.load(str(p), map_location="cpu")
                test_acc   = ckpt.get("test_accuracy", "?")
                trained_at = ckpt.get("trained_at",    "?")
                acc_str    = f"{test_acc:.3f}" if isinstance(test_acc, float) else str(test_acc)
                ts_str     = trained_at[:16] if isinstance(trained_at, str) else str(trained_at)
                print(f"  Transformer: OK  acc={acc_str}  trained={ts_str}")
            except Exception as e:
                print(f"  Transformer: OK  (could not read metadata: {e})")
        else:
            print(f"  Transformer: --  not trained")

        # PPO
        p      = SAVED_DIR / f"ppo_{asset.lower()}.zip"
        meta_p = SAVED_DIR / f"ppo_{asset.lower()}_metadata.json"
        if p.exists():
            meta: dict = {}
            if meta_p.exists():
                try:
                    with open(meta_p) as f:
                        meta = json.load(f)
                except Exception:
                    pass
            sharpe    = meta.get("sharpe",      "?")
            win_rate  = meta.get("win_rate",    "?")
            trained   = meta.get("trained_at",  "?")
            sharpe_s  = f"{sharpe:.3f}"   if isinstance(sharpe,   float) else str(sharpe)
            wr_s      = f"{win_rate:.1%}" if isinstance(win_rate, float) else str(win_rate)
            ts_s      = trained[:16]      if isinstance(trained,  str)   else str(trained)
            print(f"  PPO RL:      OK  sharpe={sharpe_s}  win_rate={wr_s}  trained={ts_s}")
        else:
            print(f"  PPO RL:      --  not trained")

        # Scaler
        p = SAVED_DIR / f"feature_scaler_{asset.lower()}.pkl"
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            print(f"  Scaler:      OK  saved {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"  Scaler:      --  not fitted")

    print("\n" + "=" * 65)
    print("  To force retrain: pass --retrain flag to run_pipeline.py")
    print("    python run_pipeline.py --retrain rl")
    print("    python run_pipeline.py --retrain transformer")
    print("    python run_pipeline.py --retrain-all")
    print("=" * 65)
    print()


if __name__ == "__main__":
    check_status()
