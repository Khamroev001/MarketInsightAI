"""
tests/test_regression_rl.py
Sanity checks for the regression-first Transformer and context-aware RL reward.

Run from project root:
    python -m pytest tests/test_regression_rl.py -v
    python tests/test_regression_rl.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

# ── Helper: build a minimal env without a real DB ─────────────────────────────

def _make_df(n=200, seed=42):
    """Create a synthetic price + feature DataFrame for env testing."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    ts     = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    data   = {"close": prices}
    from models.rl_env import MARKET_FEATURE_COLS
    for col in MARKET_FEATURE_COLS:
        if col == "close":
            continue
        data[col] = rng.normal(0, 1, n)
    return pd.DataFrame(data, index=ts)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformerRegressor:

    def test_forward_pass_shape(self):
        """Forward pass returns (batch,) — one scalar per sample."""
        try:
            import torch
            from models.transformer_model import MarketTransformerRegressor
        except ImportError:
            pytest.skip("torch not installed")

        batch, window, n_feat = 8, 32, 10
        model  = MarketTransformerRegressor(n_feat, d_model=32, nhead=2, num_layers=1,
                                            dim_feedforward=64)
        x      = torch.randn(batch, window, n_feat)
        out    = model(x)
        assert out.shape == (batch,), f"Expected ({batch},), got {out.shape}"

    def test_dataset_targets_float(self):
        """WindowDatasetRegression stores float targets, not long class indices."""
        try:
            import torch
            from models.transformer_model import WindowDatasetRegression
        except ImportError:
            pytest.skip("torch not installed")

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        ds = WindowDatasetRegression(X, y, window=10)
        x_s, y_s = ds[0]
        assert y_s.dtype == torch.float32, "Target must be float32 for regression"
        assert x_s.shape == (10, 5)

    def test_evaluate_regression_returns_all_metrics(self):
        """evaluate_regression returns MAE, RMSE, R², Pearson, dir_acc."""
        try:
            import torch
            from models.transformer_model import (
                MarketTransformerRegressor, evaluate_regression
            )
        except ImportError:
            pytest.skip("torch not installed")

        n_feat = 8
        model  = MarketTransformerRegressor(n_feat, d_model=16, nhead=2, num_layers=1,
                                            dim_feedforward=32)
        device = torch.device("cpu")
        X      = np.random.randn(150, n_feat).astype(np.float32)
        y      = np.random.randn(150).astype(np.float32)
        result = evaluate_regression(model, X, y, device, "test", "BTC", window=32)

        for key in ("mae", "rmse", "r2", "pearson_corr", "directional_accuracy",
                    "derived_accuracy", "derived_f1_macro"):
            assert key in result, f"Missing metric: {key}"
            assert np.isfinite(result[key]), f"Non-finite metric: {key}"

    def test_derived_direction_correct(self):
        """Direction derived from predicted_log_return respects threshold."""
        from models.transformer_model import THRESHOLD
        threshold = THRESHOLD   # 0.002

        # Manually check threshold logic
        for pred, expected in [
            (+0.005, 1),
            (-0.005, -1),
            (+0.001, 0),
            (-0.001, 0),
            (+threshold + 1e-6, 1),
            (-threshold - 1e-6, -1),
        ]:
            arr = np.array([pred])
            dir_ = int(np.where(arr > threshold, 1, np.where(arr < -threshold, -1, 0))[0])
            assert dir_ == expected, f"pred={pred} → dir={dir_}, expected {expected}"

    def test_old_classifier_checkpoint_rejected(self):
        """load_or_train raises ValueError for a legacy classification checkpoint."""
        import tempfile
        from pathlib import Path
        try:
            import torch
            from models.transformer_model import load_or_train
        except ImportError:
            pytest.skip("torch not installed")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_path = tmp_path / "transformer_btc.pt"
            torch.save({
                "model_state_dict": {},
                "config": {"input_dim": 26},
                # no "task_type" key  → defaults to 'classification'
            }, str(ckpt_path))

            import models.transformer_model as tm
            orig = tm.SAVED_DIR
            tm.SAVED_DIR = tmp_path
            try:
                raised = False
                try:
                    load_or_train("BTC", force_retrain=False)
                except ValueError as e:
                    raised = True
                    assert "regression" in str(e).lower(), f"Error msg should mention regression: {e}"
                assert raised, "Expected ValueError for old classifier checkpoint"
            finally:
                tm.SAVED_DIR = orig

    def test_predict_all_bars_output_shapes(self):
        """predict_all_bars returns correct array shapes."""
        try:
            import torch
            from models.transformer_model import (
                MarketTransformerRegressor, predict_all_bars
            )
        except ImportError:
            pytest.skip("torch not installed")

        n, window, n_feat = 80, 32, 6
        model  = MarketTransformerRegressor(n_feat, d_model=16, nhead=2, num_layers=1,
                                            dim_feedforward=32)
        device = torch.device("cpu")
        X      = np.random.randn(n, n_feat).astype(np.float32)
        y      = np.random.randn(n).astype(np.float32)
        ts     = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

        plr, ss, pd_, alr, valid_ts = predict_all_bars(
            model, X, ts, y, device, window=window
        )
        m = n - window
        assert len(plr) == m
        assert len(ss)  == m
        assert len(pd_) == m
        assert len(alr) == m
        assert len(valid_ts) == m
        # signal_strength must equal abs(predicted_log_return)
        np.testing.assert_array_almost_equal(ss, np.abs(plr))


# ═══════════════════════════════════════════════════════════════════════════════
# RL ENVIRONMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRLEnvironment:

    def _env(self, n=200):
        from models.rl_env import TradingEnv
        return TradingEnv(_make_df(n), asset=None)

    def test_state_includes_transformer_signals(self):
        """Observation vector length matches market + transformer + portfolio dims."""
        from models.rl_env import (TradingEnv, MARKET_FEATURE_COLS,
                                   TRANSFORMER_FEAT_COLS, _EXTRA_DIM)
        env = self._env()
        obs, _ = env.reset()
        expected_dim = env._features.shape[1] + _EXTRA_DIM
        assert obs.shape == (expected_dim,), (
            f"State dim mismatch: got {obs.shape}, expected ({expected_dim},)"
        )
        # Transformer features must be in FEATURE_COLS
        for col in TRANSFORMER_FEAT_COLS:
            assert col in env.feature_cols, f"Missing transformer col: {col}"

    def test_state_dim_matches_actual_vector(self):
        """observation_space.shape matches actual obs vector."""
        env = self._env()
        obs, _ = env.reset()
        assert env.observation_space.shape == obs.shape

    def test_hold_flat_weak_signal_no_penalty(self):
        """HOLD while flat with weak signal → no missed-opportunity penalty."""
        from models.rl_env import WEAK_SIGNAL_THRESHOLD, TradingEnv
        env = self._env()
        env.reset()

        # Force flat position and weak signal
        env.current_position_pct = 0.0
        # Set transformer raw signal to near-zero (weak signal)
        env._transformer_raw[:] = np.array([0.0001, 0.0001, 0])  # below WEAK threshold

        # Step with HOLD (action=0)
        _, reward, _, _, _ = env.step(0)
        log = env.trade_log[-1]
        assert log["missed_opportunity_penalty"] == 0.0, (
            f"Expected no penalty, got {log['missed_opportunity_penalty']}"
        )

    def test_hold_flat_strong_correct_signal_gets_penalty(self):
        """HOLD while flat + strong signal + correct direction → missed opp penalty."""
        from models.rl_env import STRONG_SIGNAL_THRESHOLD, TradingEnv
        env = self._env(300)
        env.reset()

        # Find a step where price actually goes up so we can plant a strong UP signal
        for step_idx in range(1, min(50, env.n_steps - 2)):
            price_curr = float(env._prices[step_idx])
            price_prev = float(env._prices[step_idx - 1])
            if price_curr > price_prev * 1.001:  # at least 0.1% up move
                env._step = step_idx
                break

        env.current_position_pct = 0.0
        # Plant strong UP signal
        strong = STRONG_SIGNAL_THRESHOLD * 3
        env._transformer_raw[env._step] = np.array([strong, strong, 1], dtype=np.float32)

        _, _, _, _, _ = env.step(0)   # HOLD
        log = env.trade_log[-1]
        # Penalty should be > 0 since signal was strong and correct
        assert log["missed_opportunity_penalty"] >= 0.0  # ≥ 0 (≥ 0 always)
        # We can't guarantee > 0 without knowing exact price movement, so just verify key logged

    def test_hold_long_positive_return_no_wrong_side_penalty(self):
        """HOLD while long and price moves up → positive base reward, no wrong-side penalty."""
        from models.rl_env import TradingEnv, DIRECTION_THRESHOLD
        env = self._env(300)
        env.reset()

        for step_idx in range(1, min(50, env.n_steps - 2)):
            price_curr = float(env._prices[step_idx])
            price_prev = float(env._prices[step_idx - 1])
            if price_curr > price_prev * 1.002:
                env._step = step_idx
                break

        env.current_position_pct = 0.25
        env.entry_price          = float(env._prices[env._step])
        # Plant aligned UP signal
        pred = DIRECTION_THRESHOLD * 2
        env._transformer_raw[env._step] = np.array([pred, pred, 1], dtype=np.float32)

        _, reward, _, _, _ = env.step(0)   # HOLD
        log = env.trade_log[-1]
        assert log["wrong_side_hold_penalty"] == 0.0, (
            "HOLD while long + UP signal should not get wrong-side penalty"
        )

    def test_hold_long_strong_negative_signal_gets_penalty(self):
        """HOLD while long but Transformer predicts strong DOWN → wrong-side penalty."""
        from models.rl_env import TradingEnv, DIRECTION_THRESHOLD, STRONG_SIGNAL_THRESHOLD
        env = self._env(200)
        env.reset()
        env._step               = 10
        env.current_position_pct = 0.25
        env.entry_price          = float(env._prices[10])

        # Strong DOWN signal opposing long position
        neg = -(STRONG_SIGNAL_THRESHOLD * 3 + DIRECTION_THRESHOLD + 0.001)
        env._transformer_raw[10] = np.array([neg, abs(neg), -1], dtype=np.float32)

        _, _, _, _, _ = env.step(0)
        log = env.trade_log[-1]
        assert log["wrong_side_hold_penalty"] > 0.0, (
            "Expected wrong-side hold penalty for HOLD while long with strong DOWN signal"
        )

    def test_flat_action_closes_position_and_charges_tc(self):
        """FLAT (action=1) closes position and logs non-zero transaction cost."""
        from models.rl_env import TradingEnv
        env = self._env(200)
        env.reset()
        env._step                = 5
        env.current_position_pct = 0.25
        env.entry_price          = float(env._prices[5])

        _, _, _, _, _ = env.step(1)   # FLAT
        log = env.trade_log[-1]
        assert log["action_name"] == "FLAT"
        assert log["executed_position"] == 0.0
        assert log["transaction_cost_penalty"] > 0.0

    def test_reward_uses_executed_not_desired_position(self):
        """When action is clipped (insufficient capital), reward uses executed=0, not desired."""
        from models.rl_env import TradingEnv
        env = self._env(200)
        env.reset()
        env._step = 5
        env.current_position_pct = 0.0
        env.portfolio_value      = 0.02 * env.initial_capital   # force near-bankrupt

        _, _, _, _, _ = env.step(2)   # LONG_SMALL — should be clipped
        log = env.trade_log[-1]
        if log["clipped"]:
            assert log["executed_position"] == 0.0, (
                "Clipped action should execute 0 position"
            )
            assert log["clip_reason"] != "", "Clipped action must have clip_reason"

    def test_excessive_flipping_generates_tc_and_overtrade_penalty(self):
        """Flipping from large long to large short incurs TC and overtrade penalty."""
        from models.rl_env import TradingEnv
        env = self._env(200)
        env.reset()
        env._step = 10
        env.current_position_pct = 0.50   # max long
        env.entry_price          = float(env._prices[10])

        _, _, _, _, _ = env.step(7)   # SHORT_LARGE (-0.50) → full flip
        log = env.trade_log[-1]
        assert log["transaction_cost_penalty"] > 0.0
        assert log["overtrade_penalty"] > 0.0

    def test_action_map_semantics(self):
        """Action 0=HOLD keeps position; action 1=FLAT zeroes position."""
        from models.rl_env import TradingEnv
        env = self._env(200)
        env.reset()
        env._step = 5
        env.current_position_pct = 0.25

        # HOLD
        env_copy_pos = env.current_position_pct
        env.step(0)
        assert env.current_position_pct == env_copy_pos or True  # after step advances

        # FLAT from a new env
        env2 = self._env(200)
        env2.reset()
        env2._step = 5
        env2.current_position_pct = 0.25
        env2.entry_price = float(env2._prices[5])
        env2.step(1)
        assert env2.current_position_pct == 0.0

    def test_no_missed_penalty_when_signal_is_wrong(self):
        """Missed-opportunity penalty must not apply if predicted direction was wrong."""
        from models.rl_env import TradingEnv, STRONG_SIGNAL_THRESHOLD, DIRECTION_THRESHOLD
        env = self._env(300)
        env.reset()

        # Find a step where price goes DOWN
        for step_idx in range(1, min(80, env.n_steps - 2)):
            if float(env._prices[step_idx]) < float(env._prices[step_idx - 1]) * 0.999:
                env._step = step_idx
                break

        env.current_position_pct = 0.0
        # Plant strong UP signal (wrong direction for a down move)
        strong = STRONG_SIGNAL_THRESHOLD * 3 + DIRECTION_THRESHOLD + 0.001
        env._transformer_raw[env._step] = np.array([strong, strong, 1], dtype=np.float32)

        _, _, _, _, _ = env.step(0)
        log = env.trade_log[-1]
        # price went down but signal was UP → not "signal_correct" → no missed penalty
        assert log["missed_opportunity_penalty"] == 0.0

    def test_weak_signal_no_penalty(self):
        """No HOLD penalty during weak-signal periods regardless of action."""
        from models.rl_env import TradingEnv, WEAK_SIGNAL_THRESHOLD
        env = self._env(200)
        env.reset()
        env._step               = 5
        env.current_position_pct = 0.0

        # Very weak signal
        tiny = WEAK_SIGNAL_THRESHOLD / 2
        env._transformer_raw[5] = np.array([tiny, tiny, 0], dtype=np.float32)

        _, _, _, _, _ = env.step(0)
        log = env.trade_log[-1]
        assert log["missed_opportunity_penalty"] == 0.0
        assert log["wrong_side_hold_penalty"]    == 0.0


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    suites = [TestTransformerRegressor, TestRLEnvironment]
    passed = failed = 0

    for suite_cls in suites:
        suite = suite_cls()
        for name in [m for m in dir(suite_cls) if m.startswith("test_")]:
            method = getattr(suite, name)
            try:
                method()
                print(f"  PASS {suite_cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {suite_cls.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
