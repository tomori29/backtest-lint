# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from backtest_lint import checks
from backtest_lint.checks import Bundle, run_all
from backtest_lint.loader import load_bundle
from backtest_lint.metrics import ann_arithmetic, ann_geometric


def make_bundle(n=300, mu_s=0.0008, mu_b=0.0004, vol=0.008, seed=0, meta=None):
    rng = np.random.default_rng(seed)
    rs = rng.normal(mu_s, vol, n)
    rb = rng.normal(mu_b, vol, n)
    df = pd.DataFrame({"date": pd.bdate_range("2021-01-04", periods=n).strftime("%Y-%m-%d"),
                       "strategy": rs, "benchmark": rb})
    base = {
        "universe_mode": "pit",
        "delisted_included": True,
        "benchmark_type": "buy_hold",
        "cost": {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001,
                 "applied_to_benchmark": False},
        "phases_run": 21,
        "panel_fixed": True,
        "panel_trail_history": [2700],
        "n_trials": 1,
        "signal_uses": "close_t",
        "trade_price": "open_t+1",
        "reported_t": 3.0,
        "reported_ann": float(ann_geometric(rs)),
        "reported_bench_ann": float(ann_geometric(rb)),
    }
    if meta:
        base.update(meta)
    return Bundle(daily=df, meta=base)


def status(results, cid):
    return next(r["status"] for r in results if r["id"] == cid)


def test_good_bundle_all_pass():
    rs = run_all(make_bundle())
    assert not any(r["status"] == "FAIL" for r in rs)
    assert not any(r["status"] == "WARN" for r in rs)


def test_bad_example_blocks():
    b = load_bundle("examples/bad")
    rs = run_all(b)
    fails = [r["id"] for r in rs if r["status"] == "FAIL"]
    assert set(fails) == {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"}


def test_good_example_passes():
    b = load_bundle("examples/good")
    rs = run_all(b)
    assert not any(r["status"] == "FAIL" for r in rs)
    assert not any(r["status"] == "WARN" for r in rs)


def test_static_universe_fails():
    b = make_bundle(meta={"universe_mode": "static"})
    assert status(run_all(b), "E3") == "FAIL"


def test_unknown_universe_fails():
    b = make_bundle(meta={"universe_mode": "unknown"})
    assert status(run_all(b), "E3") == "FAIL"


def test_single_phase_fails():
    b = make_bundle(meta={"phases_run": 1})
    assert status(run_all(b), "E4") == "FAIL"


def test_few_phases_warn():
    b = make_bundle(meta={"phases_run": 3})
    assert status(run_all(b), "E4") == "WARN"


def test_panel_drift_fails():
    b = make_bundle(meta={"panel_fixed": False})
    assert status(run_all(b), "E5") == "FAIL"
    b2 = make_bundle(meta={"panel_trail_history": [1040, 2700]})
    assert status(run_all(b2), "E5") == "FAIL"


def test_ew_daily_benchmark_zero_cost_fails():
    b = make_bundle(meta={"benchmark_type": "ew_daily"})
    assert status(run_all(b), "E2") == "FAIL"


def test_buy_hold_zero_cost_passes():
    b = make_bundle(meta={"benchmark_type": "buy_hold"})
    assert status(run_all(b), "E2") == "PASS"


def test_multiple_testing_bonferroni():
    b = make_bundle(meta={"n_trials": 18, "reported_t": 2.0})
    assert status(run_all(b), "E7") == "FAIL"
    b2 = make_bundle(meta={"n_trials": 18, "reported_t": 3.5})
    assert status(run_all(b2), "E7") == "PASS"


def test_annualization_mismatch_fails():
    b = make_bundle(mu_s=0.0015)
    ari = ann_arithmetic(b.daily["strategy"])
    b.meta["reported_ann"] = float(ari)
    geo = ann_geometric(b.daily["strategy"])
    if abs(ari - geo) > 0.005:
        assert status(run_all(b), "E6") == "FAIL"
    else:  # 若样本波动太小，构造一个人为偏差
        b.meta["reported_ann"] = float(geo) + 0.03
        assert status(run_all(b), "E6") == "FAIL"


def test_benchmark_arithmetic_detected():
    b = make_bundle(mu_b=0.0015)
    ari = ann_arithmetic(b.daily["benchmark"])
    b.meta["reported_bench_ann"] = float(ari)
    geo = ann_geometric(b.daily["benchmark"])
    if abs(ari - geo) > 0.005:
        assert status(run_all(b), "E1") == "FAIL"
    else:
        b.meta["reported_bench_ann"] = float(geo) + 0.03
        assert status(run_all(b), "E1") == "FAIL"


def test_survivorship_missing_warns():
    b = make_bundle()
    del b.meta["delisted_included"]
    assert status(run_all(b), "E8") == "WARN"


def test_extreme_return_warns():
    b = make_bundle()
    b.daily.loc[10, "strategy"] = 0.8
    assert status(run_all(b), "S1") == "WARN"


def test_missing_benchmark_fails():
    df = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "strategy": [0.0, 0.001]})
    b = Bundle(daily=df, meta={})
    rs = run_all(b)
    assert status(rs, "B0") == "FAIL"
    assert status(rs, "S2") == "FAIL"


def test_sample_days_mismatch_fails():
    b = make_bundle()
    b.meta["sample_days"] = len(b.daily) + 120
    assert status(run_all(b), "E6") == "FAIL"


def test_sample_days_match_passes():
    b = make_bundle()
    b.meta["sample_days"] = len(b.daily)
    assert status(run_all(b), "E6") == "PASS"


def test_benchmark_extreme_fails():
    b = make_bundle()
    b.daily.loc[10, "benchmark"] = 0.8
    assert status(run_all(b), "S3") == "FAIL"


def test_benchmark_large_warns():
    b = make_bundle()
    b.daily.loc[10, "benchmark"] = 0.25
    assert status(run_all(b), "S3") == "WARN"


# ---------------- E9 成交时点可执行性（2026-11-12） ----------------

def test_e9_missing_meta_warns():
    b = make_bundle()
    b.meta.pop("signal_uses", None)
    b.meta.pop("trade_price", None)
    assert status(run_all(b), "E9") == "WARN"


def test_e9_same_time_close_fails():
    b = make_bundle()
    b.meta["signal_uses"] = "close_t"
    b.meta["trade_price"] = "close_t"
    assert status(run_all(b), "E9") == "FAIL"


def test_e9_intraday_signal_close_trade_fails():
    b = make_bundle()
    b.meta["signal_uses"] = "high_t"
    b.meta["trade_price"] = "open_t"
    assert status(run_all(b), "E9") == "FAIL"


def test_e9_next_open_passes():
    b = make_bundle()
    b.meta["signal_uses"] = "close_t"
    b.meta["trade_price"] = "open_t+1"
    assert status(run_all(b), "E9") == "PASS"


def test_cli_print_survives_gbk_console(monkeypatch):
    """Windows 控制台默认 GBK；报告里的 ✅/⚠️/❌ 不能把 CLI 打崩。"""
    import io
    import sys

    from backtest_lint import cli

    buf = io.BytesIO()
    fake = io.TextIOWrapper(buf, encoding="gbk", errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", fake)
    cli._print("✅ PASS  ⚠️ WARN  ❌ FAIL")
    fake.flush()
    assert buf.getvalue()


def test_demo_shows_measurement_inflation():
    """demo 的意义：同一份零 alpha 数据，原样口径能做出正超额，修完就没了。"""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "demo" / "run_demo.py"
    spec = importlib.util.spec_from_file_location("demo_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    steps, _best_key, _ctx = mod.compute_steps(mod.make_market())
    assert len(steps) == 6
    naive, corrected = steps[0][1], steps[-1][1]
    assert naive > 0.02, f"原样口径应当虚增出正超额，实得 {naive:.4f}"
    assert corrected < naive - 0.04, f"修正后应明显下降，实得 {corrected:.4f}"
