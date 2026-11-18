# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from backtest_lint import checks
from backtest_lint.checks import Bundle, run_all
from backtest_lint.loader import load_bundle
from backtest_lint.metrics import ann_arithmetic, ann_geometric


def make_bundle(n=300, mu_s=0.0012, mu_b=0.0004, vol=0.008, seed=0, meta=None):
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
        "size_variable": "total_shares * close",
        "size_neutralized": False,
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
    assert set(fails) == {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"}


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


# ---------------- E13 规模变量口径（2026-11-18） ----------------

def test_e13_missing_size_variable_warns():
    b = make_bundle()
    b.meta.pop("size_variable", None)
    b.meta.pop("size_neutralized", None)
    assert status(run_all(b), "E13") == "WARN"


def test_e13_neutralized_without_variable_fails():
    b = make_bundle()
    b.meta.pop("size_variable", None)
    b.meta["size_neutralized"] = True
    assert status(run_all(b), "E13") == "FAIL"


def test_e13_price_as_size_fails():
    """收盘价当市值——本项目 2026-11-18 真实踩过的坑。"""
    b = make_bundle()
    b.meta["size_variable"] = "close"
    b.meta["size_neutralized"] = True
    assert status(run_all(b), "E13") == "FAIL"


def test_e13_adj_close_as_size_fails():
    b = make_bundle()
    b.meta["size_variable"] = "adj_close"
    assert status(run_all(b), "E13") == "FAIL"


def test_e13_chinese_price_fails():
    b = make_bundle()
    b.meta["size_variable"] = "收盘价"
    assert status(run_all(b), "E13") == "FAIL"


def test_e13_shares_times_close_passes():
    """总股本×收盘同时含股本与价格字样，应判 PASS。"""
    b = make_bundle()
    b.meta["size_variable"] = "total_shares * close"
    b.meta["size_neutralized"] = True
    assert status(run_all(b), "E13") == "PASS"


def test_e13_market_cap_passes():
    b = make_bundle()
    b.meta["size_variable"] = "market_cap"
    assert status(run_all(b), "E13") == "PASS"


def test_e13_share_price_is_price_not_cap():
    """share_price 是价格，不能被 "share" 子串误判成市值。"""
    b = make_bundle()
    b.meta["size_variable"] = "share_price"
    b.meta["size_neutralized"] = True
    assert status(run_all(b), "E13") == "FAIL"


def test_e13_bare_shares_is_not_cap():
    """单写股本/股数不是市值。"""
    b = make_bundle()
    b.meta["size_variable"] = "total_shares"
    assert status(run_all(b), "E13") == "FAIL"


def test_e13_capital_is_unknown_not_cap():
    """capital 只含 cap 子串，不是市值，应 WARN 而非 PASS。"""
    b = make_bundle()
    b.meta["size_variable"] = "capital"
    assert status(run_all(b), "E13") == "WARN"


def test_e2_missing_applied_with_daily_benchmark_fails():
    """基准每日换手却没声明是否扣成本——不能默认「都扣了」。"""
    b = make_bundle()
    b.meta["benchmark_type"] = "ew_daily"
    b.meta["cost"] = {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001}
    assert status(run_all(b), "E2") == "FAIL"


def test_e9_same_token_fails():
    """信号与成交用同一个价格写法，必然是同一时点。"""
    b = make_bundle()
    b.meta["signal_uses"] = "open_t"
    b.meta["trade_price"] = "open_t"
    assert status(run_all(b), "E9") == "FAIL"


def test_e9_unknown_token_warns():
    b = make_bundle()
    b.meta["signal_uses"] = "clsoe_t"
    b.meta["trade_price"] = "open_t+1"
    assert status(run_all(b), "E9") == "WARN"


def test_e10_uses_trials_columns_when_meta_says_one():
    """meta 写 1 次、trials 有 50 列时，必须按 50 次算，否则网格挑最好会假 PASS。"""
    import numpy as np
    import pandas as pd

    b = make_bundle()
    b.meta["n_trials"] = 1
    rng = np.random.default_rng(1)
    b.trials = pd.DataFrame(rng.normal(0, 0.01, (300, 50)))
    detail = next(r["detail"] for r in run_all(b) if r["id"] == "E10")
    assert "50 次尝试" in detail


def test_e11_all_nan_trials_is_not_pass():
    import numpy as np
    import pandas as pd

    b = make_bundle()
    b.trials = pd.DataFrame(np.full((300, 6), np.nan))
    assert status(run_all(b), "E11") != "PASS"


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


# ---------------- E10–E12 搜索规模（2026-11-16） ----------------

def _trials(n_rows=300, n_cols=8, seed=0, edge_col=None, edge_mu=0.003):
    rng = np.random.default_rng(seed)
    R = rng.normal(0, 0.01, size=(n_rows, n_cols))
    R -= R.mean(axis=0)          # 去掉「整段样本的运气」，只留纯噪声
    if edge_col is not None:
        R[:, edge_col] += edge_mu
    return pd.DataFrame(R, columns=[f"t{i}" for i in range(n_cols)])


def test_e10_passes_on_significant_bundle():
    assert status(run_all(make_bundle()), "E10") == "PASS"


def test_e10_fails_on_noise_bundle():
    b = make_bundle(mu_s=0.0)
    assert status(run_all(b), "E10") == "FAIL"


def test_e10_penalises_more_trials():
    """同一份收益，声明试了 200 次就该掉下来。"""
    b = make_bundle(meta={"n_trials": 200})
    assert status(run_all(b), "E10") in ("WARN", "FAIL")


def test_e10_missing_trials_is_info_not_failure():
    b = make_bundle()
    assert status(run_all(b), "E11") == "INFO"
    assert status(run_all(b), "E12") == "INFO"


def test_e11_pbo_flags_pure_noise():
    b = make_bundle()
    b.trials = _trials(seed=3)
    assert status(run_all(b), "E11") in ("WARN", "FAIL")


def test_e11_pbo_passes_when_one_config_really_works():
    b = make_bundle()
    b.trials = _trials(seed=1, edge_col=2)
    assert status(run_all(b), "E11") == "PASS"


def test_e12_cpcv_fails_on_noise_grid():
    b = make_bundle()
    b.trials = _trials(seed=5)
    assert status(run_all(b), "E12") in ("WARN", "FAIL")


def test_e12_cpcv_passes_when_edge_is_stable():
    b = make_bundle()
    b.trials = _trials(seed=1, edge_col=2)
    assert status(run_all(b), "E12") == "PASS"


def test_trials_loaded_from_bundle_dir(tmp_path):
    from backtest_lint.loader import load_bundle

    _trials(n_rows=60, n_cols=4, seed=2).to_csv(tmp_path / "trials.csv", index=False)
    b = make_bundle(n=60)
    b.daily.to_csv(tmp_path / "daily.csv", index=False)
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    loaded = load_bundle(tmp_path)
    assert loaded.trials is not None and loaded.trials.shape == (60, 4)


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


# ---------------- 审计修复的回归测试（2026-11-18） ----------------

def test_benchmark_all_nan_is_not_pass():
    """benchmark 列存在但全 NaN，不能算「基准存在」。"""
    b = make_bundle()
    b.daily["benchmark"] = np.nan
    rs = run_all(b)
    assert status(rs, "B0") == "FAIL"
    assert status(rs, "E1") != "PASS"


def test_all_nan_strategy_annualization_not_pass():
    """策略全 NaN 时年化无法计算，E6 不能报「与几何一致」。"""
    b = make_bundle()
    b.daily["strategy"] = np.nan
    b.meta["reported_ann"] = 0.0
    assert status(run_all(b), "E6") != "PASS"


def test_s1_extreme_threshold_matches_docs():
    """README 写的是单日 >50% 才 WARN；40% 不应触发。"""
    b = make_bundle()
    b.daily.loc[10, "strategy"] = 0.40
    assert status(run_all(b), "S1") == "PASS"
    b.daily.loc[10, "strategy"] = 0.60
    assert status(run_all(b), "S1") == "WARN"


def test_s3_message_reports_the_actual_extreme_value():
    """负向极端日不能被写成「最大 +2%」。"""
    b = make_bundle()
    b.daily.loc[10, "benchmark"] = -0.35
    r = next(x for x in run_all(b) if x["id"] == "S3")
    assert r["status"] == "FAIL"
    assert "-35.00%" in r["detail"]


def test_e7_counts_trials_columns_like_e10():
    """meta 少报 n_trials 时，E7 也必须按 trials.csv 的实际列数校正。"""
    b = make_bundle(meta={"n_trials": 1, "reported_t": 2.5})
    rng = np.random.default_rng(2)
    b.trials = pd.DataFrame(rng.normal(0, 0.01, (300, 50)))
    assert status(run_all(b), "E7") == "FAIL"


def test_e10_short_track_record_message_is_not_nonpositive():
    """正 SR 但样本不够长时，不能误报「这个 SR 不为正」。"""
    rng = np.random.default_rng(3)
    # 逐期 SR≈0.002：正但很小，需要的期数远超 MinTRL 的 20000 上限
    r = np.where(np.arange(400) % 2 == 0, 0.01002, -0.00998)
    df = pd.DataFrame({"date": range(len(r)), "strategy": r,
                       "benchmark": rng.normal(0, 0.01, len(r))})
    b = Bundle(daily=df, meta={"size_variable": "market_cap"})
    res = next(x for x in run_all(b) if x["id"] == "E10")
    assert res["status"] == "FAIL"
    assert "不为正" not in res["detail"] + res["fix"]
    assert "20000 期" in res["fix"]


def test_loader_sorts_non_padded_dates_chronologically(tmp_path):
    (tmp_path / "daily.csv").write_text(
        "date,strategy,benchmark\n"
        "2024/1/10,0.01,0.0\n2024/1/2,0.02,0.0\n2024/1/3,0.03,0.0\n",
        encoding="utf-8")
    b = load_bundle(tmp_path)
    assert b.daily["date"].tolist() == ["2024/1/2", "2024/1/3", "2024/1/10"]


def test_loader_drops_csv_index_column_from_trials(tmp_path):
    """pandas 默认写出的行号列不是候选配置，不能混进 PBO/CPCV。"""
    rng = np.random.default_rng(4)
    pd.DataFrame(rng.normal(0, 0.01, (60, 3)),
                 columns=["cfg0", "cfg1", "cfg2"]).to_csv(tmp_path / "trials.csv")
    pd.DataFrame({"date": range(60), "strategy": rng.normal(0, 0.01, 60),
                  "benchmark": rng.normal(0, 0.01, 60)}).to_csv(
        tmp_path / "daily.csv", index=False)
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    assert load_bundle(tmp_path).trials.shape == (60, 3)


def test_malformed_cost_object_does_not_crash():
    b = make_bundle()
    b.meta["cost"] = [1]                 # 非法 meta 不能让整份报告崩掉
    assert status(run_all(b), "E2") == "FAIL"


def test_sharpe_short_input_is_nan_without_runtime_warning():
    from backtest_lint.metrics import sharpe

    assert np.isnan(sharpe([]))
    assert np.isnan(sharpe([0.01]))


def test_json_report_sanitizes_nan_and_numpy_scalars():
    """meta 里的 NaN/Infinity/np.int64 不能让 JSON 报告崩或变成非法 JSON。"""
    import json

    from backtest_lint.report import to_json

    results = [{"id": "X", "title": "t", "status": "PASS", "detail": "d", "fix": ""}]
    text = to_json(results, {"nan": float("nan"), "inf": float("inf"),
                             "i": np.int64(3), "f": np.float64(1.5)})
    decoded = json.loads(text)
    assert decoded["meta"] == {"nan": None, "inf": None, "i": 3, "f": 1.5}


def test_demo_export_demo_dir_runs_and_naive_is_the_best_config(tmp_path):
    """`--demo-dir` 曾因 decile/hold 未定义直接 NameError；导出的原样序列
    也必须是 18 组里挑中的那组，而不是固定的 (0.10, 20)。"""
    import contextlib
    import importlib.util
    import io
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "demo" / "run_demo.py"
    spec = importlib.util.spec_from_file_location("demo_run_export", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with contextlib.redirect_stdout(io.StringIO()):
        assert mod.main(["--demo-dir", str(tmp_path)]) == 0
    naive = pd.read_csv(tmp_path / "naive_daily.csv")
    meta = json.loads((tmp_path / "naive_meta.json").read_text(encoding="utf-8"))
    assert (tmp_path / "corrected_daily.csv").exists()
    # reported_ann 是 ann_arith(naive 序列)，必须与导出序列对得上
    assert abs(meta["reported_ann"] - float(naive["strategy"].mean() * 252)) < 5e-4
