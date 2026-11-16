# -*- coding: utf-8 -*-
"""生成 bad / good 两个示例回测包。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def make(name: str, n: int, mu_s: float, mu_b: float, seed: int,
         vol_s: float = 0.012, vol_b: float = 0.010):
    rng = np.random.default_rng(seed)
    rs = rng.normal(mu_s, vol_s, n)
    rb = rng.normal(mu_b, vol_b, n)
    dates = pd.bdate_range("2021-01-04", periods=n).strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "strategy": rs, "benchmark": rb})
    (ROOT / name).mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / name / "daily.csv", index=False)
    geo_s = (np.prod(1 + rs)) ** (252 / n) - 1
    geo_b = (np.prod(1 + rb)) ** (252 / n) - 1
    ari_s = rs.mean() * 252
    ari_b = rb.mean() * 252
    t = rs.mean() / (rs.std(ddof=1) / np.sqrt(n))
    return geo_s, geo_b, ari_s, ari_b, t


def main():
    # bad：静态池、单相位、面板漂移、基准每日再平衡零成本、18 组网格只报最好、报告用算术年化
    geo_s, geo_b, ari_s, ari_b, t = make("bad", 320, 0.00080, 0.00040, 7)
    meta = {
        "name": "示例·踩坑回测",
        "universe_mode": "static",
        "delisted_included": False,
        "benchmark_type": "ew_daily",
        "cost": {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001,
                 "applied_to_benchmark": False},
        "phases_run": 1,
        "panel_fixed": False,
        "panel_trail_history": [1040, 2700],
        "n_trials": 18,
        # 故意写成"网格里挑出来的最好一组"：t 不显著、年化比逐日复利高 2pp
        "reported_t": 1.2,
        "reported_ann": round(float(ari_s) + 0.02, 4),
        "reported_bench_ann": round(float(ari_b), 4),
        "signal_uses": "close_t",
        "trade_price": "close_t",          # 收盘后确认的信号，同日收盘成交
    }
    (ROOT / "bad" / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # good：时点池、含退市、买入持有基准、21 相位、面板固定、单次检验
    geo_s, geo_b, ari_s, ari_b, t = make("good", 500, 0.00085, 0.00030, 11, 0.008, 0.008)
    meta = {
        "name": "示例·干净回测",
        "universe_mode": "pit",
        "delisted_included": True,
        "benchmark_type": "buy_hold",
        "cost": {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001,
                 "applied_to_benchmark": False},
        "phases_run": 21,
        "panel_fixed": True,
        "panel_trail_history": [2700],
        "n_trials": 1,
        "reported_t": round(float(t), 3),
        "reported_ann": round(float(geo_s), 4),
        "reported_bench_ann": round(float(geo_b), 4),
        "signal_uses": "close_t",
        "trade_price": "open_t+1",
    }
    (ROOT / "good" / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("generated", ROOT / "bad", ROOT / "good")


def make_multi_trial(name: str = "multi_trial"):
    """E11/E12 的示例：24 组候选，其中「最好的一组」只有前半段有效。

    这是最典型的过拟合形态——参数在前 40% 的样本里很强，之后失效。
    E1–E9 全都看不出来（单次回测内部一切正常），只有把全部候选放在一起
    才能看出「样本内最优」是挑出来的。
    """
    rng = np.random.default_rng(42)
    n, n_cols = 500, 24
    R = rng.normal(0, 0.008, size=(n, n_cols))
    R[:200, 3] += 0.0035                      # 前 40% 有效，之后失效
    dates = pd.bdate_range("2021-01-04", periods=n).strftime("%Y-%m-%d")

    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(R, columns=[f"cfg{i:02d}" for i in range(n_cols)]).to_csv(
        d / "trials.csv", index=False)

    rs, rb = R[:, 3], rng.normal(0.0003, 0.008, n)
    pd.DataFrame({"date": dates, "strategy": rs, "benchmark": rb}).to_csv(
        d / "daily.csv", index=False)

    t = rs.mean() / (rs.std(ddof=1) / np.sqrt(n))
    meta = {
        "name": "示例·24 组网格里挑出来的最好一组",
        "universe_mode": "pit",
        "delisted_included": True,
        "benchmark_type": "buy_hold",
        "cost": {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001,
                 "applied_to_benchmark": False},
        "phases_run": 21,
        "panel_fixed": True,
        "panel_trail_history": [2700],
        "n_trials": n_cols,
        "reported_t": round(float(t), 3),
        "reported_ann": round(float((np.prod(1 + rs)) ** (252 / n) - 1), 4),
        "reported_bench_ann": round(float((np.prod(1 + rb)) ** (252 / n) - 1), 4),
        "signal_uses": "close_t",
        "trade_price": "open_t+1",
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print("generated", d)


if __name__ == "__main__":
    main()
    make_multi_trial()
