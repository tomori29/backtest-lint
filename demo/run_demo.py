# -*- coding: utf-8 -*-
"""可复现的「推翻」演示：一个正超额如何被逐条修正打回零。

**合成数据**，不是真实市场：真实 alpha 设为 0，任何正超额都只能来自测量方式。
固定随机种子，逐条打印每一步修正的影响。

跑法：
    python demo/run_demo.py
    python demo/run_demo.py --demo-dir demo/_out    # 导出 csv/meta，供 backtest_lint 体检
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 20261116
N_STOCKS = 300
N_DAYS = 1500          # 约 6 年
DELIST_RATE = 0.04     # 每年退市比例；退市当天 −50%，之后消失
COST_ROUND = 0.00025 + 0.0005 + 0.001   # 佣金＋印花税＋滑点，单边合计≈0.175%/次

# 参数网格：18 组 = 原项目里那种"试了很多组、挑最好的一组来报"
# 3 个 decile × 6 个持有期 = 18；此前只有 3 个持有期（9 组），与文字说明不符
GRID = [(d, h) for d in (0.05, 0.10, 0.20) for h in (5, 10, 20, 30, 40, 60)]


# ------------------------------------------------------------------ 造数据 --
def make_market(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, size=(N_DAYS, N_STOCKS))

    alive = np.ones(N_STOCKS, dtype=bool)
    for start in range(0, N_DAYS, 250):
        end = min(start + 250, N_DAYS)
        n_dead = rng.binomial(int(alive.sum()), DELIST_RATE)
        if n_dead:
            for d in rng.choice(np.flatnonzero(alive), size=n_dead, replace=False):
                t = int(rng.integers(start + 1, end))
                rets[t, d] = -0.5
                rets[t + 1:, d] = np.nan
                alive[d] = False

    dates = pd.bdate_range("2020-12-01", periods=N_DAYS)
    return pd.DataFrame(rets, index=dates, columns=[f"S{i:03d}" for i in range(N_STOCKS)])


# -------------------------------------------------------------- 策略与基准 --
def reversal(rets: pd.DataFrame, decile: float, hold: int,
             survivors_only: bool) -> pd.Series:
    """横截面反转（做多过去 20 日最弱的 decile），每 hold 天换仓，信号次日成交。"""
    cols = rets.columns[rets.notna().all()] if survivors_only else rets.columns
    px = (1.0 + rets[cols].fillna(0.0)).cumprod()
    sig = px.pct_change(20)

    port = pd.Series(0.0, index=rets.index)
    for start in range(21, len(rets) - hold - 1, hold):
        s = sig.iloc[start].dropna()
        if len(s) < 30:
            continue
        k = max(2, int(len(s) * decile))
        picked = s.nsmallest(k).index
        w = rets[picked].iloc[start + 1:start + 1 + hold].mean(axis=1, skipna=True)
        port.loc[w.index] = w.values
    return port


def bench_ew_daily(rets: pd.DataFrame) -> pd.Series:
    """每日再平衡等权（隐含每天全额换手，却按零成本计）。"""
    return rets.mean(axis=1, skipna=True)


def bench_buy_hold(rets: pd.DataFrame) -> pd.Series:
    """期初等权买入、之后不动（零换手）。"""
    w0 = rets.notna().iloc[0].astype(float)
    w0 = w0 / w0.sum()
    val = (1.0 + rets.fillna(0.0)).cumprod().mul(w0, axis=1).sum(axis=1)
    return val.pct_change().fillna(0.0)


def ann_arith(r: pd.Series) -> float:
    return float(np.nanmean(r) * 252)


def ann_geom(r: pd.Series) -> float:
    return float(np.expm1(np.log1p(r.fillna(0.0)).mean() * 252))


# ------------------------------------------------------------------ 主流程 --
def compute_steps(rets: pd.DataFrame):
    """返回 (步骤表, 挑中的参数, 中间结果)。测试与 CLI 共用这一段。"""
    bench_ew = bench_ew_daily(rets)
    bench_bh = bench_buy_hold(rets)
    idx = rets.index[21:-1]

    # ① 原样：幸存者池、18 组里挑最好、策略不扣成本、基准零成本日再平衡
    grid = {}
    for d, h in GRID:
        s = reversal(rets, d, h, survivors_only=True)
        grid[(d, h)] = ann_arith(s.loc[idx]) - ann_arith(bench_ew.loc[idx])
    best_key = max(grid, key=grid.get)

    decile, hold = 0.10, 20              # ② 之后固定用一组事先定好的参数
    strat_surv = reversal(rets, decile, hold, survivors_only=True)
    strat_pit = reversal(rets, decile, hold, survivors_only=False)
    strat_cost = strat_pit - 2 * COST_ROUND / hold

    steps = [
        ("① 原样：幸存者池 / 18 组挑最好 / 策略不扣成本 / 基准零成本日再平衡",
         grid[best_key]),
        ("② 修多重检验：18 组取中位数，不挑最好的一组",
         float(np.median(list(grid.values())))),
        ("③ 修幸存者偏差：策略池含已退市股票",
         ann_arith(strat_pit.loc[idx]) - ann_arith(bench_ew.loc[idx])),
        ("④ 修成本：策略每期换仓扣双边成本",
         ann_arith(strat_cost.loc[idx]) - ann_arith(bench_ew.loc[idx])),
        ("⑤ 修基准：日再平衡 → 买入持有（基准不再零成本全额换手）",
         ann_arith(strat_cost.loc[idx]) - ann_arith(bench_bh.loc[idx])),
        ("⑥ 修年化口径：两边统一几何年化",
         ann_geom(strat_cost.loc[idx]) - ann_geom(bench_bh.loc[idx])),
    ]
    ctx = dict(bench_ew=bench_ew, bench_bh=bench_bh, idx=idx,
               strat_surv=strat_surv, strat_pit=strat_pit, strat_cost=strat_cost)
    return steps, best_key, ctx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", help="导出 naive / corrected 两套 csv+meta")
    args = ap.parse_args(argv)

    rets = make_market()
    n_alive = int(rets.notna().all().sum())
    print(f"合成市场：{N_DAYS} 个交易日 / {N_STOCKS} 只股票，"
          f"期末存活 {n_alive} 只（退市 {N_STOCKS - n_alive} 只），真实 alpha = 0\n")

    steps, best_key, ctx = compute_steps(rets)
    bench_ew, bench_bh, idx = ctx["bench_ew"], ctx["bench_bh"], ctx["idx"]
    # ① 原样导出必须真的是「18 组里挑中的那组」，不能拿固定的 (0.10, 20) 冒充
    strat_best = reversal(rets, best_key[0], best_key[1], survivors_only=True)

    w = 46
    print(f"{'步骤':<{w}}{'年化超额':>10}")
    print("-" * (w + 14))
    for name, v in steps:
        print(f"{name:<{w}}{v*100:>9.2f}%")
    print("-" * (w + 14))
    print(f"{'合计变化':<{w}}{(steps[-1][1]-steps[0][1])*100:>9.2f}pp")

    print(f"\n① 挑中的那组：decile={best_key[0]}, hold={best_key[1]}"
          f"（18 组里年化超额最高的）")
    print("说明：这是**合成数据**，真实 alpha 设为 0。剩下那点数字是噪声，换种子会变。"
          "演示的是「测量口径」能把同一份数据推到多远，不是任何真实策略的业绩。")

    if args.demo_dir:
        out = Path(args.demo_dir)
        out.mkdir(parents=True, exist_ok=True)
        strat_pit = ctx["strat_pit"]
        strat_cost = ctx["strat_cost"]

        pd.DataFrame({
            "date": idx.strftime("%Y-%m-%d"),
            "strategy": strat_best.loc[idx].values,
            "benchmark": bench_ew.loc[idx].values,
        }).to_csv(out / "naive_daily.csv", index=False)
        pd.DataFrame({
            "date": idx.strftime("%Y-%m-%d"),
            "strategy": strat_cost.loc[idx].values,
            "benchmark": bench_bh.loc[idx].values,
        }).to_csv(out / "corrected_daily.csv", index=False)

        metas = {
            "naive": {
                "name": f"演示·原样（幸存者池，decile={best_key[0]}, hold={best_key[1]}，18 组挑最好）",
                "universe_mode": "static", "delisted_included": False,
                "benchmark_type": "ew_daily",
                "cost": {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001,
                         "applied_to_benchmark": False},
                "phases_run": 1, "panel_fixed": False,
                "panel_trail_history": [20, 60], "n_trials": len(GRID),
                "signal_uses": "close_t", "trade_price": "open_t+1",
                "sample_days": len(idx), "reported_t": 4.2,
                "reported_ann": round(ann_arith(strat_best.loc[idx]), 4),
                "reported_bench_ann": round(ann_arith(bench_ew.loc[idx]), 4),
            },
            "corrected": {
                "name": "演示·修正后（含退市、单组参数、双边成本、买入持有基准）",
                "universe_mode": "pit", "delisted_included": True,
                "benchmark_type": "buy_hold",
                "cost": {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001,
                         "applied_to_benchmark": False},
                "phases_run": 5, "panel_fixed": True,
                "panel_trail_history": [20], "n_trials": 1,
                "signal_uses": "close_t", "trade_price": "open_t+1",
                "sample_days": len(idx), "reported_t": 0.4,
                "reported_ann": round(ann_geom(strat_cost.loc[idx]), 4),
                "reported_bench_ann": round(ann_geom(bench_bh.loc[idx]), 4),
            },
        }
        for tag, meta in metas.items():
            (out / f"{tag}_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已导出：{out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
