# -*- coding: utf-8 -*-
"""读取回测包：daily.csv + meta.json。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .checks import Bundle

DATE_ALIASES = ("date", "日期", "trade_date", "交易日")
STRATEGY_ALIASES = ("strategy", "策略", "ret", "return", "portfolio")
BENCH_ALIASES = ("benchmark", "基准", "bench", "bm")


def _pick(cols, aliases) -> str | None:
    lower = {str(c).strip().lower(): c for c in cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def load_trials(path: str | Path) -> "pd.DataFrame | None":
    """读候选配置收益矩阵：每列一个配置，每行一期。

    可选文件，没有就返回 None（E11/E12 转为「不适用」而不是失败）。
    """
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    drop = [c for c in df.columns if str(c).strip().lower() in DATE_ALIASES]
    if drop:
        df = df.drop(columns=drop)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.notna().sum() >= 8]
    if df.shape[1] < 2 or df.shape[0] < 8:
        raise ValueError(f"{p} 至少要有 2 列 × 8 行有效候选收益")
    return df.reset_index(drop=True)


def load_bundle(path: str | Path) -> Bundle:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        daily_path, meta_path = p, p.with_name("meta.json")
    else:
        daily_path = p / "daily.csv"
        meta_path = p / "meta.json"
    if not daily_path.exists():
        raise FileNotFoundError(f"找不到 {daily_path}")
    df = pd.read_csv(daily_path)
    dcol = _pick(df.columns, DATE_ALIASES)
    scol = _pick(df.columns, STRATEGY_ALIASES)
    bcol = _pick(df.columns, BENCH_ALIASES)
    if scol is None:
        raise ValueError(f"{daily_path} 缺少策略收益列（strategy / 策略）")
    out = pd.DataFrame()
    if dcol is not None:
        out["date"] = df[dcol].astype(str)
    else:
        out["date"] = [str(i) for i in range(len(df))]
    out["strategy"] = pd.to_numeric(df[scol], errors="coerce")
    if bcol is not None:
        out["benchmark"] = pd.to_numeric(df[bcol], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    meta = {}
    if meta_path.exists():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    base = daily_path.parent
    trials_name = meta.get("trials_csv") or "trials.csv"
    trials = load_trials(base / trials_name)
    return Bundle(daily=out, meta=meta, trials=trials)


TEMPLATE_META = {
    "name": "示例回测",
    "universe_mode": "pit",
    "delisted_included": False,
    "benchmark_type": "buy_hold",
    "cost": {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001,
             "applied_to_benchmark": False},
    "phases_run": 21,
    "panel_fixed": True,
    "panel_trail_history": [2700],
    "n_trials": 1,
    "signal_uses": None,
    "trade_price": None,
    "sample_days": None,
    "reported_t": None,
    "reported_ann": None,
    "reported_bench_ann": None,
}


def write_template(folder: str | Path) -> tuple[Path, Path]:
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    daily = d / "daily.csv"
    if not daily.exists():
        daily.write_text("date,strategy,benchmark\n"
                         "2024-01-02,0.0012,0.0008\n"
                         "2024-01-03,-0.0005,0.0001\n",
                         encoding="utf-8")
    meta = d / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps(TEMPLATE_META, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return daily, meta
