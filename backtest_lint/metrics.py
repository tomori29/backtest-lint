# -*- coding: utf-8 -*-
"""基础绩效计算。只依赖 numpy。"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252.0


def clean(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return r[np.isfinite(r)]


def ann_geometric(r) -> float:
    r = clean(r)
    if len(r) == 0:
        return float("nan")
    total = float(np.prod(1.0 + r))
    return total ** (TRADING_DAYS / len(r)) - 1.0 if total > 0 else float("nan")


def ann_arithmetic(r) -> float:
    r = clean(r)
    return float(np.mean(r) * TRADING_DAYS) if len(r) else float("nan")


def ann_vol(r) -> float:
    r = clean(r)
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 2 else float("nan")


def sharpe(r) -> float:
    r = clean(r)
    if len(r) < 2:                       # ddof=1 在 0/1 个点上无定义
        return float("nan")
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def mdd(r) -> float:
    r = clean(r)
    if len(r) == 0:
        return float("nan")
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def total_return(r) -> float:
    r = clean(r)
    return float(np.prod(1.0 + r) - 1.0) if len(r) else float("nan")


def bonferroni_z(n_trials: int, alpha: float = 0.05) -> float:
    """双侧 Bonferroni 临界 z。"""
    from statistics import NormalDist

    if n_trials <= 1:
        return NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * n_trials))
