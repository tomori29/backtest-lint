# -*- coding: utf-8 -*-
"""绩效指标的口径回归：最大回撤必须把初始净值 1.0 算进峰值。"""
from __future__ import annotations

import math

import numpy as np

import pytest

from backtest_lint import metrics as M


def test_mdd_counts_first_day_loss():
    # 第一天亏 50%：从 1.0 起算，回撤就是 −50%，不能是 0
    assert M.mdd([-0.5, 0.1]) == pytest.approx(-0.5)


def test_mdd_normal_path():
    assert M.mdd([0.01, -0.02, 0.03]) == pytest.approx(-0.02)


def test_mdd_empty_is_nan():
    assert math.isnan(M.mdd([]))


def test_metrics_ignore_non_finite():
    assert M.total_return([0.1, float("inf"), -0.05]) == pytest.approx(1.1 * 0.95 - 1)


def test_annualization_respects_periods():
    # 月线：每年 12 期，年化必须按 12 折算，不能按 252
    r = [0.01] * 12
    assert M.ann_geometric(r, periods=12) == pytest.approx(1.01 ** 12 - 1, rel=1e-12)
    assert M.ann_arithmetic(r, periods=12) == pytest.approx(0.12, rel=1e-12)


def test_sharpe_respects_periods():
    r = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01]
    want = float(np.mean(r)) / float(np.std(r, ddof=1)) * np.sqrt(12)
    assert M.sharpe(r, periods=12) == pytest.approx(want, rel=1e-12)
