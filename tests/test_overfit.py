# -*- coding: utf-8 -*-
"""每条测试都对着手算答案或已知性质，不测「能跑」。

这些数字来自 E10–E12 的原型：T=154 的月度样本上，最好的策略 SR 0.34
在 N=1 时 DSR 只有 0.40——「只跑一次也验不出来」是本仓库最反直觉的一条结论，
必须由测试钉住。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backtest_lint.overfit import (cpcv_decay, cpcv_splits,
                                   deflated_sharpe_ratio, expected_max_sharpe,
                                   min_track_record_length, n_paths, pbo_cscv,
                                   sr_stats)


# ---------------------------------------------------------------- 矩
def test_sr_stats_matches_hand_calculation():
    r = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
    st = sr_stats(r)
    assert st["T"] == 5
    # sr_stats 用样本标准差（ddof=1），与 metrics.sharpe 一致
    assert abs(st["sr"] - r.mean() / r.std(ddof=1)) < 1e-12
    assert abs(st["kurt"] - 3.0) < 1.5          # 5 个点的峰度不会正好是 3


def test_sr_stats_rejects_degenerate_input():
    with pytest.raises(ValueError):
        sr_stats([0.01, 0.02])
    with pytest.raises(ValueError):
        sr_stats(np.zeros(50))


# ---------------------------------------------------------------- SR0
def test_sr0_zero_when_single_trial():
    """只试一次就没有选择偏差，门槛必须是 0。"""
    assert expected_max_sharpe(120, 1) == 0.0


def test_sr0_against_hand_calculation():
    """手算：T=1200、N=100。

    sd = 1/sqrt(1199) = 0.02888
    E[max z] = (1-γ)Φ⁻¹(0.99) + γΦ⁻¹(1-1/(100e))
             = 0.4228×2.3263 + 0.5772×2.6825 = 2.5319
    SR0 = 0.02888 × 2.5319 = 0.0731
    """
    assert abs(expected_max_sharpe(1200, 100) - 0.0731) < 0.001


def test_sr0_grows_with_trials_and_shrinks_with_sample():
    assert expected_max_sharpe(120, 100) > expected_max_sharpe(120, 10)
    assert expected_max_sharpe(1200, 100) < expected_max_sharpe(120, 100)


def test_sr0_scales_as_inverse_sqrt_T():
    """SR0 ∝ 1/sqrt(T-1)。T 放大 4 倍，门槛应减半。"""
    a = expected_max_sharpe(101, 50)
    b = expected_max_sharpe(401, 50)
    assert abs(b / a - 0.5) < 0.02


# ---------------------------------------------------------------- DSR
def test_dsr_equals_one_sided_pvalue_when_N1():
    """N=1 时 DSR 退化为普通单尾检验：Φ(SR·sqrt(T-1))。"""
    sr, T = 0.10, 121
    assert abs(deflated_sharpe_ratio(sr, T, 1) - 0.8643) < 0.005


def test_dsr_decreases_with_trials():
    vals = [deflated_sharpe_ratio(0.15, 120, n) for n in (1, 10, 100, 1000)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_dsr_penalises_fat_tails():
    """同样的 SR，峰度越高越不可信。"""
    normal = deflated_sharpe_ratio(0.15, 120, 10, skew=0.0, kurt=3.0)
    fat = deflated_sharpe_ratio(0.15, 120, 10, skew=0.0, kurt=10.0)
    assert fat < normal


def test_dsr_negative_sr_gives_low_probability():
    assert deflated_sharpe_ratio(-0.10, 120, 10) < 0.05


def test_dsr_reproduces_the_154_period_case():
    """原始案例：T=154 个月、年化 SR 0.34（最好的一条策略）。

    N=1（只跑过一次）时 DSR≈0.887；N=8（挑过 8 组）时掉到 ≈0.41。
    这条是 E10 存在的理由：N=1 时勉强过关，一旦承认挑过 8 次就不过了，
    而报表上这两个数字长得一模一样。
    """
    T, ann_sr = 154, 0.34
    sr = ann_sr / math.sqrt(12)                # 月度口径
    assert abs(deflated_sharpe_ratio(sr, T, 1) - 0.887) < 0.02
    assert abs(deflated_sharpe_ratio(sr, T, 8) - 0.41) < 0.05


def test_required_sr_at_154_periods_is_about_046():
    """同样的样本长度，N=1 也要 SR≈0.46（年化）才能过 0.95——现有最好 0.34。

    这条钉住「样本长度本身就不够」，换个因子救不回来。
    """
    T = 154
    z = 1.6448536269514722                     # Φ⁻¹(0.95)
    sr_needed_ann = z / math.sqrt(T - 1) * math.sqrt(12)
    assert abs(sr_needed_ann - 0.461) < 0.005
    # 反证：刚好卡在门槛上的 SR，DSR 正好是 0.95
    assert abs(deflated_sharpe_ratio(sr_needed_ann / math.sqrt(12), T, 1) - 0.95) < 0.01


# ---------------------------------------------------------------- MTRL
def test_mtrl_is_where_dsr_crosses_alpha():
    sr, N = 0.10, 20
    T = min_track_record_length(sr, N, alpha=0.05)
    assert deflated_sharpe_ratio(sr, int(T), N) >= 0.95
    assert deflated_sharpe_ratio(sr, int(T) - 1, N) < 0.95


def test_mtrl_infinite_for_nonpositive_sr():
    assert math.isinf(min_track_record_length(0.0, 10))
    assert math.isinf(min_track_record_length(-0.05, 10))


# ---------------------------------------------------------------- PBO
def test_pbo_about_half_for_pure_noise():
    """全是噪声时，IS 最优在 OOS 没有优势，PBO 应接近 0.5。"""
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.04, size=(240, 20))
    assert 0.35 < pbo_cscv(R, n_blocks=8)["pbo"] < 0.65


def test_pbo_low_when_one_strategy_really_works():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.04, size=(240, 20))
    R[:, 3] += 0.02
    assert pbo_cscv(R, n_blocks=8)["pbo"] < 0.15


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 5)), n_blocks=7)


# ---------------------------------------------------------------- CPCV
def test_n_paths_matches_formula():
    assert n_paths(6, 2) == 5          # C(6,2)=15, 2*15/6=5
    assert n_paths(10, 2) == 9         # C(10,2)=45, 2*45/10=9
    with pytest.raises(ValueError):
        n_paths(4, 4)


def test_split_count_is_binomial_and_disjoint():
    s = cpcv_splits(120, n_groups=6, k_test=2)
    assert len(s) == 15
    for x in s:
        assert len(x["test"]) == 40            # 两组，每组 20
        assert len(set(x["test"]) & set(x["train"])) == 0


def test_purge_removes_overlapping_labels():
    h = 3
    s = cpcv_splits(120, n_groups=6, k_test=2, label_horizon=h)
    for x in s:
        for t in x["test"]:
            assert np.all(np.abs(x["train"] - t) > h)


def test_purge_actually_drops_samples():
    raw = cpcv_splits(120, n_groups=6, k_test=2, label_horizon=0)
    purged = cpcv_splits(120, n_groups=6, k_test=2, label_horizon=3)
    assert sum(len(x["train"]) for x in purged) < sum(len(x["train"]) for x in raw)


def test_embargo_removes_samples_after_test_block():
    s = cpcv_splits(120, n_groups=6, k_test=2, label_horizon=0, embargo=5)
    edges = np.linspace(0, 120, 7).astype(int)
    for x in s:
        for g in x["groups"]:
            end = edges[g + 1] - 1
            banned = set(range(end + 1, min(end + 6, 120)))
            assert not (set(x["train"]) & banned)


def test_cpcv_decay_high_for_pure_noise():
    rng = np.random.default_rng(0)
    res = cpcv_decay(rng.normal(0, 0.01, size=(240, 12)))
    assert res["n_splits"] == 15
    assert res["n_paths"] == 5
    assert res["decay"] > 0.3


def test_cpcv_decay_zero_for_a_stable_edge():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.01, size=(240, 12))
    R[:, 5] += 0.004
    res = cpcv_decay(R)
    assert res["decay"] == 0.0
    assert res["n_distinct_picks"] == 1
