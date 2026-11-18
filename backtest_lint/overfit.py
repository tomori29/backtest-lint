# -*- coding: utf-8 -*-
"""搜索规模折价：把「试了几次」写成一个概率，而不是一句「t 够大就行」。

E1–E9、E13 检查的是**单次回测内部**的测量误差；这一层检查的是**多次尝试之间**的
选择偏差——同一段行情、同一份数据，你看了 N 个参数组合，然后把最好的那个
当成结论，这件事本身就会让夏普虚高，而单次回测的报表里看不出来。

五个函数：

  expected_max_sharpe(T, N)          纯噪声下 N 次尝试的最大夏普期望（= SR0）
  deflated_sharpe_ratio(...)         给定 SR / T / N / 偏度峰度，返回「真实 SR>0」的概率
  min_track_record_length(...)       要让这个 SR 在 N 次搜索后仍显著，最少需要多少期
  pbo_cscv(R)                        CSCV 法算回测过拟合概率（Bailey et al. 2017）
  cpcv_decay(R)                      CPCV 净化切分，看「样本内最优」在样本外还剩多少

出处：
  Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", JPM 40(5)
  Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest Overfitting", JCF
  López de Prado (2018), "Advances in Financial Machine Learning", ch. 7 & 12

口径（重要）：
  所有 SR 都是**逐期**口径（日收益就按日算），不是年化。
  年化 SR 要除以 sqrt(年交易日数) 再进来；T 是期数。
  正态近似下 SR0 = sqrt(1/(T-1)) · E[max z]，非正态时真实 SR0 更大，
  所以这里的 SR0 是**偏乐观**的（低估门槛）——用的时候要知道方向。
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

EULER_GAMMA = 0.5772156649015329
# MinTRL 的数值反解上限：超过这个期数仍达不到门槛就返回 inf（由调用方区分
# 「SR 不为正」和「样本还不够长」）
MTRL_CAP = 20000


# ---------------------------------------------------------------- 正态分位
try:                                                  # scipy 不是必需依赖
    from scipy.stats import norm

    def _z(p: float) -> float:
        return float(norm.ppf(p))

    def _phi(x: float) -> float:
        return float(norm.cdf(x))
except Exception:                                     # noqa: BLE001
    def _z(p: float) -> float:
        """Acklam 有理逼近（|误差| < 1e-9）。"""
        if not 0.0 < p < 1.0:
            raise ValueError("p 必须在 (0,1)")
        t = math.sqrt(-2.0 * math.log(min(p, 1 - p)))
        c = (2.515517, 0.802853, 0.010328)
        d = (1.432788, 0.189269, 0.001308)
        num = c[0] + c[1] * t + c[2] * t * t
        den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
        z = t - num / den
        return -z if p < 0.5 else z

    def _phi(x: float) -> float:
        """Abramowitz-Stegun 26.2.17（|误差| < 7.5e-8）。"""
        sign = 1.0 if x >= 0 else -1.0
        x = abs(x) / math.sqrt(2.0)
        tt = 1.0 / (1.0 + 0.3275911 * x)
        y = 1.0 - (((((1.061405429 * tt - 1.453152027) * tt)
                     + 1.421413741) * tt - 0.284496736) * tt
                   + 0.254829592) * tt * math.exp(-x * x)
        return 0.5 * (1.0 + sign * y)


# ---------------------------------------------------------------- 收益序列的矩
def sr_stats(r) -> dict:
    """逐期 SR、期数、偏度、未超额峰度。全部按总体矩算（与 scipy 默认一致）。"""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    T = int(len(r))
    if T < 3:
        raise ValueError("有效期数少于 3")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))          # Sharpe 用样本标准差（与 metrics.sharpe 一致）
    sd_pop = float(r.std())            # 偏度/峰度按总体矩标准化（DSR 的通行定义）
    if sd <= 0 or sd_pop <= 0:
        raise ValueError("收益序列方差为 0")
    z = (r - mu) / sd_pop
    return {"sr": mu / sd, "T": T, "mean": mu, "sd": sd,
            "skew": float(np.mean(z ** 3)),
            "kurt": float(np.mean(z ** 4))}


# ---------------------------------------------------------------- SR0
def expected_max_sharpe(T: int, N: int) -> float:
    """纯噪声下 N 次尝试的最大夏普期望（逐期口径）。

    E[max SR] = sqrt(Var(SR)) · [(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne))]
    Var(SR)   取 1/(T-1)。
    """
    if T < 2:
        raise ValueError("T 至少为 2")
    if N < 1:
        raise ValueError("N 至少为 1")
    if N == 1:
        return 0.0
    g = EULER_GAMMA
    sd = 1.0 / math.sqrt(T - 1)
    return sd * ((1 - g) * _z(1 - 1.0 / N)
                 + g * _z(1 - 1.0 / (N * math.e)))


# ---------------------------------------------------------------- DSR
def deflated_sharpe_ratio(sr: float, T: int, N: int,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """真实 SR > 0 的概率，已对「试了 N 次」和非正态做修正。

    DSR = Φ[ (SR - SR0)·sqrt(T-1) / sqrt(1 - γ3·SR + ((γ4-1)/4)·SR²) ]

    kurt 用**未超额**峰度（正态 = 3）。
    """
    sr0 = expected_max_sharpe(T, N)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        raise ValueError("分母非正：偏度/峰度与 SR 的组合不合法")
    return _phi((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom))


def min_track_record_length(sr: float, N: int, alpha: float = 0.05,
                            skew: float = 0.0, kurt: float = 3.0,
                            cap: int = MTRL_CAP) -> float:
    """要让 SR 在 N 次搜索后仍显著，至少需要多少期。

    SR0 本身随 T 变化，所以直接数值反解，不做解析近似。
    """
    if sr <= 0:
        return float("inf")
    for T in range(2, cap):
        if deflated_sharpe_ratio(sr, T, N, skew, kurt) >= 1 - alpha:
            return float(T)
    return float("inf")


# ---------------------------------------------------------------- 过拟合概率
def pbo_cscv(R, n_blocks: int = 16) -> dict:
    """CSCV 法算回测过拟合概率（PBO）。

    R : (T 期 × N 策略) 的收益矩阵，每列一个候选配置。
    n_blocks : 必须为 >=4 的偶数。

    把 T 切成 n_blocks 个连续块，枚举「取一半块当样本内」的全部组合，
    每次在样本内挑夏普最高的那一列，看它在样本外的排名。
    样本内最优在样本外掉到中位数以下的频率，就是 PBO。
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("R 必须是二维矩阵")
    T, N = R.shape
    if N < 2:
        raise ValueError("至少需要 2 个候选配置")
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks 必须是 >=4 的偶数")
    if T < n_blocks:
        raise ValueError("期数少于块数")

    # 截断到能整除的长度，避免 linspace 产生 18/19 行混排的不等长块
    T_use = (T // n_blocks) * n_blocks
    R = R[:T_use]
    size = T_use // n_blocks
    blocks = [R[i * size:(i + 1) * size] for i in range(n_blocks)]

    half = n_blocks // 2
    logits = []
    for is_idx in combinations(range(n_blocks), half):
        oos_idx = [i for i in range(n_blocks) if i not in is_idx]
        IS = np.vstack([blocks[i] for i in is_idx])
        OOS = np.vstack([blocks[i] for i in oos_idx])
        best = int(np.argmax(_sharpe_cols(IS)))
        sr_oos = _sharpe_cols(OOS)
        rank = int((sr_oos > sr_oos[best]).sum()) + 1
        logits.append(math.log((N + 1 - rank) / rank)
                      if 0 < rank <= N else -math.inf)

    logits = np.asarray(logits, dtype=float)
    return {
        "pbo": float((logits <= 0).mean()),
        "n_splits": int(len(logits)),
        "mean_logit": float(logits.mean()),
    }


# ---------------------------------------------------------------- CPCV
def n_paths(n_groups: int, k_test: int) -> int:
    """CPCV 能拼出多少条完整样本外路径：k·C(N,k)/N。"""
    if not 1 <= k_test < n_groups:
        raise ValueError("要求 1 <= k_test < n_groups")
    return k_test * math.comb(n_groups, k_test) // n_groups


def cpcv_splits(n_samples: int, n_groups: int = 6, k_test: int = 2,
                label_horizon: int = 1, embargo: int = 0) -> list[dict]:
    """枚举全部 C(N,k) 个切分，每个切分已做好净化（purge）与隔离（embargo）。"""
    if n_groups < 2:
        raise ValueError("n_groups 至少为 2")
    if not 1 <= k_test < n_groups:
        raise ValueError("要求 1 <= k_test < n_groups")
    if n_samples < n_groups:
        raise ValueError("样本数少于组数")

    # 同样截断到整除，保证每组样本数相等
    n_use = (n_samples // n_groups) * n_groups
    size = n_use // n_groups
    groups = [np.arange(i * size, (i + 1) * size) for i in range(n_groups)]
    n_samples = n_use
    all_idx = np.arange(n_samples)

    out = []
    for combo in combinations(range(n_groups), k_test):
        test = np.sort(np.concatenate([groups[g] for g in combo]))
        train = np.setdiff1d(all_idx, test, assume_unique=False)

        if label_horizon > 0 and len(train) and len(test):
            keep = np.ones(len(train), dtype=bool)
            for t in test:
                keep &= np.abs(train - t) > label_horizon
            train = train[keep]

        if embargo > 0:
            drop = set()
            for g in combo:
                end = groups[g][-1]
                drop.update(range(end + 1, min(end + 1 + embargo, n_samples)))
            if drop:
                train = train[~np.isin(train, np.fromiter(drop, dtype=int))]

        out.append({"test": test, "train": train, "groups": combo})
    return out


def cpcv_decay(R, n_groups: int = 6, k_test: int = 2,
               label_horizon: int = 1, embargo: int = 0) -> dict:
    """样本内最优的配置，在净化后的样本外还剩下多少。

    对每个切分：在训练段挑夏普最高的那一列，记它在测试段的夏普。
    返回 OOS SR <= 0 的比例（decay）、中位 OOS SR、路径数 n_paths。

    与 PBO 的区别：PBO 问「样本内最优在样本外排第几」，decay 问
    「样本内最优在样本外还赚不赚钱」——前者是相对排名，后者是绝对符号。
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("R 必须是二维矩阵")
    T, N = R.shape
    if N < 2:
        raise ValueError("至少需要 2 个候选配置")
    if T < n_groups:
        raise ValueError("期数少于组数")

    splits = cpcv_splits(T, n_groups, k_test, label_horizon, embargo)
    oos_sr, picks = [], []
    for s in splits:
        if len(s["train"]) < 3 or len(s["test"]) < 2:
            continue
        best = int(np.argmax(_sharpe_cols(R[s["train"]])))
        picks.append(best)
        oos_sr.append(float(_sharpe_cols(R[s["test"]])[best]))

    if not oos_sr:
        raise ValueError("没有可用的切分")
    oos_sr = np.asarray(oos_sr, dtype=float)
    return {
        "decay": float((oos_sr <= 0).mean()),
        "median_oos_sr": float(np.median(oos_sr)),
        "n_splits": int(len(oos_sr)),
        "n_paths": n_paths(n_groups, k_test),
        "n_distinct_picks": int(len(set(picks))),
    }


def _sharpe_cols(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, -np.inf)
    return sr
