# -*- coding: utf-8 -*-
"""测量误差检查（E1–E9、E13）+ 搜索规模三检（E10–E12）+ 四项基础健全性检查。

每条检查返回：
  id / title / status(PASS|WARN|FAIL|INFO) / detail / fix
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import metrics as mt
from . import overfit as of


@dataclass
class Bundle:
    daily: object                     # pandas.DataFrame: date, strategy, benchmark
    meta: dict = field(default_factory=dict)
    trials: object = None             # pandas.DataFrame: 每列一个候选配置（可选）

    @property
    def strategy(self) -> np.ndarray:
        return np.asarray(self.daily["strategy"], dtype=float)

    @property
    def benchmark(self) -> np.ndarray | None:
        if "benchmark" not in self.daily.columns:
            return None
        return np.asarray(self.daily["benchmark"], dtype=float)


def _r(check_id, title, status, detail, fix=""):
    return {"id": check_id, "title": title, "status": status,
            "detail": detail, "fix": fix}


def _pct(x: float | None, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _num(x: float | None, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# ---------------- 1. 基准口径 ----------------

def check_benchmark_present(b: Bundle):
    if b.benchmark is None:
        return _r("B0", "基准存在", "FAIL",
                  "日收益文件里没有 benchmark 列，无法计算超额。",
                  "补一列基准日收益；没有基准就无法判断策略是否真的跑赢'什么都不做'。")
    ok = int(np.isfinite(b.benchmark).sum())
    if ok == 0:
        return _r("B0", "基准存在", "FAIL",
                  "benchmark 列全为缺失值，等价于没有基准。",
                  "补一列有效的基准日收益；全 NaN 的列不能当基准。")
    return _r("B0", "基准存在", "PASS",
              f"基准列有效 {ok}/{len(b.benchmark)} 天。")


def check_benchmark_accounting(b: Bundle):
    """基准年化：几何 vs 算术，差多少；报告值更像哪一个。"""
    if b.benchmark is None:
        return _r("E1", "基准记账口径", "FAIL", "缺基准，无法检查。", "补基准。")
    r = b.benchmark
    geo = mt.ann_geometric(r)
    ari = mt.ann_arithmetic(r)
    if not (np.isfinite(geo) and np.isfinite(ari)):
        return _r("E1", "基准记账口径", "WARN",
                  "基准没有有效收益（或收益 ≤ -100%），几何/算术年化无法计算，"
                  "无法核对记账口径。",
                  "先补有效的基准日收益，再核对报告值用的是几何还是算术年化。")
    gap = ari - geo
    reported = b.meta.get("reported_bench_ann")
    try:
        reported = float(reported) if reported is not None else None
    except (TypeError, ValueError):
        reported = None
    detail = (f"基准几何年化 {_pct(geo)}，算术年化 {_pct(ari)}，"
              f"差 {gap * 100:+.2f}pp。")
    if reported is not None and np.isfinite(reported):
        d_geo = abs(reported - geo)
        d_ari = abs(reported - ari)
        detail += f" 报告值 {_pct(reported)}。"
        if d_ari + 0.001 < d_geo and abs(gap) > 0.005:
            return _r("E1", "基准记账口径", "FAIL",
                      detail + " 报告值更接近算术年化，与逐日复利（几何）不一致；"
                      "两种口径混用会让超额失真。",
                      "基准与策略一律用几何（复利）口径；等权基准的日收益必须是算术平均 "
                      "mean(1+r)-1，不能用 exp(mean(log r))。")
    if abs(gap) > 0.01:
        return _r("E1", "基准记账口径", "WARN",
                  detail + " 两种口径差超过 1pp，报告必须声明用的是哪一种。",
                  "在结果里同时给出几何与算术年化，并注明主口径。")
    return _r("E1", "基准记账口径", "PASS", detail)


# ---------------- 2. 基准成本对称 ----------------

def check_cost_symmetry(b: Bundle):
    cost = b.meta.get("cost") or {}
    if not isinstance(cost, dict):
        return _r("E2", "成本对称", "FAIL",
                  f"meta.cost 必须是对象（含 commission/stamp/slip），实际是 "
                  f"{type(cost).__name__}。",
                  'meta.cost 写成 {"commission": ..., "stamp": ..., "slip": ..., '
                  '"applied_to_benchmark": true/false}。')
    applied = cost.get("applied_to_benchmark")
    btype = str(b.meta.get("benchmark_type", "unknown")).lower()
    declared = all(k in cost for k in ("commission", "stamp", "slip"))
    if not declared:
        return _r("E2", "成本对称", "FAIL",
                  "meta.cost 缺 commission/stamp/slip，成本口径未声明。",
                  "写明佣金、印花税、滑点，以及基准是否同样扣成本。")
    if applied is True:
        return _r("E2", "成本对称", "PASS", "策略与基准均扣成本。")
    if applied in (False, 0, "false", "no", "0", "False"):
        if btype in ("ew_daily", "strategy_like", "daily_rebalance"):
            return _r("E2", "成本对称", "FAIL",
                      f"基准类型 {btype} 隐含每日全额换手，却按零成本计；"
                      "策略扣成本、基准不扣，超额被系统性低估（或高估）。",
                      "改用零换手的买入持有基准，或给基准加同等成本。")
        if btype in ("buy_hold", "index", "bh"):
            return _r("E2", "成本对称", "PASS",
                      f"基准类型 {btype}，零换手零成本，成本不对称问题不成立。")
        return _r("E2", "成本对称", "WARN",
                  f"基准未扣成本，类型未声明（{btype}）。",
                  "声明基准类型；若是每日再平衡口径，必须改买入持有。")
    # applied_to_benchmark 缺失或无法识别：不能默认「都扣了」
    if btype in ("ew_daily", "strategy_like", "daily_rebalance"):
        return _r("E2", "成本对称", "FAIL",
                  f"基准类型 {btype} 隐含每日换手，但未声明基准是否扣成本。",
                  'meta.cost 补 "applied_to_benchmark": true/false。')
    return _r("E2", "成本对称", "WARN",
              f"未声明基准是否扣成本（applied_to_benchmark={applied!r}）。",
              'meta.cost 补 "applied_to_benchmark": true/false。')


# ---------------- 3. 股票池 ----------------

def check_universe(b: Bundle):
    mode = str(b.meta.get("universe_mode", "unknown")).lower()
    if mode == "pit":
        return _r("E3", "股票池时点性", "PASS", "逐日时点池（只用当日及以前信息）。")
    if mode == "static":
        return _r("E3", "股票池时点性", "FAIL",
                  "静态池：用末端（或全期）流动性/市值挑票，等于用未来信息筛历史成分。",
                  "逐日重建池子：上市满 N 日 + 当日成交额分位；禁止用最后一天的排名。")
    return _r("E3", "股票池时点性", "FAIL",
              "未声明股票池构造方式（universe_mode 缺失）。",
              "meta 里写 pit / static；写不清就按 static 处理。")


# ---------------- 4. 调仓相位 ----------------

def check_phase(b: Bundle):
    phases = b.meta.get("phases_run")
    if phases is None:
        return _r("E4", "调仓相位", "FAIL",
                  "未声明相位（phases_run 缺失）；单批建仓的年度超额可因起点翻符号。",
                  "至少跑 4–21 个相位，报告相位带或相位中性均值。")
    try:
        phases = int(phases)
    except (TypeError, ValueError):
        return _r("E4", "调仓相位", "WARN", f"phases_run 不是整数：{phases!r}。")
    if phases >= 5:
        return _r("E4", "调仓相位", "PASS", f"跑了 {phases} 个相位。")
    if phases >= 2:
        return _r("E4", "调仓相位", "WARN",
                  f"只跑了 {phases} 个相位，相位带可能很宽。",
                  "补到 5 个以上相位，或报告最小值/最大值。")
    return _r("E4", "调仓相位", "FAIL",
              "单一相位（phases_run=1）：结果依赖首次调仓日，不可作为证据。",
              "改为多相位平均；至少 4 个，最好覆盖整个调仓周期。")


# ---------------- 5. 面板固定 ----------------

def check_panel(b: Bundle):
    fixed = b.meta.get("panel_fixed")
    trail_hist = b.meta.get("panel_trail_history")
    if isinstance(trail_hist, (list, tuple)) and len(set(trail_hist)) > 1:
        return _r("E5", "面板固定", "FAIL",
                  f"同一次比较里改过回看窗口：{sorted(set(trail_hist))}；"
                  "EMA 类特征有无限记忆，分数不可比。",
                  "固定面板起点与回看窗口；只移动评估起点，不改 --trail。")
    if fixed is True:
        return _r("E5", "面板固定", "PASS", "面板起点与窗口固定。")
    if fixed is False:
        return _r("E5", "面板固定", "FAIL",
                  "面板随参数/窗口漂移，同一日期在不同设置下算出不同分数。",
                  "固定面板；把评估窗口的移动与特征窗口的移动分开。")
    return _r("E5", "面板固定", "WARN",
              "未声明面板是否固定（panel_fixed 缺失）。",
              "meta 里写 panel_fixed: true，并记录窗口长度。")


# ---------------- 6. 年化口径 ----------------

def check_annualization(b: Bundle):
    r = b.strategy
    geo = mt.ann_geometric(r)
    ari = mt.ann_arithmetic(r)
    reported = b.meta.get("reported_ann")
    try:
        reported = float(reported) if reported is not None else None
    except (TypeError, ValueError):
        reported = None
    days = b.meta.get("sample_days")
    if days is not None:
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = None
    if days is not None and days != len(r):
        return _r("E6", "年化口径", "FAIL",
                  f"报告样本 {days} 天，日收益文件 {len(r)} 天，两者不是同一个样本，"
                  "年化不可直接比较。",
                  "让报告里的样本区间与日收益文件一致；或在 meta 里写清实际评估区间。")
    if not (np.isfinite(geo) and np.isfinite(ari)):
        return _r("E6", "年化口径", "WARN",
                  "策略没有有效收益（或收益 ≤ -100%），年化无法计算；"
                  "报告值与日收益文件无法核对。",
                  "先修好日收益序列（缺失/异常值），再核对报告年化。")
    detail = f"策略几何年化 {_pct(geo)}，算术年化 {_pct(ari)}。"
    if reported is None or not np.isfinite(float(reported)):
        return _r("E6", "年化口径", "WARN",
                  detail + " 未给 reported_ann，无法核对报告值。",
                  "在 meta 里写 reported_ann；主表一律用几何。")
    reported = float(reported)
    if abs(reported - geo) > 0.005:
        hint = "，更接近算术年化" if abs(reported - ari) < abs(reported - geo) else ""
        return _r("E6", "年化口径", "FAIL",
                  detail + f" 报告值 {_pct(reported)}{hint}，与逐日复利不符。",
                  "统一用几何（复利）年化；算术年化只能作辅助列。")
    return _r("E6", "年化口径", "PASS", detail + f" 报告值 {_pct(reported)} 与几何一致。")


# ---------------- 7. 多重检验 ----------------

def check_multiple_testing(b: Bundle):
    n = b.meta.get("n_trials", 1)
    try:
        n = max(1, int(n))
    except (TypeError, ValueError):
        return _r("E7", "多重检验", "WARN", f"n_trials 非法：{n!r}。")
    # trials.csv 的列数也是「试过几次」的证据：meta 少报时按实际列数算（同 E10）
    n = max(n, _n_trials(b))
    t = b.meta.get("reported_t")
    try:
        t = float(t) if t is not None else None
    except (TypeError, ValueError):
        t = None
    if t is None or not np.isfinite(t):
        return _r("E7", "多重检验", "WARN",
                  f"声明了 {n} 组尝试，但 reported_t 缺失或非法"
                  f"（{b.meta.get('reported_t')!r}），无法核对显著性。",
                  "报告主检验的 t 值，并按尝试次数做 Bonferroni 校正。")
    thr = mt.bonferroni_z(n)
    if abs(t) < thr:
        return _r("E7", "多重检验", "FAIL",
                  f"{n} 组尝试的 Bonferroni 门槛 |t|>{thr:.2f}，实测 t={t:.2f}，不显著。",
                  "减少尝试次数，或把结果降级为探索性；不要从网格里挑最好的一组当结论。")
    return _r("E7", "多重检验", "PASS",
              f"{n} 组尝试，门槛 |t|>{thr:.2f}，实测 t={t:.2f}。")


# ---------------- 8. 幸存者偏差 ----------------

def check_survivorship(b: Bundle):
    inc = b.meta.get("delisted_included")
    if inc is True:
        return _r("E8", "幸存者偏差", "PASS", "池子包含退市/已消失标的。")
    if inc is False:
        return _r("E8", "幸存者偏差", "FAIL",
                  "池子只有当前在市标的；退市股缺失会系统性高估历史收益。",
                  "补退市股，或在结论里把它列为不可量化偏差，并给出方向（通常不利于策略）。")
    return _r("E8", "幸存者偏差", "WARN",
              "未声明是否包含退市股（delisted_included 缺失）。",
              "meta 里写 true/false；写不清按'不含'处理并强制披露。")


# ---------------- 9. 成交时点可执行性（E9，2026-11-12 新增） ----------------

# 收盘后才完整的价格：用它们做信号，就不能再用同一时点的价格成交
_AFTER_CLOSE = {
    "close_t", "close", "收盘", "high_t", "low_t", "volume_t", "amount_t", "vwap_t",
    "high", "low", "volume", "amount", "vwap",
}
# 同一交易日内的成交价
_SAME_DAY_TRADE = {"open_t", "close_t", "open", "close", "开盘", "收盘", "收盘价",
                   "high_t", "low_t", "vwap_t", "amount_t", "volume_t",
                   "high", "low", "vwap", "amount", "volume"}


def check_execution_timing(b: Bundle):
    """信号用了某个价格，成交价就不能是同一个时点的同一价格。

    判据：若 signal_uses 是收盘后才确认的价格（close_t / high_t / volume_t / vwap_t …），
    而 trade_price 是同一交易日内的价格（open_t / close_t），即视为前视成交。
    """
    sig = str(b.meta.get("signal_uses") or "").strip().lower()
    trd = str(b.meta.get("trade_price") or "").strip().lower()

    if not sig or not trd:
        return _r("E9", "成交时点可执行性", "WARN",
                  "未声明信号价与成交价（signal_uses / trade_price），无法判断是否存在前视成交。",
                  'meta 里补 "signal_uses": "close_t"、"trade_price": "open_t+1"（次日开盘）。')

    if sig == trd:
        return _r("E9", "成交时点可执行性", "FAIL",
                  f"信号与成交价都用 `{sig}`——同一时点同一价格，不可能成交。",
                  "成交价改成 open_t+1（次日开盘）。")

    if sig in _AFTER_CLOSE and trd in _SAME_DAY_TRADE:
        return _r("E9", "成交时点可执行性", "FAIL",
                  f"信号用 `{sig}`（收盘后才完整），成交价也用 `{trd}`（同一交易日）——"
                  "等于用收盘后才知道的价格在收盘时成交。",
                  "把成交价改成次日开盘（open_t+1），或把信号改成盘中可得的价格；"
                  "只用收盘→次日收盘这一段收益，等于把不可执行的一段算进了业绩。")

    _known_sig = _AFTER_CLOSE | {"open_t", "open", "开盘"}
    _known_trd = _SAME_DAY_TRADE | {"open_t+1", "open+1", "次日开盘", "next_open"}
    if sig not in _known_sig or trd not in _known_trd:
        return _r("E9", "成交时点可执行性", "WARN",
                  f"无法识别的口径：信号 `{sig}`、成交 `{trd}`。",
                  '用 "close_t" / "open_t" 这类标准写法；成交写 "open_t+1"。')

    return _r("E9", "成交时点可执行性", "PASS",
              f"信号 `{sig}`、成交 `{trd}`：成交价在信号确认之后，时序合法。")


# ---------------- 10. 搜索规模折价（E10，2026-11-16 新增） ----------------

def _n_trials(b: Bundle) -> int:
    """尝试次数：meta 声明与 trials 实际列数取大者。

    meta 写 1 但 trials.csv 里有 50 列时，按 50 算——否则「网格里挑最好」
    会被当成单次检验，DSR 直接 PASS。
    """
    try:
        meta_n = max(1, int(b.meta.get("n_trials", 1) or 1))
    except (TypeError, ValueError):
        meta_n = 1
    t = getattr(b, "trials", None)
    if t is not None and getattr(t, "ndim", 0) == 2:
        try:
            meta_n = max(meta_n, int(t.shape[1]))
        except (IndexError, TypeError):
            pass
    return meta_n


def check_deflated_sharpe(b: Bundle):
    """DSR：把「试了 N 次」和收益的偏度峰度一起折进去，再看 SR 还显著吗。"""
    try:
        st = of.sr_stats(b.strategy)
    except ValueError as e:
        return _r("E10", "搜索规模折价（DSR）", "WARN",
                  f"无法计算：{e}。", "补足有效日收益（至少 3 条、方差非 0）。")

    T, sr, N = st["T"], st["sr"], _n_trials(b)
    sr0 = of.expected_max_sharpe(T, N)
    try:
        dsr = of.deflated_sharpe_ratio(sr, T, N, st["skew"], st["kurt"])
    except ValueError as e:
        return _r("E10", "搜索规模折价（DSR）", "WARN",
                  f"无法计算：{e}。", "检查收益序列的偏度/峰度是否异常。")
    ann = sr * math.sqrt(mt.TRADING_DAYS)
    head = (f"{N} 次尝试下，纯噪声的门槛 SR0={sr0:.4f}/期，"
            f"实测 SR={sr:.4f}/期（年化 {ann:.2f}），DSR={dsr:.3f}。")
    tail = f"偏度 {st['skew']:.2f}、峰度 {st['kurt']:.2f}，T={T} 期。"

    if dsr >= 0.95:
        return _r("E10", "搜索规模折价（DSR）", "PASS", head + " 扣掉搜索规模后仍显著。" + tail)

    try:
        need = of.min_track_record_length(sr, N, skew=st["skew"], kurt=st["kurt"])
    except ValueError:
        need = float("nan")
    if sr <= 0:
        need_txt = "这个 SR 不为正，再长的样本也验不出来。"
    elif not np.isfinite(need):
        need_txt = (f"同样的 SR 和尝试次数，在 {of.MTRL_CAP} 期内都达不到 "
                    f"DSR 0.95，现有 {T} 期。")
    else:
        need_txt = (f"同样的 SR 和尝试次数，需要约 {need:.0f} 期才能让 DSR 过 0.95，"
                    f"现有 {T} 期。")

    if dsr >= 0.90:
        return _r("E10", "搜索规模折价（DSR）", "WARN",
                  head + " 扣掉搜索规模后临界显著，不足以当结论。" + tail,
                  need_txt + " 补样本，或把结论降级为探索性。")
    return _r("E10", "搜索规模折价（DSR）", "FAIL",
              head + " 扣掉搜索规模后不显著——这个 SR 可能是搜索挑出来的。" + tail,
              need_txt + " 减少尝试次数、延长样本，或换一个不依赖挑选的信号。")


# ---------------- 11. 回测过拟合概率（E11） ----------------

def _trial_matrix(b: Bundle):
    if b.trials is None:
        return None, "未提供候选配置收益矩阵"
    R = np.asarray(b.trials, dtype=float)
    if R.ndim != 2 or R.shape[1] < 2:
        return None, "候选矩阵列数不足 2"
    # 死列（全 NaN / 常量）必须剔除，否则会把 PBO 的分母撑大、把 PBO 压低
    valid = np.isfinite(R).sum(0) >= max(8, int(0.5 * R.shape[0]))
    varying = np.zeros(R.shape[1], dtype=bool)
    for j in np.flatnonzero(valid):
        col = R[:, j]
        col = col[np.isfinite(col)]
        varying[j] = col.size > 1 and col.std() > 1e-12
    keep = valid & varying
    if keep.sum() < 2:
        return None, (f"候选矩阵只有 {int(keep.sum())} 列是有效且有波动的"
                      "（其余为全 NaN 或常量）")
    R = R[:, keep]
    rows = np.isfinite(R).all(1)
    if rows.sum() < 8:
        return None, f"对齐后只剩 {int(rows.sum())} 行有效收益，不足以算 PBO"
    return R[rows], ""


def check_pbo(b: Bundle):
    """PBO：样本内最优的那组配置，在样本外排到中位数以下的频率。"""
    R, why = _trial_matrix(b)
    if R is None:
        return _r("E11", "回测过拟合概率（PBO）", "INFO",
                  f"跳过：{why}（单条回测无法计算过拟合概率）。",
                  "把跑过的全部候选配置写成 trials.csv（每列一组日收益），"
                  "放在回测包目录里，这一项才会真正执行。")

    T, N = R.shape
    if T < 16:
        return _r("E11", "回测过拟合概率（PBO）", "INFO",
                  f"跳过：候选矩阵只有 {T} 期，至少需要 16 期才能切块。")

    n_blocks = 16 if T >= 160 else 8
    try:
        res = of.pbo_cscv(R, n_blocks=n_blocks)
    except ValueError as e:
        return _r("E11", "回测过拟合概率（PBO）", "WARN", f"无法计算：{e}。")

    pbo = res["pbo"]
    detail = (f"{N} 组候选 × {T} 期，切成 {n_blocks} 块枚举 {res['n_splits']} 次切分："
              f"PBO={pbo:.3f}（样本内最优在样本外掉到中位数以下的频率）。")
    if pbo <= 0.20:
        return _r("E11", "回测过拟合概率（PBO）", "PASS", detail + " 选择偏差可控。")
    if pbo <= 0.50:
        return _r("E11", "回测过拟合概率（PBO）", "WARN",
                  detail + " 有过拟合迹象：最优配置多半是挑出来的。",
                  "报告全部候选的结果，而不是只报最好的一组；或换用样本外的选择规则。")
    return _r("E11", "回测过拟合概率（PBO）", "FAIL",
              detail + " 样本内最优在样本外基本没有优势——这组结果不能当证据。",
              "按预注册口径重跑，或把结论降级为「未发现可用的配置」。")


# ---------------- 12. 样本外衰减（E12） ----------------

def check_cpcv_decay(b: Bundle):
    """CPCV：净化切分后，样本内最优配置在样本外还赚不赚钱。"""
    R, why = _trial_matrix(b)
    if R is None:
        return _r("E12", "样本外衰减（CPCV）", "INFO",
                  f"跳过：{why}（需要多组候选才能做样本外选择）。",
                  "同 E11：提供 trials.csv 后这一项才会执行。")

    T, N = R.shape
    if T < 6:
        return _r("E12", "样本外衰减（CPCV）", "INFO",
                  f"跳过：只有 {T} 期，不足以切成 6 组。")

    try:
        # 净化窗按持仓期取：持有 N 日的策略，训练/测试边界要空出 N 期
        try:
            lh = max(0, int(b.meta.get("holding_period", 1) or 1))
        except (TypeError, ValueError):
            lh = 1
        res = of.cpcv_decay(R, n_groups=6, k_test=2, label_horizon=lh)
    except ValueError as e:
        return _r("E12", "样本外衰减（CPCV）", "WARN", f"无法计算：{e}。")

    detail = (f"{N} 组候选 × {T} 期，6 组取 2 组的净化切分共 {res['n_splits']} 次"
              f"（可拼 {res['n_paths']} 条完整样本外路径）："
              f"样本内最优在样本外的中位 SR={res['median_oos_sr']:.4f}/期，"
              f"{res['decay']:.1%} 的切分样本外不赚钱，"
              f"共选出 {res['n_distinct_picks']} 个不同的最优配置。")

    if res["decay"] <= 0.10:
        return _r("E12", "样本外衰减（CPCV）", "PASS", detail + " 样本内外方向一致。")
    if res["decay"] <= 0.30:
        return _r("E12", "样本外衰减（CPCV）", "WARN", detail + " 有不稳定性。",
                  "检查最优配置随时间是否漂移；漂移大就把选择规则固定下来再跑。")
    return _r("E12", "样本外衰减（CPCV）", "FAIL",
              detail + " 样本内挑出来的配置在样本外多数不赚钱。",
              "这组候选不能当证据：要么拉长样本，要么把「挑配置」这一步写进预注册。")


# ---------------- 13/14. 基础健全性 ----------------

def check_returns_sanity(b: Bundle):
    r = b.strategy
    n = len(r)
    nan_ratio = float(np.mean(~np.isfinite(r))) if n else 1.0
    finite = r[np.isfinite(r)]
    if len(finite) < 20:
        return _r("S1", "收益序列健全性", "FAIL", f"有效日收益只有 {len(finite)} 条。")
    if np.std(finite) < 1e-12:
        return _r("S1", "收益序列健全性", "FAIL", "策略日收益方差为 0，疑似没有真实成交。")
    if nan_ratio > 0.05:
        return _r("S1", "收益序列健全性", "WARN",
                  f"{nan_ratio:.1%} 的日收益缺失。", "检查停牌/未上市/估值日为 NaN 的处理。")
    extreme = finite[np.abs(finite) > 0.5]
    if len(extreme):
        return _r("S1", "收益序列健全性", "WARN",
                  f"出现 {len(extreme)} 天单日涨跌超过 30%，先排查复权/拆股错误。",
                  "检查复权因子对齐；一个对齐 bug 就能造出几千倍假收益。")
    return _r("S1", "收益序列健全性", "PASS",
              f"{len(finite)} 天有效，最大单日 {_pct(finite.max())}，最小 {_pct(finite.min())}。")


def check_benchmark_sanity(b: Bundle):
    if b.benchmark is None:
        return _r("S3", "基准序列健全性", "FAIL", "缺基准列。")
    r = b.benchmark
    finite = r[np.isfinite(r)]
    if len(finite) < 20:
        return _r("S3", "基准序列健全性", "FAIL", f"基准只有 {len(finite)} 条有效值。")
    if np.std(finite) < 1e-12:
        return _r("S3", "基准序列健全性", "FAIL", "基准日收益方差为 0。")
    extreme = finite[np.abs(finite) > 0.3]
    if len(extreme):
        worst = float(finite[np.argmax(np.abs(finite))])
        return _r("S3", "基准序列健全性", "FAIL",
                  f"基准出现 {len(extreme)} 天单日涨跌超过 30%（最大绝对 {_pct(worst)}），"
                  "等权组合不可能有这种日收益——先排查日期对齐/复权/成分污染。",
                  "逐日核对基准的构造：日期是否错位、是否混入单只股票的极端收益、是否把净值当收益。")
    large = finite[np.abs(finite) > 0.15]
    if len(large):
        worst = float(finite[np.argmax(np.abs(finite))])
        return _r("S3", "基准序列健全性", "WARN",
                  f"基准有 {len(large)} 天单日涨跌超过 15%（最大绝对 {_pct(worst)}），先确认口径。",
                  "检查基准的极端日：是市场真实波动，还是数据对齐问题。")
    return _r("S3", "基准序列健全性", "PASS",
              f"{len(finite)} 天有效，最大单日 {_pct(finite.max())}，最小 {_pct(finite.min())}。")


def check_data_alignment(b: Bundle):
    if b.benchmark is None:
        return _r("S2", "对齐", "FAIL", "缺基准列。")
    s, m = b.strategy, b.benchmark
    if len(s) == 0:
        return _r("S2", "对齐", "FAIL", "序列为空，无法判断对齐。",
                  "检查 daily.csv 是否读进来了。")
    ok = np.isfinite(s) & np.isfinite(m)
    ratio = float(ok.mean())
    if not np.isfinite(ratio):
        return _r("S2", "对齐", "FAIL", "对齐比例无法计算（全为缺失）。",
                  "检查策略/基准列是否是数值。")
    if ratio < 0.95:
        return _r("S2", "对齐", "FAIL",
                  f"策略与基准只有 {ratio:.1%} 的日期同时有效。",
                  "对齐交易日历；不要在缺失日上补 0 或前向填充收益。")
    return _r("S2", "对齐", "PASS", f"{ratio:.1%} 的日期同时有效。")


# ---------------- 17. 规模变量口径（E13，2026-11-18 新增） ----------------

# 明确表示「市值」的写法（不含裸 cap，否则 capital/capacity 会误命中）
_SIZE_CAP_TOKENS = {"market_cap", "marketcap", "mcap", "市值", "总市值", "流通市值"}
# 股本/股数：**单独出现不是市值**
_SIZE_SHARE_TOKENS = {"shares", "total_shares", "float_shares", "股本", "总股本",
                      "流通股本", "股数"}
# 价格
_SIZE_PRICE_TOKENS = {"close", "adj_close", "close_t", "price", "px",
                      "收盘", "收盘价", "股价", "价格"}


def check_size_variable(b: Bundle):
    """规模中性化用的必须是市值，不能是价格。

    价格与市值相关但不等价。用收盘价当市值做「规模中性化」，
    实际做的是价格中性化，两者会给出不同的结果，甚至凭空制造出显著性。
    """
    raw = str(b.meta.get("size_variable") or "").strip()
    neut = b.meta.get("size_neutralized")

    if not raw:
        if neut is True:
            return _r("E13", "规模变量口径", "FAIL",
                      "声明做了规模中性化，但没写 size_variable，无法核实用的是市值还是价格。",
                      '补 "size_variable": "total_shares * close"。')
        if neut is False:
            return _r("E13", "规模变量口径", "PASS",
                      "已声明未做规模中性化，本项不适用。")
        return _r("E13", "规模变量口径", "WARN",
                  "未声明是否做规模中性化，也没写 size_variable。",
                  '做过的写 "size_neutralized": true，"size_variable": "total_shares * close"；'
                  '没做过的写 "size_neutralized": false。')

    v = raw.lower()
    has_cap = any(t in v for t in _SIZE_CAP_TOKENS)
    has_share = any(t in v for t in _SIZE_SHARE_TOKENS)
    has_price = any(t in v for t in _SIZE_PRICE_TOKENS)

    if has_cap:
        return _r("E13", "规模变量口径", "PASS",
                  f"规模变量 `{raw}`：明确写了市值口径。")
    if has_share and has_price:
        return _r("E13", "规模变量口径", "PASS",
                  f"规模变量 `{raw}`：股本 × 价格，即市值，可用。")
    if has_price:
        return _r("E13", "规模变量口径", "FAIL",
                  f"规模变量写成 `{raw}`——这是价格，不是市值。"
                  "用它做「规模中性化」实际是价格中性化，结论不可采信。",
                  '改成 "total_shares * close"。价格与市值相关但不等价：'
                  "用价格中性化会残留真实的规模暴露，可能把规模溢价伪装成 alpha。")
    if has_share:
        return _r("E13", "规模变量口径", "FAIL",
                  f"规模变量写成 `{raw}`——这是股数，不是市值。"
                  "股数相同的两家公司市值可以差十倍。",
                  '写成 "total_shares * close"，把价格乘进去。')
    return _r("E13", "规模变量口径", "WARN",
              f"无法识别规模变量 `{raw}`。",
              '用 "total_shares * close" 或 "market_cap" 这类可核实的写法。')


ALL_CHECKS = [
    check_benchmark_present,
    check_benchmark_accounting,
    check_cost_symmetry,
    check_universe,
    check_phase,
    check_panel,
    check_annualization,
    check_multiple_testing,
    check_survivorship,
    check_execution_timing,
    check_deflated_sharpe,
    check_pbo,
    check_cpcv_decay,
    check_returns_sanity,
    check_benchmark_sanity,
    check_data_alignment,
    check_size_variable,
]


def run_all(b: Bundle) -> list[dict]:
    return [fn(b) for fn in ALL_CHECKS]
