# -*- coding: utf-8 -*-
"""九类测量误差检查（E1–E9）+ 四项基础健全性检查（2026-11-12 增补 E9）。

每条检查返回：
  id / title / status(PASS|WARN|FAIL|INFO) / detail / fix
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics as mt


@dataclass
class Bundle:
    daily: object                     # pandas.DataFrame: date, strategy, benchmark
    meta: dict = field(default_factory=dict)

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
    ok = np.isfinite(b.benchmark).sum()
    return _r("B0", "基准存在", "PASS",
              f"基准列有效 {ok}/{len(b.benchmark)} 天。")


def check_benchmark_accounting(b: Bundle):
    """基准年化：几何 vs 算术，差多少；报告值更像哪一个。"""
    if b.benchmark is None:
        return _r("E1", "基准记账口径", "FAIL", "缺基准，无法检查。", "补基准。")
    r = b.benchmark
    geo = mt.ann_geometric(r)
    ari = mt.ann_arithmetic(r)
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
    applied = cost.get("applied_to_benchmark")
    btype = str(b.meta.get("benchmark_type", "unknown")).lower()
    declared = all(k in cost for k in ("commission", "stamp", "slip"))
    if not declared:
        return _r("E2", "成本对称", "FAIL",
                  "meta.cost 缺 commission/stamp/slip，成本口径未声明。",
                  "写明佣金、印花税、滑点，以及基准是否同样扣成本。")
    if applied is False:
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
    return _r("E2", "成本对称", "PASS", "策略与基准均扣成本。")


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
    t = b.meta.get("reported_t")
    if t is None or not np.isfinite(float(t)):
        return _r("E7", "多重检验", "WARN",
                  f"声明了 {n} 组尝试，但没给 reported_t，无法核对显著性。",
                  "报告主检验的 t 值，并按尝试次数做 Bonferroni 校正。")
    t = float(t)
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
_SAME_DAY_TRADE = {"open_t", "close_t", "open", "close", "开盘", "收盘"}


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

    if sig in _AFTER_CLOSE and trd in _SAME_DAY_TRADE:
        return _r("E9", "成交时点可执行性", "FAIL",
                  f"信号用 `{sig}`（收盘后才完整），成交价也用 `{trd}`（同一交易日）——"
                  "等于用收盘后才知道的价格在收盘时成交。",
                  "把成交价改成次日开盘（open_t+1），或把信号改成盘中可得的价格；"
                  "只用收盘→次日收盘这一段收益，等于把不可执行的一段算进了业绩。")

    return _r("E9", "成交时点可执行性", "PASS",
              f"信号 `{sig}`、成交 `{trd}`：成交价在信号确认之后，时序合法。")


# ---------------- 10/11. 基础健全性 ----------------

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
    extreme = finite[np.abs(finite) > 0.3]
    if len(extreme):
        return _r("S1", "收益序列健全性", "WARN",
                  f"出现 {len(extreme)} 天单日涨跌超过 50%，先排查复权/拆股错误。",
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
        return _r("S3", "基准序列健全性", "FAIL",
                  f"基准出现 {len(extreme)} 天单日涨跌超过 30%（最大 {_pct(finite.max())}），"
                  "等权组合不可能有这种日收益——先排查日期对齐/复权/成分污染。",
                  "逐日核对基准的构造：日期是否错位、是否混入单只股票的极端收益、是否把净值当收益。")
    large = finite[np.abs(finite) > 0.15]
    if len(large):
        return _r("S3", "基准序列健全性", "WARN",
                  f"基准有 {len(large)} 天单日涨跌超过 15%（最大 {_pct(finite.max())}），先确认口径。",
                  "检查基准的极端日：是市场真实波动，还是数据对齐问题。")
    return _r("S3", "基准序列健全性", "PASS",
              f"{len(finite)} 天有效，最大单日 {_pct(finite.max())}，最小 {_pct(finite.min())}。")


def check_data_alignment(b: Bundle):
    if b.benchmark is None:
        return _r("S2", "对齐", "FAIL", "缺基准列。")
    s, m = b.strategy, b.benchmark
    ok = np.isfinite(s) & np.isfinite(m)
    ratio = float(ok.mean())
    if ratio < 0.95:
        return _r("S2", "对齐", "FAIL",
                  f"策略与基准只有 {ratio:.1%} 的日期同时有效。",
                  "对齐交易日历；不要在缺失日上补 0 或前向填充收益。")
    return _r("S2", "对齐", "PASS", f"{ratio:.1%} 的日期同时有效。")


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
    check_returns_sanity,
    check_benchmark_sanity,
    check_data_alignment,
]


def run_all(b: Bundle) -> list[dict]:
    return [fn(b) for fn in ALL_CHECKS]
