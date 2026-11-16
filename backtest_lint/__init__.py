# -*- coding: utf-8 -*-
"""回测体检工具（backtest_lint）。

把九类测量误差（E1–E9）和三类搜索规模折价（E10–E12）做成可运行的检查：
输入一份回测的日收益与元数据（可选的候选矩阵 trials.csv），输出 PASS/WARN/FAIL 报告卡。
"""
from .checks import ALL_CHECKS, Bundle, run_all
from .loader import load_bundle, write_template

__all__ = ["Bundle", "ALL_CHECKS", "run_all", "load_bundle", "write_template"]
__version__ = "0.3.0"
