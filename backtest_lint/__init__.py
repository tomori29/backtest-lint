# -*- coding: utf-8 -*-
"""回测体检工具（backtest_lint）。

把九类测量误差（E1–E9）做成可运行的检查：输入一份回测的日收益与元数据，
输出 PASS/WARN/FAIL 报告卡。
"""
from .checks import ALL_CHECKS, Bundle, run_all
from .loader import load_bundle, write_template

__all__ = ["Bundle", "ALL_CHECKS", "run_all", "load_bundle", "write_template"]
__version__ = "0.1.0"
