# -*- coding: utf-8 -*-
"""命令行入口。

  python -m backtest_lint --bundle examples/bad
  python -m backtest_lint --bundle examples/bad --md report.md --json report.json
  python -m backtest_lint --init my_bundle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .checks import run_all
from .loader import load_bundle, write_template
from .report import counts, to_json, to_markdown, verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="backtest_lint",
        description="回测体检：九类测量误差（E1–E9）+ 四项健全性检查")
    ap.add_argument("--bundle", help="包含 daily.csv 与 meta.json 的目录")
    ap.add_argument("--md", help="Markdown 报告输出路径")
    ap.add_argument("--json", dest="json_out", help="JSON 报告输出路径")
    ap.add_argument("--init", metavar="DIR", help="生成一份模板回测包")
    ap.add_argument("--quiet", action="store_true", help="只输出结论行")
    args = ap.parse_args(argv)

    if args.init:
        daily, meta = write_template(args.init)
        print(f"已生成模板：{daily}")
        print(f"           {meta}")
        return 0

    if not args.bundle:
        ap.print_help()
        return 2

    bundle = load_bundle(args.bundle)
    results = run_all(bundle)
    md = to_markdown(results, bundle.meta)
    js = to_json(results, bundle.meta)

    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(js, encoding="utf-8")

    if not args.quiet:
        print(md)
    else:
        print(verdict(results))

    return 1 if counts(results)["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
