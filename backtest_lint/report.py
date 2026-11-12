# -*- coding: utf-8 -*-
"""报告渲染：Markdown / JSON。"""
from __future__ import annotations

import json

ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}


def counts(results: list[dict]) -> dict:
    c = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
    for r in results:
        c[r["status"]] = c.get(r["status"], 0) + 1
    return c


def verdict(results: list[dict]) -> str:
    c = counts(results)
    if c["FAIL"]:
        return f"不通过：{c['FAIL']} 项阻断、{c['WARN']} 项警告"
    if c["WARN"]:
        return f"带警告通过：{c['WARN']} 项警告"
    return "通过"


def to_markdown(results: list[dict], meta: dict | None = None) -> str:
    meta = meta or {}
    c = counts(results)
    lines = [
        "# 回测体检报告",
        "",
        f"- 标的：{meta.get('name', '（未命名）')}",
        f"- 结论：**{verdict(results)}**",
        f"- 统计：{c['PASS']} 通过 / {c['WARN']} 警告 / {c['FAIL']} 阻断",
        "",
        "| # | 检查项 | 状态 | 依据 |",
        "|---|---|---|---|",
    ]
    for r in results:
        detail = str(r["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r['id']} | {r['title']} | {ICON.get(r['status'], r['status'])} "
                     f"{r['status']} | {detail} |")
    fixes = [r for r in results if r["status"] in ("FAIL", "WARN") and r.get("fix")]
    if fixes:
        lines += ["", "## 修复建议", ""]
        for r in fixes:
            lines.append(f"- **{r['id']} {r['title']}**：{r['fix']}")
    lines += ["", "---", "",
              "本工具只检查'回测是否可能骗人'，不判断策略未来能否赚钱。",
               "E1–E8 全部朝同一个方向——让策略看起来比实际更好；E9（成交时点）两个方向都可能；"
              "任何一项未通过，结果都不能当作证据。"]
    return "\n".join(lines) + "\n"


def to_json(results: list[dict], meta: dict | None = None) -> str:
    return json.dumps({"meta": meta or {}, "verdict": verdict(results),
                       "counts": counts(results), "checks": results},
                      ensure_ascii=False, indent=2)
