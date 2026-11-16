# backtest_lint —— 回测体检工具

> **A CLI that audits a backtest before you believe it.**
> 9 classes of measurement error (E1–E9) turned into runnable checks.
> Input: daily returns + a small `meta.json`. Output: a PASS / WARN / FAIL report card.
>
> ```bash
> pip install -r requirements.txt
> python -m backtest_lint --bundle examples/bad    # 故意踩坑的示例：9 项阻断
> python -m backtest_lint --bundle examples/good   # 干净示例：全部通过
> python -m pytest tests -q                        # 25 passed
> ```

> 把九类测量误差（含 2026-11-12 新增的 E9）做成一个能跑的 CLI：输入一份回测的日收益与元数据，
> 输出 **PASS / WARN / FAIL** 报告卡。
>
> 它不判断策略未来能不能赚钱，只检查**这份回测有没有可能骗人**。

---

## 为什么做这个

一个 A 股短期反转策略，最初报告 **+10.42%/年、IR 0.31、六年全正**。
逐项修正后，同一份数据、同一套成本，变成 **−1.76%/年、IR −0.04**；
换成零成本买入持有基准，是 **−5.61%/年、六年五年跑输**。

九类误差中有八类全部朝同一个方向——让策略看起来比实际更好：

| # | 误差 | 实测幅度（该项目） |
|---|---|---|
| E1 | 基准记账：`exp(mean(log r))` 不是等权组合收益 | 基准被压低 **10.87pp/年** |
| E2 | 基准每日再平衡（100% 日换手）却按零成本计 | 与策略成本不对称 |
| E3 | 静态池：用末端流动性挑历史成分 | 同窗口基准 +22.71% vs −7.07%，差 **29.8pp** |
| E4 | 单批建仓的调仓相位 | 年度超额可从 **−20.6% 到 +17.4%** 翻符号 |
| E5 | 面板/回看窗口漂移（EMA 无限记忆） | 同一日期算出不同分数 |
| E6 | 算术年化与几何年化混用、报告区间与日收益文件不一致 | σ=3% 时约 **12pp/年** |
| E7 | 参数网格不校正多重检验 | 18 组网格 Bonferroni 门槛 \|t\|>2.99 |
| E8 | 退市股缺失（幸存者偏差） | 方向通常不利于策略，无法量化 |
| **E9** | **成交时点可执行性**：信号用的是收盘价，成交价也按收盘价算 | 实测：同一信号换成交口径，毛年化 **−42.8% ↔ +50.5%**（差 93pp） |

> E9 是 2026-11-12 新增的第九类。前八类全部让策略「看起来更好」，E9 两个方向都可能，所以不并入上面那句总结。

这个工具把每一条变成可执行的检查。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1) 跑一个故意踩坑的示例（9 项阻断）
python -m backtest_lint --bundle examples/bad

# 2) 跑一个干净示例（全部通过）
python -m backtest_lint --bundle examples/good

# 3) 生成一份空白模板
python -m backtest_lint --init my_bundle
```

> 还没发到 PyPI，上面是从源码直接跑；`pip install .` 之后也可以改用 `backtest-lint` 命令。
> 在 Windows 的 cmd / PowerShell 里跑也没问题，报告里的 ✅/⚠️/❌ 会按控制台编码自动降级。

输出 Markdown 报告卡；`--md` / `--json` 可另存。有 FAIL 时退出码为 1，可直接接 CI。

```bash
python -m backtest_lint --bundle my_bundle --md report.md --json report.json
```

---

## 输入格式

一个目录，两个文件：

### `daily.csv`

```csv
date,strategy,benchmark
2024-01-02,0.0012,0.0008
2024-01-03,-0.0005,0.0001
```

- `date` 可省略；`strategy` 必填；`benchmark` 缺失会直接判 FAIL。
- 列名支持中文别名：`日期` / `策略` / `基准`。
- 收益是**日简单收益**（0.0012 = 0.12%），不是净值。

### `meta.json`

```json
{
  "name": "我的策略",
  "universe_mode": "pit",
  "delisted_included": false,
  "benchmark_type": "buy_hold",
  "cost": {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001,
           "applied_to_benchmark": false},
  "phases_run": 21,
  "panel_fixed": true,
  "panel_trail_history": [2700],
  "n_trials": 1,
  "sample_days": 1425,
  "reported_t": 3.2,
  "reported_ann": 0.12,
  "reported_bench_ann": 0.05
}
```

| 字段 | 取值 | 说明 |
|---|---|---|
| `universe_mode` | `pit` / `static` / `unknown` | 股票池是否逐日时点 |
| `delisted_included` | `true` / `false` | 池子是否含退市股 |
| `benchmark_type` | `buy_hold` / `index` / `ew_daily` / `strategy_like` | 基准类型 |
| `cost.applied_to_benchmark` | `true` / `false` | 基准是否同样扣成本 |
| `phases_run` | 整数 | 跑了几个调仓相位 |
| `panel_fixed` | `true` / `false` | 面板起点/回看窗口是否固定 |
| `panel_trail_history` | 数组 | 同一次比较里用过的窗口；多个值直接 FAIL |
| `n_trials` | 整数 | 参数网格/变体尝试次数 |
| `sample_days` | 整数 | 报告里的样本天数；与日收益文件不一致会 FAIL |
| `reported_t` | 浮点 | 主检验 t 值 |
| `reported_ann` | 浮点 | 报告的策略年化 |
| `reported_bench_ann` | 浮点 | 报告的基准年化 |
| `signal_uses` | `close_t` / `high_t` / `vwap_t` … | 信号用了哪个时点的价格（E9） |
| `trade_price` | `open_t` / `close_t` / `open_t+1` … | 成交价用的哪个时点（E9） |

---

## 检查项与判据

| # | 检查 | PASS | WARN | FAIL |
|---|---|---|---|---|
| B0 | 基准存在 | 有 benchmark 列 | — | 缺基准 |
| E1 | 基准记账 | 报告值与几何一致或差 <1pp | 几何/算术差 >1pp | 报告值更接近算术且差 >0.5pp |
| E2 | 成本对称 | 基准也扣成本，或基准是买入持有/指数 | 基准类型未声明 | `ew_daily`/`strategy_like` 且基准零成本 |
| E3 | 股票池时点性 | `pit` | — | `static` 或未声明 |
| E4 | 调仓相位 | ≥5 个相位 | 2–4 个 | 1 个或未声明 |
| E5 | 面板固定 | `panel_fixed=true` 且窗口唯一 | 未声明 | `false` 或窗口有多个值 |
| E6 | 年化口径 | 报告年化与几何一致 | 未给报告值 | 差 >0.5pp，或样本天数不一致 |
| E7 | 多重检验 | \|t\| > Bonferroni 门槛 | 未给 t | \|t\| < 门槛 |
| E8 | 幸存者偏差 | 含退市股 | 未声明 | 只有当前在市标的 |
| **E9** | **成交时点可执行性** | 成交价在信号确认之后（如 `close_t` → `open_t+1`） | 未声明 `signal_uses`/`trade_price` | 信号用收盘后价格、成交也用同一交易日价格 |
| S1 | 收益健全性 | 无异常 | 缺失 >5% 或单日 >50% | 有效样本 <20 或方差为 0 |
| S2 | 对齐 | ≥95% 日期同时有效 | — | <95% |
| S3 | 基准健全性 | 无异常 | 单日 >15% | 单日 >30% 或有效样本 <20 |

Bonferroni 门槛：`z = Φ⁻¹(1 − 0.025/n_trials)`；n=18 时约 **2.99**，n=1 时 **1.96**。

---

## 真实案例：V2.0 反转策略的体检报告

`examples/v2_reversal/` 是项目里真实跑过的结果（2020-12 ~ 2026-10，时点池，错峰 4）。
体检结论：**4 项阻断、4 项警告**：

| 检查 | 结果 | 说明 |
|---|---|---|
| E2 | ❌ | 基准是每日再平衡等权（100% 日换手）却零成本 |
| E6 | ❌ | 报告样本 **1545** 天，日收益文件 **1425** 天，不是同一个样本 |
| E7 | ❌ | 18 组网格，门槛 \|t\|>2.99，实测 t=0.80 |
| E8 | ❌ | 只有当前在市标的，退市股缺失 |
| E1 | ⚠️ | 基准几何 −5.95% vs 算术 −2.55%，差 3.40pp，报告未声明口径 |
| E4 | ⚠️ | 只跑了 4 个相位 |
| E9 | ⚠️ | meta 未声明 `signal_uses` / `trade_price`，成交时点无法判断 |
| S3 | ⚠️ | 基准有 1 天单日 **+18.70%**，先确认是真实波动还是对齐问题 |

完整报告见 `examples/v2_reversal_report.md`。
**E6 那条尤其值得看**：报告正文的 −6.24% 和日收益文件里的 −7.87%，不是同一个样本——
这种"数字对不上"在没有工具时很难被发现。

---

## 目录

```
backtest-lint/
├─ .github/workflows/  CI：Python 3.10–3.13 跑 25 个单测
├─ backtest_lint/
│  ├─ checks.py      九类误差（E1–E9）+ 四项健全性检查
│  ├─ metrics.py     几何/算术年化、NW、Bonferroni
│  ├─ loader.py      daily.csv + meta.json 读取与模板
│  ├─ report.py      Markdown / JSON 报告
│  └─ cli.py         命令行入口
├─ examples/
│  ├─ bad/           故意踩坑（9 项阻断）
│  ├─ good/          干净示例（全部通过）
│  └─ v2_reversal/   项目真实结果（4 阻断 4 警告）
├─ tests/            25 个单测
├─ requirements.txt
└─ README.md
```

---

## 测试

```bash
python -m pytest tests -q
# 25 passed
```

---

## 局限（写论文/交付时必须说）

1. **只检查可观测的口径**：特征前视、执行前视、代码 bug（如复权因子对齐错误）
   需要读代码或对拍数据，工具查不出来。
2. **不判断未来收益**：通过体检 ≠ 能赚钱；它只排除"因为记账错误而看起来能赚"。
3. **meta 靠自报**：`universe_mode`、`phases_run` 等字段填错，检查就失效。
   工具能做的是把"必须声明"变成硬门槛。
4. **E1/E6 依赖报告值**：没有 `reported_ann` / `reported_bench_ann` 时只能给 WARN。
5. **幸存者偏差只能披露，不能修复**：本地没有退市股就是没有。

---

## 它适合谁

- 写完回测、准备对外报数字的人：先跑一遍，把阻断项消掉再说话。
- 看别人回测报告的人：把对方的日收益和参数填进来，看哪些项过不了。
- 教学/复试：九类误差每一条都能落到一个可运行的检查上，比背概念有用。

> 这个工具的全部价值，来自一个项目把自己的 **+10.42%** 亲手推翻到 **−1.76%** 的过程。
> 它不是"策略生成器"，是"策略证伪器"。
