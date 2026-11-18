# backtest_lint — audit a backtest before you believe it

[![tests](https://github.com/tomori29/backtest-lint/actions/workflows/tests.yml/badge.svg)](https://github.com/tomori29/backtest-lint/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/backtest-lint)](https://pypi.org/project/backtest-lint/)
[![Downloads](https://img.shields.io/pypi/dm/backtest-lint)](https://pypi.org/project/backtest-lint/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*[中文](README.md) · English*

> Ten classes of measurement error (E1–E9, E13) plus three classes of search-size discount
> (E10–E12), turned into runnable checks.
> Input: daily returns + a small `meta.json` (+ optional `trials.csv`).
> Output: a PASS / WARN / FAIL report card.
>
> ```bash
> pip install backtest-lint
> python -m backtest_lint --bundle examples/bad          # a deliberately broken example: 10 blockers
> python -m backtest_lint --bundle examples/good         # a clean example: all pass
> python -m backtest_lint --bundle examples/multi_trial  # best of 24 configs: E1–E9 pass, E12 blocks
> python -m pytest tests -q                              # 86 passed
> ```
>
> It does not predict whether a strategy will make money. It asks one question:
> **could this backtest be lying to you?**

---

## Why this exists

A short-horizon A-share reversal strategy first reported **+10.42%/yr, IR 0.31, positive in all six years**.
After fixing the bookkeeping — same data, same costs — it became **−1.76%/yr, IR −0.04**.
Against a zero-cost buy-and-hold benchmark: **−5.61%/yr, underperforming in five of six years**.

Eight of the nine error classes push in the same direction — they make the strategy look better:

| # | Error | Measured effect (that project) |
|---|---|---|
| E1 | Benchmark bookkeeping: `exp(mean(log r))` is not an equal-weight portfolio return | benchmark understated by **10.87pp/yr** |
| E2 | Benchmark rebalanced daily (100% daily turnover) but costed at zero | cost asymmetry vs. the strategy |
| E3 | Static universe: picking historical constituents by end-of-sample liquidity | same window: **+22.71% vs −7.07%**, a **29.8pp** gap |
| E4 | Rebalance phase of a single entry date | annual excess flips sign from **−20.6% to +17.4%** |
| E5 | Panel / lookback window drift (unbounded EMA memory) | different scores for the same date |
| E6 | Mixing arithmetic and geometric annualization; report window disagreeing with the return file | ≈**12pp/yr** at σ=3% |
| E7 | Parameter grid without multiple-testing correction | 18-config grid: Bonferroni threshold \|t\|>2.99 |
| E8 | Missing delisted stocks (survivorship bias) | usually biased against the strategy, not quantifiable |
| **E9** | **Execution timing**: signal uses the close, fill is priced at the same close | switching fill convention moved gross annualized return **−42.8% ↔ +50.5%** (93pp) |
| **E10** | **Search-size discount (DSR)**: fold N trials into the significance bar | T=154 periods, 0.34 annual SR: DSR 0.887 at N=1, **0.41** at `n_trials=8` |
| **E11** | **Probability of backtest overfitting (PBO)**: how often the in-sample best lands below the out-of-sample median | the winner may be entirely a product of the search |
| **E12** | **Out-of-sample decay (CPCV)**: after purged splits, does the in-sample best still make money? | example bundle: 46.7% of splits lose money out of sample |
| **E13** | **Size-variable basis**: using close price as "market cap" for size neutralisation | same factor: price-neutralised Sharpe **1.23**, true market cap **0.64**; recent-half excess 4.84% → **0.20%** |

> E9 came last in the measurement-error group; E10–E12 were added in 2026-11-16.
> E1–E9 and E13 audit **inside one backtest**; E10–E12 audit **across the attempts behind it** —
> the report looks identical either way. A "good" result from one run and a "good" result
> picked from 42 runs are the same number on the page.

Every one of them is a runnable check.

---

## Quick start

```bash
# Option 1: from PyPI (recommended)
pip install backtest-lint

# Option 2: straight from the repo (tracks the latest commit)
pip install "git+https://github.com/tomori29/backtest-lint.git"

# Option 3: clone and run
git clone https://github.com/tomori29/backtest-lint && cd backtest-lint
pip install -r requirements.txt

# 1) a deliberately broken example (10 blockers)
python -m backtest_lint --bundle examples/bad

# 2) a clean example (all pass)
python -m backtest_lint --bundle examples/good

# 3) "best of a 24-config grid": E1–E9 pass, E12 blocks
python -m backtest_lint --bundle examples/multi_trial

# 4) generate a blank template
python -m backtest_lint --init my_bundle
```

Output is a Markdown report card; `--md` / `--json` writes it to disk. Exit code is 1 when anything FAILs, so it drops into CI.

```bash
python -m backtest_lint --bundle my_bundle --md report.md --json report.json
```

> Live on PyPI: <https://pypi.org/project/backtest-lint/>. The `backtest-lint` command is equivalent to `python -m backtest_lint`.
> **Version note**: PyPI currently ships **0.3.0**; this repository is **0.4.1**. E13 (size-variable basis)
> and several E2/E7/E9/E10/E11 fixes are not published yet, so `pip install backtest-lint` does not include them.
> Windows cmd / PowerShell works: the ✅/⚠️/❌ symbols degrade gracefully to the console encoding.

---

## Watch it dismantle a fake excess (`demo/`)

A **synthetic market** with true alpha set to **zero**, then the reporting conventions are added back one at a time:

```bash
python demo/run_demo.py
```

```
step                                                       annualized excess
① as-is: survivor pool / best of 18 configs / no cost / zero-cost daily benchmark    +4.83%
② fix multiple testing: median of 18 configs, not the best one                       +3.56%
③ fix survivorship: universe includes delisted stocks                                +1.01%
④ fix costs: charge round-trip cost on every rebalance                                −3.40%
⑤ fix benchmark: daily rebalanced → buy-and-hold                                     −3.98%
⑥ fix annualization: geometric on both sides                                         −3.95%
```

**The whole 8.78pp swing is bookkeeping**, because true alpha is zero. Fixed seed, fully reproducible —
you can verify it without any real data. See [`demo/README.md`](demo/README.md) (Chinese).

---

## Input format

One directory, two files.

### `daily.csv`

```csv
date,strategy,benchmark
2024-01-02,0.0012,0.0008
2024-01-03,-0.0005,0.0001
```

- `date` optional; `strategy` required; a missing `benchmark` is an immediate FAIL.
- Chinese column names accepted: `日期` / `策略` / `基准`.
- Returns are **daily simple returns** (0.0012 = 0.12%), not a NAV series.

### `meta.json`

```json
{
  "name": "my strategy",
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
  "reported_bench_ann": 0.05,
  "signal_uses": "close_t",
  "trade_price": "open_t+1"
}
```

| Field | Values | Meaning |
|---|---|---|
| `universe_mode` | `pit` / `static` / `unknown` | are constituents point-in-time? |
| `delisted_included` | `true` / `false` | does the universe include delisted names? |
| `benchmark_type` | `buy_hold` / `index` / `ew_daily` / `strategy_like` | what the benchmark is |
| `cost.applied_to_benchmark` | `true` / `false` | is the benchmark charged the same costs? |
| `phases_run` | int | how many rebalance phases were run |
| `panel_fixed` | `true` / `false` | is the panel start / lookback fixed? |
| `panel_trail_history` | array | windows used in one comparison; more than one is a FAIL |
| `n_trials` | int | parameter grid / variant count; if `trials.csv` has more columns, the column count is used (E7/E10) |
| `sample_days` | int | sample length claimed in the report; mismatch with the file is a FAIL |
| `reported_t` | float | headline t-statistic |
| `reported_ann` | float | headline strategy annualized return |
| `reported_bench_ann` | float | headline benchmark annualized return |
| `signal_uses` | `close_t` / `high_t` / `vwap_t` … | which price the signal reads (E9) |
| `trade_price` | `open_t` / `close_t` / `open_t+1` … | which price the fill uses (E9) |
| `size_variable` | `market_cap` / `total_shares * close` … | variable used for size neutralisation; `close` is a FAIL (E13) |
| `size_neutralized` | `true` / `false` | whether size neutralisation was done; `true` without `size_variable` is a FAIL (E13) |
| `trials_csv` | filename | path of the candidate matrix, defaults to `trials.csv` (E11/E12) |

### `trials.csv` (optional, enables E11/E12)

**One column per configuration you actually ran, one row per period**, at the same frequency
as `daily.csv`. Without this file E11/E12 report `INFO skipped` — a single backtest genuinely
cannot estimate overfitting probability, so that is not a failure.

```csv
cfg00,cfg01,cfg02
0.0012,-0.0043,0.0008
-0.0031,0.0022,-0.0011
```

> Filling in only the config you reported defeats the purpose: E11/E12 exist to look at
> **the ones you did not report**. Keep every variant you ran — parameter grid, lookback
> windows, industry-neutral on/off …

---

## Checks

| # | Check | PASS | WARN | FAIL |
|---|---|---|---|---|
| B0 | Benchmark present | benchmark column present | — | missing benchmark |
| E1 | Benchmark bookkeeping | reported value matches geometric (<1pp) | geometric vs. arithmetic gap >1pp | reported value closer to arithmetic, gap >0.5pp |
| E2 | Cost symmetry | benchmark costed too, or is buy-and-hold / index | benchmark type undeclared | `ew_daily` / `strategy_like` benchmark at zero cost, or cost status undeclared |
| E3 | Universe point-in-time | `pit` | — | `static` or undeclared |
| E4 | Rebalance phase | ≥5 phases | 2–4 | 1 or undeclared |
| E5 | Fixed panel | `panel_fixed=true`, one window | undeclared | `false`, or several windows |
| E6 | Annualization basis | reported matches geometric | no reported value | gap >0.5pp, or sample length mismatch |
| E7 | Multiple testing | \|t\| > Bonferroni threshold | no t reported | \|t\| below threshold |
| E8 | Survivorship | delisted names included | undeclared | current constituents only |
| **E9** | **Execution timing** | fill after the signal is confirmed (e.g. `close_t` → `open_t+1`) | `signal_uses` / `trade_price` undeclared | signal from the close, fill at a price in the same session |
| **E10** | **Search-size discount (DSR)** | DSR ≥ 0.95 | 0.90–0.95 | DSR < 0.90 |
| **E11** | **Overfitting probability (PBO)** | PBO ≤ 0.20 | 0.20–0.50 | > 0.50 |
| **E12** | **Out-of-sample decay (CPCV)** | ≤10% of splits lose money out of sample | 10–30% | >30% |
| **E13** | **Size-variable basis** | explicitly market cap (`market_cap` / `total_shares * close`) | `size_variable` undeclared or unrecognised | price (`close` / `adj_close`) or bare share count used as size |
| S1 | Return sanity | clean | >5% missing or a >50% day | <20 valid observations or zero variance |
| S2 | Alignment | ≥95% of dates valid on both | — | <95% |
| S3 | Benchmark sanity | clean | a >15% day | a >30% day or <20 valid observations |

Bonferroni threshold: `z = Φ⁻¹(1 − 0.025/n_trials)` — about **2.99** at n=18, **1.96** at n=1.

E11/E12 need `trials.csv`; without it they return `INFO` (skipped), which counts as neither pass nor fail.
E10 needs only `daily.csv` and `n_trials` — so **it always runs**, and only means something if `n_trials` is honest.

---

## A real case: the V2.0 reversal strategy

`examples/v2_reversal/` is a real run from the project (2020-12 → 2026-10, point-in-time universe, 4 phases).
Verdict: **5 blockers, 5 warnings**.

| Check | Result | Detail |
|---|---|---|
| E2 | ❌ | benchmark is daily-rebalanced equal weight (100% daily turnover) at zero cost |
| E6 | ❌ | report cites **1545** days, the return file has **1425** — not the same sample |
| E7 | ❌ | 18-config grid, threshold \|t\|>2.99, observed t=0.80 |
| E8 | ❌ | current constituents only; delisted names missing |
| E10 | ❌ | at 18 trials SR0=0.0491/period, observed SR=−0.0101/period, DSR=0.013 |
| E1 | ⚠️ | benchmark geometric −5.95% vs arithmetic −2.55%, 3.40pp apart, basis undeclared |
| E4 | ⚠️ | only 4 phases |
| E9 | ⚠️ | `signal_uses` / `trade_price` undeclared, execution timing unknown |
| E13 | ⚠️ | `size_variable` undeclared; cannot tell whether size neutralisation used market cap or price |
| S3 | ⚠️ | one benchmark day at **+18.70%** — confirm it is real volatility, not an alignment bug |

Full report: `examples/v2_reversal_report.md`.

---

## Layout

```
backtest-lint/
├─ .github/workflows/  CI: Python 3.10–3.13, 86 tests + Windows encoding job
├─ backtest_lint/
│  ├─ checks.py      ten error classes (E1–E9, E13) + search size (E10–E12) + four sanity checks
│  ├─ overfit.py     DSR / MTRL / PBO / CPCV (numpy only, scipy optional)
│  ├─ metrics.py     geometric/arithmetic annualization, NW, Bonferroni
│  ├─ loader.py      daily.csv + meta.json (+ trials.csv) loading and template
│  ├─ report.py      Markdown / JSON report
│  └─ cli.py         command-line entry point
├─ demo/             reproducible "fake excess" demo (synthetic, true alpha = 0)
├─ examples/
│  ├─ bad/           deliberately broken (10 blockers)
│  ├─ good/          clean (all pass)
│  ├─ multi_trial/   best of a 24-config grid (E12 blocks)
│  └─ v2_reversal/   real project output (5 blockers, 5 warnings)
├─ tests/            86 unit tests
├─ requirements.txt
└─ README.md
```

---

## Limitations (state these when you report results)

1. **Only observable conventions are checked.** Look-ahead in features or execution, and outright code bugs
   (e.g. misaligned adjustment factors), need code review or data reconciliation — this tool cannot see them.
2. **It does not predict returns.** Passing the audit is not evidence of alpha; it only rules out "looks profitable because of bookkeeping".
3. **`meta.json` is self-reported.** If `universe_mode` or `phases_run` is filled in dishonestly, the checks are void.
   What the tool can do is make "must declare" a hard gate.
4. **E1/E6 need reported values.** Without `reported_ann` / `reported_bench_ann` they can only WARN.
5. **Survivorship can only be disclosed, not fixed.** If the local dataset has no delisted names, it has none.
6. **E10–E12 only see the candidate set you declare.** Leave trials out of `trials.csv` and PBO
   drops — the check audits *the number of attempts you admit to*, not the number you actually ran.
7. **The DSR bar uses a normal approximation.** With fat tails, autocorrelation or highly
   correlated trials the true bar is higher, so the DSR reported here is optimistic.
   E11/E12 likewise assume the candidate configs are comparable.

---

## Who it is for

- Anyone about to publish a backtest number: run it first and clear the blockers.
- Anyone reading someone else's backtest: drop their daily returns and parameters in and see what fails.
- Teaching: each of the thirteen error classes maps to a runnable check instead of a slide.

> The tool's only claim to credibility is a project that took its own **+10.42%** apart down to **−1.76%**.
> It is not a strategy generator. It is a strategy falsifier.
