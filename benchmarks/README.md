# Benchmarks

All numbers in this repository come from llama-benchy 0.4.0 through
`sparkrun benchmark perf`, against the live instance (`--skip-run --no-stop`).
The profile is `ab-4cell.yaml`:

```yaml
framework: llama-benchy
args:
  depth: [0, 8192]
  pp: [2048]
  tg: [128]
  concurrency: [1, 10]
  prefix_caching: true
```

Four decode cells (depth 0/8192 × concurrency 1/10), each generating 128 tokens
after a 2048-token prompt, plus the matching prefill measurements.

## Method

- Every pass is 3 runs per cell. Where two passes are combined, n=6.
- Numbers in the README and `summary.csv` are **medians** of the per-run values
  in the JSON, not the CSV’s aggregate mean. The d8k c1 cell has a single-outlier
  problem; medians are the honest statistic.
- The first pass after a boot is discarded. Wait at least 10 minutes after
  `/health`.
- Do not reuse a bench state directory across boots. `sparkrun benchmark perf`
  keys its state on cluster + profile + args, not the recipe file; without
  `--fresh` it can re-emit old numbers in ~2 seconds and send zero requests.
- No other inference traffic during the run. The bench measures the same GPUs
  that serve production.
- Host state was checked around the reported passes: GPU 60–75 °C, SM clock
  2.4–2.5 GHz, no throttle, swap-in near zero.

Command:

```bash
sparkrun benchmark perf recipe/production.yaml \
  --skip-run --no-stop --fresh \
  --profile benchmarks/ab-4cell.yaml \
  --output benchmarks/ab-$(date +%Y%m%d-%H%M).yaml
```

## Result table

| Lane | d0 c1 | d0 c10 | d8k c1 | d8k c10 | pp c1 | pp c10 | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| `source_exp_1017` | 44.9 | 110.7 | 52.6 | 84.4 | 3337 | 4787 | 3 |
| `baseline_orig_checkpoint` | 65.7 | 120.5 | 59.9 | 91.5 | 3742 | 5016 | 3 |
| `knapcio_v24_cert_first` | 64.2 | 106.8 | 61.8 | 85.4 | 3434 | 4097 | 3 |
| `knapcio_v24_cert_warm` | 70.8 | 114.2 | 55.2 | 85.8 | 3773 | 4440 | 6 |
| `knapcio_v24_nocert` | 64.6 | 114.9 | 69.9 | 85.1 | 3692 | 4950 | 6 |
| `knapcio_bridge` | 64.9 | 118.0 | 71.1 | 88.0 | 3896 | 5169 | 6 |
| `fastload_nocert` | 63.7 | 120.4 | 60.2 | 91.1 | 3956 | 5267 | 6 |
| `production_overlay` | 59.9 | 114.6 | 69.4 | 92.0 | 3929 | 5042 | 3 |

## Reading the table

- **CERT off is the clean experiment.** `knapcio_v24_cert_warm` and
  `knapcio_v24_nocert` use the same image, weights, and recipe; only
  `DSV41_CERT_HEAD` changed, which requires a restart, so the two sides are
  different boots. d8k c1 55.2 → 69.9 (+26.6%, p=0.020); the other three cells
  moved <1%.
- **The source lane is one pass.** `source_exp_1017` is a single n=3 pass of the
  published `@experimental` recipe; it was not repeated warm. The +33%/+32%
  deltas in the README are measured against that pass and are indicative until a
  warm A/B is run.
- **The bridge lane is the closest warm upstream comparison.** It is at parity
  with production at long context and ~8% ahead at d0 c1. Production is not a
  universal win over every upstream shape.
- **The original checkpoint lane is not the 1017 checkpoint.** Its `model` field
  is `deepseek-ai/DeepSeek-V4.1-Flash` (revision `dba1be0a`), not `-1017`
  (`1bae0e10`). Do not use it as a 1017 comparison.

## Files

```text
ab-4cell.yaml             profile definition
summary.csv               medians per lane, computed from the JSON
csv/ab-*.csv              llama-benchy aggregate CSV per pass
raw/ab-*.json             per-run values (tg_throughput.values etc.)
```

`summary.csv` was generated from the `raw/` JSON: for each benchmark entry with
`response_size=128`, `prompt_size=2048`, and `is_context_prefill_phase=false`,
take `tg_throughput.values`; for prefill take `pp_throughput.values` from the
depth-0 entries. Group the passes per lane, concatenate the runs, and take the
median.

The JSON files contain no hostnames, IPs, or credentials. They do contain the
model label and raw timings.

## Caveats

- n=3 per pass is small. Treat single-cell ratios under ~15% as noise unless
  they are replicated (as the CERT result is, at n=6 with a permutation test).
- `d8k c1` is the noisiest cell. Across the day it printed 54.8, 55.6, 59.9,
  61.8, 69.9, 70.8, 71.5, and 72.0 across lanes. A single run can move the mean
  by 10%.
- `pp2048 d0 c1` is bimodal (3.17–3.34k vs 3.74–4.00k). Prefill comparisons
  need the same mode, which is hard to control; decode comparisons are more
  stable.
- The benchmark process itself runs on the head node and shares its CPU with the
  tokenizer/scheduler. That is true for every lane here, so the relative
  comparison is fair, but the absolute numbers include that overhead.
