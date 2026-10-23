# Tuning log

Chronological record of the 2026-10-19 tuning window. Times are local (BST);
serve-log timestamps are UTC. All decode numbers are tok/s, median per cell,
llama-benchy 0.4.0, 3 runs/cell per pass unless noted.

The starting point was a September knapcio CLI lane that had been running on the
cluster. The immediate source for this work was the Little Cedar
`@experimental` 1017 recipe and the knapcio v2.4 `canary-roce` image.

## Reference points

| Lane | d0 c1 | d0 c10 | d8k c1 | d8k c10 | pp c1 | pp c10 |
|---|---:|---:|---:|---:|---:|---:|
| September knapcio CLI reference | 72.0 | 119.6 | 62.1 | 88.2 | 3947 | 5016 |
| Little Cedar `@experimental` 1017 | 44.9 | 110.7 | 52.6 | 84.4 | 3337 | 4787 |

The September lane is a different launch shape and CLI; it is a sanity
reference, not an A/B control. The `@experimental` lane is the recipe this work
replaced.

## 2026-10-19 sequence

### 1. Baseline: original checkpoint + v2.4 image

`knapcio-baseline-diag.yaml` (original checkpoint `dba1be0a`, v2.4 image,
diagnostic recipe):

```text
d0 c1 65.7   d0 c10 120.5   d8k c1 59.9   d8k c10 91.5   pp 3742 / 5016
```

This established that the v2.4 image alone, with the old checkpoint and the
diagnostic launch shape, was healthy but did not recover the September d0 c1
number.

### 2. First v2.4 lane with the 1017 checkpoint

`deepseek-v41-knapcio-v24-native-tp4.yaml` (1017, v2.4 image, CERT on, stock
loader). Cold pass:

```text
d0 c1 64.2   d0 c10 106.8   d8k c1 61.8   d8k c10 85.4   pp 3434 / 4097
```

Two warm passes on the same boot:

```text
warm 1: 68.9 / 98.8 / 55.6 / 86.5
warm 2: 72.7 / 117.8 / 54.8 / 85.1
```

The warm pair showed two things: the first pass after boot is not a measurement,
and the long-context single-stream cell was consistently ~55, well below the
September 62 and the bridge 71.

### 3. Bridge experiment: isolate the missing CLI flags

The “bridge” lane used the diagnostic recipe shape + knapcio CLI command + 1017
weights + the v2.4 lane’s revision-scoped Engram pack. Same image, same weights,
same pack as the v2.4 lane; it differed in server args:

```text
--enable-deepseek-v4-fp4-indexer   bridge yes | v24 NO
--min-free-slots-delay 1           bridge yes | v24 NO
--sleep-on-idle / --enable-cache-report  bridge yes | v24 NO
--watchdog-timeout                 bridge 1800 | v24 300
--max-total-tokens                 bridge 8.0M | v24 9.4M -> capped 8.86M
DSV41_CERT_HEAD                    bridge unset | v24 1
DSV41_FAST_LOAD                    bridge 1 | v24 0
```

Only two touch the decode path: the fp4 indexer and the CERT head. The bridge
measured:

```text
cold: 63.2 / 116.6 / 71.5 / 89.2
warm: 66.7 / 119.5 / 70.8 / 86.8
```

The d8k c1 gap was therefore not the indexer (bridge had it enabled and was
fast), and not the loader. The remaining suspect was the CERT head.

### 4. CERT off

Same 1017 weights, same image, same recipe; `DSV41_CERT_HEAD=0` only. Changing
it requires a restart, so the two sides are different boots. Two passes, six
runs per cell:

| Cell | CERT on | CERT off | Delta | permutation p |
|---|---:|---:|---:|---:|
| d0 c1 | 70.8 | 64.6 | −8.7% | 0.87 (noise) |
| d0 c10 | 114.2 | 114.9 | +0.6% | 0.47 |
| d8k c1 | 55.2 | **69.9** | **+26.6%** | **0.020** |
| d8k c10 | 85.8 | 85.1 | −0.8% | 0.57 |

This closed the long-context single-stream hole. The d0 c1 difference is inside
the spread and has no mechanism; the d8k c1 result is outside it and holds
under a permutation test.

Not contaminated: swap-in ~0 from 18:05 (4.4 GB stays swapped but cold), GPU
60–75 °C, SM clock 2.4–2.5 GHz, no throttle. Engram hit rate was still climbing
in pass 1 (92.5% at 18:05 → 98.5% at 18:11), yet pass 1 scored the better d8k c1
(72.0 vs 67.8), so the gain is not cache warming.

### 5. Fast loader

The prepare hook originally rejected `DSV41_FAST_LOAD=1` as a policy choice.
The same adapter set had already booted with the fast loader in
`knapcio-baseline-diag`, so the assert was made an opt-in
(`DSV41_PREPARE_ALLOW_FAST_LOAD=1`, see `mods/`).

Measured:

```text
boot-to-ready        482 s -> 148 s
target load          305.99 -> 66.76 s
draft load            66.48 -> 4.30 s
KV pool at 0.80     9,016,832 -> 6,804,224  (-24.5%)
```

Decode effect, two warm passes vs the stock-loader no-CERT passes:

| Cell | Stock | Fast | Delta | p (two-sided) |
|---|---:|---:|---:|---:|
| d0 c1 | 64.6 | 63.7 | −1.4% | 0.86 |
| d0 c10 | 114.9 | 120.4 | +4.9% | 0.27 |
| d8k c1 | 69.9 | 60.2 | −13.9% | 0.061 |
| d8k c10 | 85.1 | 91.1 | +7.1% | 0.095 |

Prefill reads +6–7% and TTFT 5–10% lower, but that is not a loader effect:
`pp2048 d0 c1` is bimodal run-to-run across the day (3.17–3.34k vs 3.74–4.00k),
and the fast-loader passes landed in the high mode twice. The d8k c1 −14% is
the noisiest cell in the profile (values across the day: 54.8, 55.6, 59.9, 61.8,
69.9, 70.8, 71.5, 72.0) and is not separable at n=6. No defensible decode gain,
no defensible regression.

### 6. Production overlay

`recipe/overlay.yaml` — the Little Cedar `@experimental` 1017 recipe, v2.4
image, CERT off, TP_PAD 0, stock loader, three base-recipe env settings.
Its first boot was run with the **fast loader engaged** (to validate that the
overlay did not silently fall back to mmap):

```text
load_weight=74.73 s  kv_cache_allocation=0.66
scheduler_e2e=145.36  tokenizer_e2e=150.71
graphs target_verify=15.37 + draft=2.26
max_total_num_tokens=6,193,408  (fast loader, mem_fraction 0.80)
```

We then set `DSV41_FAST_LOAD=0` to restore the KV pool; the live instance
reports **9,120,256** tokens. One pass, n=3:

```text
d0 c1 59.9   d0 c10 114.6   d8k c1 69.4   d8k c10 92.0   pp 3929 / 5042
```

This reproduces the fast-loader lane inside session noise on all four cells and
is clearly ahead of the `@experimental` source lane, though not ahead of the
warm bridge on d0 c1.

## All passes

See `benchmarks/summary.csv` for the machine-readable table. The lanes are:

| Lane key | Passes | Notes |
|---|---|---|
| `source_exp_1017` | `ab-current` | published `@experimental` recipe, one cold pass |
| `baseline_orig_checkpoint` | `ab-baseline` | original checkpoint `dba1be0a`, v2.4 image |
| `knapcio_v24_cert_first` | `ab-v24` | first v2.4 lane, CERT on, cold |
| `knapcio_v24_cert_warm` | `warm1` + `warm2` | CERT on, two warm passes |
| `knapcio_v24_nocert` | `nocert1` + `nocert2` | CERT off, two passes |
| `knapcio_bridge` | `bridge` + `bridge-warm` | diagnostic recipe + knapcio CLI |
| `fastload_nocert` | `fastload1` + `fastload2` | fast loader, CERT off |
| `production_overlay` | `ab-prod` | this repo’s recipe, stock loader |

## Pitfalls worth carrying forward

- **`sparkrun benchmark perf` ignores the recipe file in its bench ID.** Two
  different recipes on the same cluster and profile share a `bench_<hash>`
  state directory. Without `--fresh`, a run can print
  `Prior benchmark state ... is COMPLETE — re-emitting its recorded results`
  and exit in ~2 s with zero requests.
- **The bench output directory must exist.** The `--output` parent is not
  created for you.
- **Wait ~10 minutes after `/health`.** A fresh boot is still paging in weights
  and filling kernel caches.
- **Quote medians, not means, for d8k c1.** One 98.7 outlier moved the mean from
  70.5 to 79.2 in one pass.
- **`sparkrun_serve.log` is UTC, the host is BST.** A line that looks 45 minutes
  old is usually just the offset.
- **SGLang does not validate `served_model_name`.** `/v1/models` will echo any
  label you give it; verify the checkpoint through the container mounts or
  `model_path`, not the served name.
- **First pass after boot is not a measurement.** Two identical passes five
  minutes apart differed by 19% at d0 c10.
