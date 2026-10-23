# DeepSeek-V4.1-Flash-1017 on 4× DGX Spark

Production notes, exact configuration, and A/B measurements for a four-node
NVIDIA DGX Spark (GB10) serving lane running **DeepSeek-V4.1-Flash-1017** with
SGLang, sparkrun-native TP4, 1M context, and a 9.12M-token KV pool.

This repository is written so that **Little Cedar Group** can compare this lane
with its own build and reuse anything useful. It is an independent production
note, not an official Little Cedar Group or knapcio repository.

The short version: the kernels and most of the stack are upstream work. What is
different here is the sparkrun-native integration, two configuration changes
that closed a large long-context decode gap, and the measurements showing which
changes mattered and which did not.

The checkpoint is **`deepseek-ai/DeepSeek-V4.1-Flash-1017`**, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, published 2026-10-17. All
measurements in this repository were taken during the 2026-10-19 tuning window.

## Lane at a glance

| Item | Value |
|---|---|
| Hardware | 4× NVIDIA DGX Spark (GB10), 1 GPU and 128 GB unified memory each |
| OS / kernel | Ubuntu 24.04.5 LTS (DGX OS 7.x), `7.0.0-1019-nvidia` |
| Driver | NVIDIA 580.178.04 |
| Runtime | sparkrun 0.4.1, Docker 29.6.2 |
| Fabric | 2× 100 Gb/s RoCE per node, dual rail, MikroTik CRS504 |
| Model | DeepSeek-V4.1-Flash-1017, HF revision `1bae0e10…` |
| Parallelism | TP4 / EP1 / PP1 / DP1 (one GPU per node) |
| Context | 1,048,576 tokens |
| KV cache | fp8_e4m3, **9,120,256 tokens** (`max_total_num_tokens`) |
| Speculative decode | DSpark k=5 |
| Image | `dsv41-knapcio-v24:canary-roce-local`, `sha256:6cc83f46…` |
| Weights | ~71.6 GB per rank, ~286 GB total resident |
| Boot to ready | ~482 s stock loader; ~148 s fast loader (see trade below) |
| Endpoint | OpenAI-compatible on port 8000 |

Live `/get_server_info` at the time of writing:

```text
tp_size=4 ep_size=1 dp_size=1 pp_size=1
context_length=1048576
max_total_num_tokens=9120256
mem_fraction_static=0.8
chunked_prefill_size=4096 max_prefill_tokens=16384
max_running_requests=16 cuda_graph_max_bs_decode=16
attention_backend=dsv4
moe_runner_backend=flashinfer_mxfp4
fp8_gemm_runner_backend=flashinfer_cutlass
speculative_algorithm=DSPARK speculative_dspark_block_size=5
kv_cache_dtype=fp8_e4m3
enable_deepseek_v4_fp4_indexer=False
```

## Measured results

All numbers are **decode tok/s, median of 3 runs per cell** (or 6 when two
passes are combined), from llama-benchy 0.4.0 through
`sparkrun benchmark perf`. Profile: prompt 2048, generation 128, depth 0 or
8192, concurrency 1 or 10, prefix caching on. “pp” columns are prefill tok/s at
depth 0.

| Lane | d0 c1 | d0 c10 | d8k c1 | d8k c10 | pp c1 | pp c10 | runs/cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| Little Cedar `@experimental` 1017 (source recipe) | 44.9 | 110.7 | 52.6 | 84.4 | 3337 | 4787 | 3 |
| knapcio v2.4 + CERT head | 70.8 | 114.2 | 55.2 | 85.8 | 3773 | 4440 | 6 |
| knapcio v2.4 + CERT **off** | 64.6 | 114.9 | **69.9** | 85.1 | 3692 | 4950 | 6 |
| **Production overlay** (this repo) | 59.9 | 114.6 | 69.4 | **92.0** | 3929 | 5042 | 3 |
| knapcio “bridge” lane (for context) | 64.9 | 118.0 | 71.1 | 88.0 | 3896 | 5169 | 6 |
| Fast-loader variant (CERT off) | 63.7 | 120.4 | 60.2 | 91.1 | 3956 | 5267 | 6 |
| Original checkpoint + v2.4 image | 65.7 | 120.5 | 59.9 | 91.5 | 3742 | 5016 | 3 |

Difference between the production overlay and the published source recipe it
replaces:

| Cell | Source | Production | Delta |
|---|---:|---:|---:|
| d0 c1 | 44.9 | 59.9 | **+33.4%** |
| d0 c10 | 110.7 | 114.6 | +3.5% |
| d8k c1 | 52.6 | 69.4 | **+31.9%** |
| d8k c10 | 84.4 | 92.0 | +9.0% |
| prefill c1 | 3337 | 3929 | +17.7% |
| prefill c10 | 4787 | 5042 | +5.3% |

**Read the source-lane comparison as indicative, not a controlled A/B.** The
source recipe was measured once (n=3) and was not repeated warm. The clean,
single-variable result in this repo is the CERT-head experiment: same image,
same weights, same recipe, six runs per side; only `DSV41_CERT_HEAD` changed (a
restart is required, so the two sides are different boots). That moved d8k c1
from 55.2 to 69.9 (**+26.6%**, permutation p=0.020 one-sided) while the other
three cells moved less than 1%.

The production overlay is *not* faster than every upstream lane. Against the
warm knapcio “bridge” lane it is at parity at long context (69.4 vs 71.1 at
8k c1; 92.0 vs 88.0 at 8k c10) and ~8% lower at d0 c1. The gains documented
here are relative to the source `@experimental` recipe and to the CERT-on
variant, not a claim of a new decode kernel record.

## What actually changed

### 1. `DSV41_TP_PAD=0` unlocks the replicated split

The base recipe sets `DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv`, but the adapter
defaults `DSV41_TP_PAD` to `1`. With padding on, the column split cannot run:
every layer logs

```text
[replicated_split] ... slice failed on rank 0: AssertionError()
[replicated_split] ... split OFF
```

for all 42 layers. The `wqkv_a` 1792×5120 and `engram.wkv` 25600×6144 column
splits, and the compact RoCE gather, never happen. Setting `DSV41_TP_PAD=0`
produces:

```text
[replicated_split] model.layers.0.self_attn.wqkv_a 1792x5120:
  split ON, window GEMMs (no pad) (rank 0 window True, slice True)
[replicated_split] model.layers.0.self_attn.wqkv_a:
  compact gather (roce) bit-identical, ON (rank 0 equal True)
[replicated_split] model.layers.1.engram.wkv 25600x6144:
  split ON, window GEMMs (no pad)
```

This was first observed on the Little Cedar `canary-roce` image, where all 42
layers failed the split with the default. The knapcio v2.4 lane pins `0`.

### 2. `DSV41_CERT_HEAD=0` removes the long-context single-stream hole

The certified verify head is a correctness feature, and it costs throughput at
batch size 1. Same image, same weights, same recipe, six runs per side; only
`DSV41_CERT_HEAD` changed:

| Cell | CERT on | CERT off | Delta | permutation p |
|---|---:|---:|---:|---:|
| d0 c1 | 70.8 | 64.6 | −8.7% | 0.87 (noise) |
| d0 c10 | 114.2 | 114.9 | +0.6% | 0.47 |
| d8k c1 | 55.2 | **69.9** | **+26.6%** | **0.020** |
| d8k c10 | 85.8 | 85.1 | −0.8% | 0.57 |

The shape fits a per-decode-step cost: a fixed verify-head screen hurts single
stream at long context, but disappears at batch 10. The d0 c1 −8.7% has no
mechanism (CERT cannot speed up short-context decode), is inside the observed
spread (CERT 60.2–76.7, no-CERT 56.8–74.2), and p=0.87.

**Quality caveat:** this is a throughput measurement, not a quality one. CERT is
a certified-verify feature. We did not capture a matched acceptance-rate
comparison across boots, so if the guarantee matters to you, keep the head and
pay the ~15 tok/s at long-context single stream, or re-measure quality first.

### 3. The image carries the v2.4 adapters; the YAML alone does nothing

The recipe is a thin overlay over the Little Cedar `@experimental` 1017 recipe.
The added environment names are read by adapters compiled into
`dsv41-knapcio-v24`, built from knapcio’s `Dockerfile.canary-roce`:

- SGLang dsv4.1 tree `f80c91a4b`
- RoCEnante patch + TP4 adaptation
- b12x and b12x_next (armed build `a7d7d29b`)
- v2.4 adapter set, including `replicated_split`, `moe_b12x_next`,
  `cert_head`, `fast_load`, and the engram row store

On the Little Cedar image those variables are inert (no v2.4 adapters, and its
prepare step rejects `FAST_LOAD=1`). The image is the fork.

Fifteen environment variables were added relative to the base recipe:
`DSV41_TP_PAD=0`, `DSV41_CERT_HEAD=0` (+ the `M` list),
`DSV41_MXFP8_BACKEND=b12x`, the five `DSV41_FAST_LOAD_*` knobs +
`DSV41_PREPARE_ALLOW_FAST_LOAD=1`, `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192`,
`DSV41_STATS_SECONDS=60`, `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`,
`SPARK_PREFILL_TP_MIN_CONTEXT=32768`, and `SPARK_PREFILL_TP_MIN_ROWS=1024`.
One default was added: `fp8_gemm_backend: flashinfer_cutlass`.

### 4. The fast loader is a boot-time trade, not a decode gain

`DSV41_FAST_LOAD=1` (with the prepare hook’s opt-in) cuts boot-to-ready from
482 s to 148 s:

| | Stock loader | Fast loader |
|---|---:|---:|
| Target weight load | 305.99 s | 66.76 s |
| Draft (DSpark) load | 66.48 s | 4.30 s |
| Boot to ready | **482 s** | **148 s** |
| KV pool at `mem_fraction_static=0.80` | 9,016,832 | 6,804,224 |
| Pool change | — | **−24.5%** |

The loader reads the checkpoint eagerly (155.2 GB over 45 shards instead of the
stock path’s 211 GB cgroup read for 71.75 GB resident), which raises resident
weight memory by ~4.9 GB and shrinks the KV pool. It did **not** change decode
throughput beyond session noise.

The production lane currently runs `DSV41_FAST_LOAD=0` to keep the full
**9.12M-token** pool. If boot time matters more than context capacity, set
`DSV41_FAST_LOAD=1` and raise `gpu_memory_utilization` to ~0.83 for pool parity
(+0.01 is worth ~1.22 GB ≈ 0.73M tokens on this box).

## What did not help

- **The fp4 indexer was not the missing piece.** The v24 lane runs
  `enable_deepseek_v4_fp4_indexer=False`; the no-CERT lane (no indexer) measured
  at bridge parity (indexer enabled) on all four decode cells, −1 to −4%.
- **Engram warmth is not a decode story.** Warming the L1 row-store cache cut
  NVMe reads ~5× (65.9k/29.5k → 2.4k–32.2k reads/min, hit 94–99%) with no
  measurable decode change: d8k c1 medians 71.5 → 70.8. Cache the pack for I/O
  and TTFT stability, not tok/s.
- **`max_total_tokens` is a hint, not a dial.** Requesting 9.4M is capped to the
  profiled 9,120,256. Raise `mem_fraction_static` instead, inside the proven
  limit (0.86 works at full 1M context; 0.88 early-OOMs; 0.90 hard-fails).
- **The first pass after a boot is not a measurement.** Two identical passes on
  one boot, five minutes apart, scored 98.8 vs 117.8 tok/s at d0 c10. The first
  pass overlaps weight page-in and first-touch kernel autotune. Discard it.

## Reproduction

### 0. Prerequisites

- Four DGX Spark nodes, one GPU each, 128 GB unified memory.
- 2× 100 Gb/s RoCE per node, lossless PFC configured. Our fabric notes:
  <https://github.com/MashieSaldana/DGX-Spark-Networking>.
- sparkrun ≥ 0.4.0 (we run 0.4.1), Docker on every node.
- Run the container as root through the cluster’s `executor_config`:
  `{user: root}`. This also avoids a non-root SemLock spawn race when both RoCE
  interfaces are up.
- NVMe headroom: ~476 GB for the checkpoint plus ~47 GiB/node for the Engram
  pack (two ~25 GiB files), on every node.

### 1. Model

The checkpoint is public and MIT-licensed:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170
```

sparkrun distributes the model from the head to the workers over the detected
IB IP. If you pre-seed the HF cache on each node, keep the same revision and
layout (`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017`).

### 2. Build the image

The image is built unmodified from knapcio’s `Dockerfile.canary-roce`. Pin the
tree; the image tag is local.

```bash
git clone https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4.git
cd DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4
git checkout 403a5aa2de356bedadeb55a1413824daa7883ef7

docker build -f Dockerfile.canary-roce \
  --build-arg BASE_IMAGE=lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860 \
  -t dsv41-knapcio-v24:canary-roce-local .
```

The Dockerfile replaces `/sgl-workspace/sglang/python` with the dsv4.1 tree at
`f80c91a4b`, applies the RoCEnante patch and TP4 adaptation, installs
`sglang-kernel 0.4.7` / `sgl-deep-gemm 0.2.0`, copies b12x and b12x_next,
builds `librow_store.so`, and runs the CPU-only adapter test suite as a build
gate. The resulting image is 33.5 GB.

Our image ID, identical on all four nodes:

```text
sha256:6cc83f463773a6a0f28e9220f47f564214f2d8a49d8954ea51402843142a9e52
```

Distribute it by local registry, or:

```bash
docker save dsv41-knapcio-v24:canary-roce-local | gzip -1 | \
  ssh <node> 'gunzip | docker load'
```

### 3. Prepare mod

The recipe uses the `dsv41-native-prepare` mod from the spark-arena
`experimental` registry. The production lane runs `DSV41_FAST_LOAD=0`, so the
stock mod works unmodified.

The patch in `mods/prepare-allow-fast-load.patch` is only needed if you want to
run the fast-loader variant (`DSV41_FAST_LOAD=1`). It changes one policy assert
in `prepare.py` into an opt-in:

```diff
-        env.get("DSV41_FAST_LOAD", "0") == "0",
-        "this experimental baseline requires DSV41_FAST_LOAD=0",
+        env.get("DSV41_FAST_LOAD", "0") == "0"
+        or env.get("DSV41_PREPARE_ALLOW_FAST_LOAD", "0") == "1",
+        "this experimental baseline requires DSV41_FAST_LOAD=0 "
+        "(set DSV41_PREPARE_ALLOW_FAST_LOAD=1 to opt this lane into the fast loader)",
```

Apply it next to the recipe (the mod resolves from the recipe’s own directory
before the registry copy):

```bash
cd <dir-containing>/dsv41-native-prepare
patch -p1 < /path/to/prepare-allow-fast-load.patch
```

### 4. Recipe

`recipe/production.yaml` is the flattened, self-contained recipe (131 lines).
`recipe/overlay.yaml` is the readable overlay it came from: a ~15-env delta over
`@experimental/deepseek-v4.1-flash-1017-b12x-dspark-sglang-tp4`.

Validate before launching:

```bash
sparkrun recipe validate recipe/production.yaml
```

### 5. Launch

```bash
sparkrun run recipe/production.yaml
```

Add `--cluster <name>` if you do not use sparkrun’s default cluster.
`sparkrun run` blocks until the instance is ready. Ctrl+C detaches from the logs;
it does not stop the job. First boot on a new revision packs Engram on every node
(~47 GiB, detached, ~1–3 min/node). The engine can serve before the pack
finishes; wait for the packer’s `wrote … .bin` line before benchmarking.

### 6. Verify

```bash
curl -sf http://<head>:8000/health
curl -s http://<head>:8000/get_server_info | \
  jq '{max_total_num_tokens, context_length, tp_size, mem_fraction_static}'
```

Expected: `max_total_num_tokens=9120256`, `context_length=1048576`,
`tp_size=4`, `mem_fraction_static=0.8`.

Then check the boot gates in `docs/gates.md`. The three that matter most:

```text
[replicated_split] ... split ON, window GEMMs (no pad)
[replicated_split] ... compact gather (roce) bit-identical, ON
[moe_b12x_next] armed: routed MoE on b12x_next a7d7d29b
RoCEnante ready: world=4 hcas=rocep1s0f0,roceP2p1f1 gid_index=3
```

and the absence of `slice failed on rank 0` and `CERT head armed`.

`scripts/verify-gates.sh` was run against the live production lane on
2026-10-23; all gates passed.

### 7. Benchmark

Wait at least 10 minutes after `/health`, then bench the live instance:

```bash
sparkrun benchmark perf recipe/production.yaml \
  --skip-run --no-stop --fresh \
  --profile benchmarks/ab-4cell.yaml \
  --output benchmarks/ab-prod-$(date +%Y%m%d-%H%M).yaml
```

Use `--fresh` whenever the boot differs from the one that produced the stored
state. The benchmark ID ignores the recipe file, so without `--fresh` sparkrun
can re-emit old numbers in two seconds and send zero requests.

## Repository layout

```text
recipe/
  production.yaml          flattened, self-contained recipe
  overlay.yaml             readable overlay + why each delta exists
mods/
  prepare-allow-fast-load.patch
docs/
  gates.md                 exact log lines to verify a boot
  boot-and-memory.md       boot timeline, KV sizing, fast-loader trade
  tuning-log.md            chronological experiments, including failures
benchmarks/
  ab-4cell.yaml            4-cell profile used for every number here
  summary.csv              medians per lane, generated from the raw JSON
  csv/                     llama-benchy aggregate CSV per pass
  raw/                     llama-benchy JSON with per-run values
scripts/
  verify-gates.sh          run on the head node
  bench-4cell.sh           wrapper around the command above
```

## Caveats and open work

1. **The source-lane comparison is one pass.** The +33%/+32% numbers are
   against a single n=3 pass of the published `@experimental` recipe; it was
   not repeated warm. A warm A/B on the same image is the missing experiment.
2. **TP_PAD was not isolated with a same-image control.** The split failure is
   proven by logs (42/42 layers), but the full throughput delta between
   `TP_PAD=1` and `TP_PAD=0` on the *same* v2.4 image was not run.
3. **CERT quality was not measured.** CERT is a certified-verify feature. The
   throughput result above says nothing about answer quality or acceptance
   rate. If the guarantee matters, measure it before turning the head off.
4. **The final lane was not run through the 28-cell `spark-arena-v1` profile.**
   All numbers here are the 4-cell probe. A full profile would settle the
   deep-context tail and the c10 cells.
5. **Production overlay vs native v24 lane at d0 c1 is inside the noise band**
   (59.9 vs 64.6, n=3 vs n=6). The base recipe adds three settings
   (`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`,
   `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192`, `DSV41_STATS_SECONDS=60`) and costs
   ~0.9 GB of pool; whether it also costs decode is unresolved. If you do not
   need the base recipe’s behavior, the native v24 recipe is the faster starting
   point.
6. **Fast-loader pool parity was estimated, not measured.** 0.83 should land
   near 9.0M tokens by the measured 1.22 GB per 0.01 rule; only 0.80 was run.

## Attribution and licenses

This repository is **MIT** licensed (see `LICENSE`), covering the text,
recipes, scripts, and the patch.

Upstream work it depends on or references:

- **Little Cedar Group / spark-arena** — source recipe, native sparkrun recipe
  shape, runtime policy, and the `dsv41-native-prepare` mod. MIT.
  <https://github.com/littlecedar/sparkrun-recipe-registry>
  and <https://github.com/spark-arena/recipe-registry>
- **knapcio** — production TP4 profile, `Dockerfile.canary-roce`, the v2.4
  adapter set, b12x/b12x_next, RoCEnante integration. **AGPL-3.0**; the
  Dockerfile and image are not redistributed here.
  <https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4>
- **MiaAI-Lab / Mia** — original DGX Spark DeepSeek deployment, Engram row
  store, launcher lineage.
  <https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks>
- **SGLang**, **FlashInfer**, **DeepGEMM**, **PyTorch**, **NVIDIA** components —
  their respective upstream licenses.
- **DeepSeek-V4.1-Flash-1017** — model weights, MIT, not redistributed here.

See `NOTICE` for the full component list and license notes.

## Questions / comparisons

If you run the same 4-cell profile on your build, open an issue with the
`summary.csv` row and the `get_server_info` capacity. The useful comparison is
same-image, one variable at a time; the tuning log records the sequence we used.
