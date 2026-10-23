# Boot timeline, KV sizing, and the fast-loader trade

These are the numbers behind the recipe’s `gpu_memory_utilization` and
`DSV41_FAST_LOAD` choices. They were measured on the same four nodes, same
checkpoint, same image.

## Stock-loader boot

Timeline from `sparkrun_serve.log` (UTC), one representative boot:

```text
17:56:44  Load weight begin (target)
17:56:50  Multi-thread loading shards 0/48
17:57:48  48/48 -> Finished streaming dequant fp8 wo_a -> twins 40 -> DROP 40
          [240 s with no log line]
18:01:48  DSV41 shared-expert K padding
18:01:50  Load weight end. elapsed=305.99 s
18:01:52  draft (DSpark) load begin
18:02:59  Load weight end. elapsed=66.48 s
```

The 240-second gap is not the adapter: it is bulk NVMe page-in of the
checkpoint on all four nodes, seen only because the loader goes quiet there.
During that window:

- `nvme0n1` read 0.7–0.8 GB/s on the head, 1.2–1.6 GB/s on the workers;
  ~115 KB per request, sequential.
- GPU utilization 0–2%, package power ~13 W, CPU 12–16%.
- ~180 GB read on the head; the container’s cgroup `io.stat` after boot shows
  211 GB read for 71.75 GB resident target weights (~3× amplification from
  mmap page faults on TP-sliced tensors).
- MemAvailable 126 → 40 GB at 17:57; 5–6 GB of swap touched; ~1.3k swap-ins/s
  in the tail. No OOM.

Memory is unified on GB10. Read free memory from `/proc/meminfo`, not
`nvidia-smi`; the GPU has no separate VRAM number to report.

## Where the KV pool comes from

The engine sizes the pool from the memory left after the weight loads:

```text
bytes_per_full_token=1670.75
available_bytes=15.18 GB
available_gpu_mem=...
max_total_num_tokens=9120256
```

At `mem_fraction_static=0.80` on the production lane the pool is **9,120,256
tokens**. The `max_total_tokens: 9400000` in the recipe is a request, not a
setting; the engine logs the profiled value and caps to it.

Rules measured on this box:

| Change | Effect |
|---|---|
| `mem_fraction_static` +0.01 | ~1.22 GB more pool budget ≈ 0.73M tokens |
| 0.80 | 9.0–9.1M tokens, stable |
| 0.86 | ~full 1M context, stable; the practical ceiling |
| 0.88 | early-OOM risk |
| 0.90 | hard failure |

The production lane keeps 0.80 because 9.12M tokens is comfortably above the
1M-context working set and leaves headroom for the detached Engram packer and
page cache.

## Engram packs

First boot of a checkpoint revision writes the packed row store on **every
node** under:

```text
{runtime_cache_dir}/engram/v1/{revision}/
  layer-1.bin   ~25 GiB
  layer-14.bin  ~25 GiB
  identity.json
```

Budget ~47 GiB/node. Packing runs detached at low CPU/I/O priority; the engine
serves before it finishes, reading from the checkpoint until the pack is ready.
The `identity.json` binds the pack to the image SHA, config/index hashes,
revision, and TP degree. A different image digest or revision invalidates it and
forces a re-pack. The packer log is
`{runtime_cache_dir}/state/engram-pack-<rev>-r<rank>.log`.

Warming the pack’s L1 cache cuts NVMe reads ~5× but did not change decode
throughput in our tests (see `tuning-log.md`). It is worth having for I/O
stability and TTFT consistency, not for tok/s.

## Fast-loader trade

`DSV41_FAST_LOAD=1` reads the checkpoint eagerly instead of relying on mmap page
faults:

| Metric | Stock | Fast |
|---|---:|---:|
| Target weight load | 305.99 s | 66.76 s |
| Draft (DSpark) load | 66.48 s | 4.30 s |
| Boot to ready | 482 s | 148 s |
| Eager I/O | 211 GB cgroup read | 155.2 GB over 45 shards, 0 skipped |
| Resident weights | 73.70 GB | 78.57 GB (+4.87 GB) |
| Pinned peak | — | 78–99 slabs × 256 MB = 19.4–24.6 GB/node |
| KV pool at 0.80 | 9,016,832 | 6,804,224 (−24.5%) |

The pinned-memory peak bottomed MemFree at 8.4 GB on one node with no OOM and
no swap alarm. All four ranks printed the same reduced pool, so the loss is
resident weight memory, not a sizing race.

The trade in one sentence: **~5.5 minutes off every boot, paid for with ~2.2M
tokens of KV at the same `mem_fraction_static`**. If you need the pool back, run
0.83 (estimated ~9.0M by the 1.22 GB per 0.01 rule; not measured here).

The fast loader did not change decode throughput beyond session noise. Its
value is restart time, not tok/s.

## First-pass rule

Do not benchmark the first pass after a boot. Two identical passes on one boot,
five minutes apart, scored 98.8 vs 117.8 tok/s at d0 c10. The first pass
overlaps the tail of the weight pull and first-touch kernel autotune. Discard
it; use the second pass, or wait at least 10 minutes after `/health`.
