# Boot gates

Every claim in this repository depends on the adapters actually being armed.
A boot can be healthy on `/health` while a gate is silently off; that costs
throughput without an error. Check these lines after every launch on a new
image, recipe, or environment block.

## Where the log is

`docker logs <container>` is empty for a sparkrun SGLang job: the serve command
redirects stdout inside the container. The real log is
`/tmp/sparkrun_serve.log` **inside the container**, which is also visible from
the host through the scheduler process’s root:

```bash
pid=$(pgrep -f 'sglang::scheduler' | head -1)
sudo -n grep -a <pattern> /proc/$pid/root/tmp/sparkrun_serve.log
```

Log timestamps are UTC; the host clock is local. Each rank has its own log, so
repeat this on every node if a gate is missing on the head. `scripts/verify-gates.sh`
automates the head-node checks.

## Gates that must be present

### replicated split

```text
[replicated_split] armed for ['wqkv_a', 'engram.wkv'] (rows <= 96; compact gather roce + minimal)
[replicated_split] model.layers.0.self_attn.wqkv_a 1792x5120: split ON, window GEMMs (no pad) (rank 0 window True, slice True; 1/1 layers on so far)
[replicated_split] model.layers.0.self_attn.wqkv_a: compact gather (roce) bit-identical, ON (rank 0 equal True; 1/1 layers on so far)
[replicated_split] model.layers.1.engram.wkv 25600x6144: split ON, window GEMMs (no pad) (rank 0 window True, slice True; 2/2 layers on so far)
```

**Failure:** if you see

```text
[replicated_split] ... slice failed on rank 0: AssertionError()
[replicated_split] ... split OFF
```

on every layer (42/42), the adapter’s `DSV41_TP_PAD` is still at its default
`1`. Set `DSV41_TP_PAD=0`.

### MoE b12x_next

```text
[moe_b12x_next] armed: routed MoE on b12x_next a7d7d29b (W4A8, in-place repack); graph bs [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16], rows 5/6, ladder [128, 256, 512, 1024, 2048, 4096], tune=True
[moe_b12x_next] geometry ready {'geometry': {'E': 384, 'K': 5120, 'N': 576, 'topk': 6, ...}}
```

**Failure:** no `armed` line means the image does not carry b12x_next, or
`DSV41_MOE_B12X_NEXT` is not `1`. The v2.4 image always carries it.

### RoCEnante fabric

```text
RoCEnante ready: world=4 hcas=rocep1s0f0,roceP2p1f1 gid_index=3 max_size=2097152
```

The HCA names must be the **active** ports on each node. On our nodes the
sibling ports are down (`enp1s0f1np1`, `enP2p1s0f0np0`), so the pair is
`rocep1s0f0` + `roceP2p1f1`. `world=4` must match the node count.

### Spec sync free / eager glue / L2 prefetch

```text
[spec_sync_free] armed (all)
[eager_glue] ...
[L2_PREFETCH] ... v3
```

These are knapcio v2.4 adapter gates. A missing gate is a throughput loss with
no error; check that the v2.4 environment block is present.

### Capacity

```text
DSV4 memory calculation: ... bytes_per_full_token=1670.75, available_bytes=15.34 GB, ... full_token=9120256
max_total_num_tokens=9120256, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=16, context_len=1048576, available_gpu_mem=14.04 GB
```

`max_total_num_tokens` is the profiled KV capacity. It is not the value you
passed in `max_total_tokens`; the engine caps to its own profile. If it is much
lower than expected, see `boot-and-memory.md`.

## Gates that must be absent

When running the production lane:

```text
CERT head armed
```

must not appear. `DSV41_CERT_HEAD=0` is deliberate; if the line is present, the
environment did not reach the adapter.

Also check for `XID`, `NV_ERR_NO_MEMORY`, `OOM`, and `slice failed` anywhere in
the log. A single `slice failed` line can be a harmless probe, but a run of 42
means the split is off.

## One-line summary

`scripts/verify-gates.sh` prints PASS/FAIL for the split, MoE, RoCEnante, CERT,
and capacity gates, and exits non-zero on failure. Run it on the head node after
`/health` returns 200.
