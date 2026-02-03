# NIXL Multi-Rail Benchmarks: Getting Both 100G Links Working

*Follow-up to [TCP Bonding vs RDMA](02_Multi_Rail_Tutorial.md)*

---

## The Problem We Left Unsolved

In the previous article, I benchmarked TCP bonding versus RDMA on DGX Spark's dual 100G RoCE links. The results were clear: RDMA bypasses the kernel and delivers 93.4 Gbps on a single link with microsecond latency. TCP bonding, even with parallel streams, could not match it.

But one question remained: can NIXL actually aggregate both links to approach 200 Gbps?

The previous tests showed NIXL dual-rail achieving only 93.4 Gbps, the same as single-rail. UCX was supposed to stripe transfers across both links automatically, but coordination overhead seemed to cancel out the bandwidth gains.

This article documents systematic benchmarking with `nixlbench` to measure what dual-rail actually delivers.

---

## Test Configuration

Two DGX Spark nodes connected directly via dual 100G RoCE links:
- `rocep1s0f0` and `rocep1s0f1` on each node
- IP addresses: 192.168.100.10 (spark-01) and 192.168.100.11 (spark-02)
- No switch, direct cable connections
- MTU 9000 (jumbo frames)

The benchmark tool: `nixlbench` from the NIXL repository. Unlike the Python-based tests in the previous article, nixlbench is a compiled C++ benchmark that eliminates interpreter overhead.

---

## Results

| Test | Memory | Bandwidth | Link 1 | Link 2 | Notes |
|------|--------|-----------|--------|--------|-------|
| Single-Rail DRAM | CPU | 92.5 Gbps | 12.18 GB | 0 GB | Baseline |
| Dual-Rail DRAM | CPU | 105.4 Gbps | 6.09 GB | 6.09 GB | Both links active |
| Single-Rail VRAM | GPU | 4.0 Gbps | 7.32 GB | 0 GB | GPUDirect limited |
| Dual-Rail VRAM | GPU | 3.2 Gbps | 7.32 GB | 0 GB | Single link only |

### DRAM Transfers: Dual-Rail Works

The dual-rail DRAM test confirmed what we hoped: both links carried traffic. RDMA counters showed a perfect 50/50 split (6.09 GB on each link). UCX successfully striped the transfers.

However, the aggregate throughput was 105.4 Gbps, not the theoretical 176+ Gbps. The 13% improvement over single-rail (92.5 Gbps) is real but modest.

Why not 2x? UCX multi-rail introduces coordination overhead. At 67 MB block sizes with 1000+ iterations, the benchmark measured 13.17 GB/sec (105.4 Gbps). The limiting factor appears to be synchronization between the two transport lanes rather than raw link bandwidth.

### VRAM Transfers: GPUDirect Limitations

The VRAM tests showed dramatically lower throughput: 4.0 Gbps for single-rail, 3.2 Gbps for dual-rail.

This confirms what we documented in the previous article: DGX Spark's unified memory architecture does not support true GPUDirect RDMA. Transfers to GPU memory must stage through host bounce buffers.

```
GPU Memory Path:
NIC → Host Bounce Buffer → cuda_copy → GPU Memory
```

The dual-rail VRAM test only used one link (7.32 GB on link 1, 0 on link 2) despite UCX multi-rail configuration. When transfers bottleneck on the `cuda_copy` staging step, adding a second RDMA link provides no benefit.

---

## Verifying Link Utilization

One lesson from this benchmarking: always verify which links are actually carrying traffic.

Reading RDMA port counters before and after each test:

```
/sys/class/infiniband/rocep1s0f0/ports/1/counters/port_xmit_data
/sys/class/infiniband/rocep1s0f1/ports/1/counters/port_xmit_data
```

These counters report transmitted data in 4-byte words. Multiply by 4 to get bytes.

For the dual-rail DRAM test:
- Before: rocep1s0f0 = 1,193 GB, rocep1s0f1 = 420 GB
- After: rocep1s0f0 = 1,199 GB, rocep1s0f1 = 426 GB
- Delta: 6.09 GB on each link

This confirms UCX striped the transfers across both links. Without this verification, the benchmark output alone cannot distinguish between "using both links at 50% each" and "using one link at 100%."

---

## Configuration That Works

### Dual-Rail DRAM (CPU Memory)

```bash
/usr/local/nixlbench/bin/nixlbench \
  --etcd_endpoints http://192.168.100.11:2379 \
  --backend UCX \
  --initiator_seg_type DRAM \
  --target_seg_type DRAM \
  --total_buffer_size 4294967296 \
  --start_block_size 65536 \
  --max_block_size 67108864 \
  --num_iter 1000 \
  --warmup_iter 100
```

Omitting `--device_list` allows UCX to auto-select all available RoCE devices.

### Single-Rail (Baseline)

Add `--device_list rocep1s0f0` to restrict to one link.

---

## What This Means for Disaggregated Inference

For KV-cache transfers between prefill and decode nodes:

| Transfer Size | Single-Rail (92.5 Gbps) | Dual-Rail (105.4 Gbps) |
|---------------|-------------------------|------------------------|
| 1 GB | 86 ms | 76 ms |
| 4 GB | 346 ms | 303 ms |
| 8 GB | 691 ms | 607 ms |

The 14% improvement from dual-rail is meaningful for large transfers but may not justify the additional complexity for smaller KV-cache sizes.

More importantly: if your transfers involve GPU memory on DGX Spark, dual-rail provides no benefit. The cuda_copy bottleneck dominates.

---

## Recommendations

1. **Use dual-rail for CPU memory transfers** when you need maximum throughput. The 14% improvement is real.

2. **Do not expect dual-rail benefits for GPU memory** on DGX Spark. The architecture requires bounce buffers that serialize transfers regardless of available RDMA links.

3. **Verify link utilization** with RDMA port counters. Configuration changes may not have the expected effect.

4. **Consider using `cudaHostAlloc`** instead of `cudaMalloc` for RDMA buffers. Pinned host memory achieves full link speed and remains accessible to CUDA kernels.

---

## Full Benchmark Code

The complete Jupyter notebook with all tests, RDMA counter verification, and automated result parsing is available at:

[03_NixlBench.ipynb](03_NixlBench.ipynb)

Previous article: [TCP Bonding vs RDMA](02_Multi_Rail_Tutorial.md)

---

*Tested on DGX Spark with dual ConnectX-7 100G RoCE NICs, UCX 1.20, NIXL main branch.*
