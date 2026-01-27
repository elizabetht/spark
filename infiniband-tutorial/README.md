# RDMA Networking Deep Dive: A Hands-On Tutorial with NVIDIA DGX Spark

> **Target Audience**: Technology leaders, senior directors, and managers seeking to understand high-performance networking for AI/ML infrastructure.
>
> **Note**: DGX Spark uses **RoCE (RDMA over Converged Ethernet)**, not native InfiniBand. Same RDMA benefits, same tools, same performance—just running over Ethernet hardware. This tutorial covers both.

## Table of Contents
1. [What is RDMA?](#what-is-rdma)
2. [InfiniBand vs RoCE vs Ethernet](#infiniband-vs-roce-vs-ethernet)
3. [Hardware Setup](#hardware-setup)
4. [Hands-On Experiments](#hands-on-experiments)
5. [Results Analysis](#results-analysis)
6. [Business Implications](#business-implications)

---

## Tutorials in This Repository

| Tutorial | Description | Format |
|----------|-------------|--------|
| [01_InfiniBand_Tutorial.ipynb](01_InfiniBand_Tutorial.ipynb) | Hands-on RDMA benchmarking with code cells | Jupyter Notebook |
| [02_Multi_Rail_Tutorial.ipynb](02_Multi_Rail_Tutorial.ipynb) | Bonding vs RDMA performance comparison tests | Jupyter Notebook |
| [02_Multi_Rail_Tutorial.md](02_Multi_Rail_Tutorial.md) | Bonding vs NCCL vs NIXL for dual 100G links | Markdown |
| [LINKEDIN_ARTICLE.md](LINKEDIN_ARTICLE.md) | Summary article with benchmark results | Markdown |

---

## What is RDMA?

### The Simple Explanation

Imagine you're running a warehouse:
- **Regular Ethernet (TCP/IP)** is like having workers manually carry packages, check addresses, confirm delivery—lots of overhead
- **RDMA (Remote Direct Memory Access)** is like having dedicated pneumatic tubes between stations—direct, fast, and the tubes handle all the delivery logistics automatically

### Technical Definition

**RDMA** allows network cards to read/write directly to application memory, bypassing the CPU entirely. There are two main ways to get RDMA:

| Technology | InfiniBand | RoCE (RDMA over Converged Ethernet) |
|------------|------------|-------------------------------------|
| **Physical Layer** | InfiniBand fabric | Standard Ethernet |
| **Switches** | InfiniBand switches | Regular Ethernet switches |
| **Configuration** | Works out of the box | Needs PFC/ECN setup |
| **Used By** | H100/B200 clusters | DGX Spark, many cloud instances |
| **Performance** | Excellent | Excellent |

### Key Characteristics (Both InfiniBand and RoCE)

| Feature | Description |
|---------|-------------|
| **RDMA** | Remote Direct Memory Access - bypasses CPU for data transfers |
| **Low Latency** | Sub-microsecond latency (vs milliseconds for Ethernet) |
| **High Bandwidth** | Up to 400 Gbps (NDR) per port |
| **Lossless** | Hardware-level flow control prevents packet loss |
| **CPU Offload** | Network adapter handles protocol processing |

---

## InfiniBand vs Wireless Ethernet

### The Fundamental Differences

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA TRANSFER COMPARISON                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WIRELESS ETHERNET (Wi-Fi 6E):                                      │
│  ┌─────┐    Radio     ┌─────────┐    Radio     ┌─────┐             │
│  │ CPU │◄──Waves──────│ Router  │──────Waves──►│ CPU │             │
│  │     │   (shared    │(process │   (shared    │     │             │
│  │ RAM │    medium)   │ packets)│    medium)   │ RAM │             │
│  └─────┘              └─────────┘              └─────┘             │
│    ▲                                              ▲                 │
│    └──── CPU must process every packet ──────────┘                 │
│                                                                     │
│  INFINIBAND (RDMA):                                                 │
│  ┌─────┐   Direct    ┌─────────┐    Direct    ┌─────┐             │
│  │ CPU │   Cable     │   IB    │    Cable     │ CPU │             │
│  │     │             │ Switch  │              │     │             │
│  │ RAM │◄═══════════►│(optional)◄════════════►│ RAM │             │
│  └─────┘  Memory to  └─────────┘  Memory to   └─────┘             │
│           Memory                   Memory                          │
│    ▲                                              ▲                 │
│    └──── CPU is FREE to do other work ───────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Aspect | Wireless Ethernet (Wi-Fi 6E) | Wired Ethernet (10GbE) | InfiniBand (HDR) |
|--------|------------------------------|------------------------|------------------|
| **Max Bandwidth** | ~2.4 Gbps | 10 Gbps | 200 Gbps |
| **Typical Latency** | 1-10 ms | 50-150 µs | 0.5-2 µs |
| **CPU Overhead** | Very High | High | Near Zero (RDMA) |
| **Packet Loss** | Common | Rare | Zero (lossless) |
| **Interference** | High | None | None |
| **Distance** | ~100m | 100m (copper) | 100m (copper), 10km (fiber) |
| **Power Consumption** | Low | Medium | Medium-High |
| **Cost** | Low | Medium | High |
| **Use Case** | Consumer/Office | Enterprise | HPC/AI/Storage |

### Why InfiniBand Matters for AI/ML

1. **GPU-to-GPU Communication**: During distributed training, GPUs need to exchange gradients millions of times. InfiniBand's low latency is critical.

2. **RDMA**: The GPU/CPU doesn't waste cycles managing network transfers - data moves directly between memory regions.

3. **Predictable Performance**: No interference, no packet loss = consistent training times.

---

## Hardware Setup

### Your Equipment
- 2x NVIDIA DGX Spark systems
- 1-2x high-speed cables (ConnectX adapters with RoCE)

### Physical Connection Options

**Option 1: Single Cable Direct Connection**
```
┌──────────────┐         ┌──────────────┐
│  DGX Spark   │◄═══════►│  DGX Spark   │
│    Node 1    │  RoCE   │    Node 2    │
│              │  Link   │              │
│  Port 1 ════════════════════ Port 1  │
│  Port 2 (unused)       (unused) Port 2│
└──────────────┘         └──────────────┘
        ~100 Gbps per link
```

**Option 2: Dual Cable Direct Connection (Recommended)**
```
┌──────────────┐         ┌──────────────┐
│  DGX Spark   │◄═══════►│  DGX Spark   │
│    Node 1    │  2x IB  │    Node 2    │
│              │ Cables  │              │
│  Port 1 ════════════════════ Port 1  │
│  Port 2 ════════════════════ Port 2  │
└──────────────┘         └──────────────┘
        ~200 Gbps aggregated
```

**Option 3: With Switch (Scalable)**
```
┌──────────────┐         ┌──────────────┐
│  DGX Spark   │         │  DGX Spark   │
│    Node 1    │         │    Node 2    │
└──────┬───────┘         └───────┬──────┘
       │                         │
       │      ┌─────────┐       │
       └──────│IB Switch│───────┘
              └─────────┘
```

### Connection Steps

1. **Power off both systems** (recommended for first-time setup)
2. **Connect InfiniBand cable(s)** to the IB ports (usually labeled, distinctive QSFP connector)
3. **Power on systems**
4. **Verify connections** with `ibstat` (look for "State: Active")

---

## Files in This Repository

| File | Description |
|------|-------------|
| `01_InfiniBand_Tutorial.ipynb` | Interactive notebook: RDMA basics, hardware checks, speed tests |
| `02_NIXL_Multi_Rail_Tutorial.ipynb` | Interactive notebook: Multi-rail configuration and NIXL transfers |
| `02_NIXL_Multi_Rail_Tutorial.md` | Article: Bonding, NCCL multi-rail, and NIXL for point-to-point |
| `LINKEDIN_ARTICLE.md` | Summary article for sharing |
| `README.md` | This file |
| `agents.md` | Writing guidelines for technical content |

---

## Hands-On Experiments

### Tutorial 01: RDMA Basics

The first notebook covers:

1. **Hardware detection** - Verify InfiniBand adapters are recognized
2. **Link status** - Confirm cables are connected and active
3. **Single-link bandwidth** - Test with `ib_write_bw` (~96 Gbps expected)
4. **Single-link latency** - Test with `ib_write_lat` (~1-2 µs expected)
5. **Dual-link bandwidth** - Compare single vs dual cable performance
6. **TCP/IP comparison** - iperf3 over IPoIB vs native RDMA

Run the notebook: `01_InfiniBand_Tutorial.ipynb`

### Tutorial 02: Multi-Rail and NIXL

The second notebook covers:

1. **Linux bonding** - Aggregate IPoIB interfaces (TCP/IP traffic only)
2. **NIXL installation** - NVIDIA Inference Xfer Library setup
3. **Basic NIXL transfer** - Point-to-point GPU memory transfer
4. **Multi-rail configuration** - Using both RoCE ports with UCX
5. **NCCL vs NIXL** - When to use collectives vs point-to-point

Run the notebook: `02_NIXL_Multi_Rail_Tutorial.ipynb`

---

## Expected Results Summary

| Test | Single Cable | Dual Cable | Notes |
|------|--------------|------------|-------|
| `ib_write_bw` | ~12,000 MB/sec (~96 Gbps) | ~24,000 MB/sec (~192 Gbps) | RDMA native |
| `ib_write_lat` | ~1-2 µs | ~1-2 µs | Latency unchanged |
| `iperf3` | ~35 Gbps | ~70 Gbps | TCP/IP overhead |
| NCCL all_gather | N/A | ~22 GB/s busbw | Collectives |
| NIXL point-to-point | ~12 GB/s | ~22-24 GB/s | Direct transfers |

---

## Business Implications

### ROI Considerations

| Factor | Impact |
|--------|--------|
| **Training Time** | 2-10x faster distributed training |
| **GPU Utilization** | Higher utilization = better ROI on expensive GPUs |
| **Scalability** | Linear scaling with more nodes |
| **Total Cost of Ownership** | Higher upfront, lower long-term per-compute-unit |

### When to Choose InfiniBand

✅ **Use InfiniBand when:**
- Running distributed AI/ML training
- GPUs need to communicate frequently
- Latency-sensitive workloads
- Building HPC clusters

❌ **Ethernet is sufficient when:**
- Single-node training
- Batch inference (latency less critical)
- Cost is primary constraint
- Standard enterprise networking needs

---

## Next Steps

1. Run the experiment scripts in this repository
2. Document your results
3. Share findings with your team

---

*Created with NVIDIA DGX Spark | January 2026*
