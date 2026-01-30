# DGX Spark Dual 100G Links: TCP Bonding vs RDMA

## The Challenge

DGX Spark systems include two physical 100 Gigabit Ethernet ports running RoCE (RDMA over Converged Ethernet). The hardware supports 200 Gbps aggregate bandwidth. The question: how to utilize both links for point-to-point data transfers?

Two approaches exist:

1. **Linux bonding** - Standard kernel networking that aggregates multiple interfaces into one logical interface
2. **RDMA (Remote Direct Memory Access)** - Hardware-level data transfers that bypass the operating system kernel

This article focuses on point-to-point transfers for disaggregated inference (prefill nodes sending KV-cache to decode nodes). Collective operations (all-reduce, all-gather) are covered in the [first article](01_Infiniband_Tutorial.md) using NCCL. The [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides executable benchmarking code.

## Measured Performance

The [first article](01_Infiniband_Tutorial.md) established baseline RDMA performance: 93.4 Gbps on a single link with 1-2 microsecond latency. This article adds TCP bonding and NIXL (NVIDIA Inference Xfer Library) measurements:

| Configuration | Throughput | Description |
|---------------|------------|-------------|
| TCP single stream (bonded) | 33.7 Gbps | One connection uses one physical link |
| TCP 4 parallel streams (bonded) | 93.0 Gbps | Multiple connections distribute across both links |
| NIXL single-rail (CPU memory) | 81.8 Gbps | Application-level RDMA with one port |
| NIXL single-rail (GPU memory) | Lower | Stages through host memory (see GPU section) |
| NIXL dual-rail (CPU memory) | 93.4 Gbps | Application-level RDMA with both ports |

**Key findings:**

1. **TCP bonding limitations** - Single streams are limited to one physical link (33.7 Gbps) despite 200 Gbps aggregate capacity. Four parallel streams achieve 93.0 Gbps by distributing across both links, matching single-port RDMA performance but requiring application-level parallelism.

2. **NIXL single-rail (81.8 Gbps)** - Application-level RDMA using one port. The ~12% reduction compared to raw RDMA (see [first article](01_Infiniband_Tutorial.md)) comes from Python interpreter overhead and the UCX (Unified Communication X) abstraction layer.

3. **NIXL dual-rail (93.4 Gbps)** - Configured to use both ports simultaneously, but measured throughput matches single-port performance. UCX automatically stripes large transfers across both links, but coordination overhead (managing two transport lanes, synchronization) negates bandwidth gains for the tested transfer sizes.

4. **Latency characteristics** - RDMA hardware achieves 1-2 microseconds per transfer, while TCP ranges from 50-200 microseconds. NIXL single-rail adds Python layer overhead (17.4 μs average), while dual-rail coordination increases latency to 58.6 μs average. The Python interpreter and user-space scheduling account for the difference between raw RDMA and NIXL measurements.

---

## Linux Bonding

Bonding aggregates multiple network interfaces into a single logical interface. The kernel distributes packets across member interfaces based on the bonding mode.

Bonding provides value for high availability with automatic failover on link failure, management traffic such as SSH, monitoring, and logs, IP-based storage like NFS mounts, and standard TCP services that do not require RDMA. Bonding operates at the IP layer, making it transparent to applications using standard networking APIs.

However, bonding does not help RDMA applications. RDMA uses the verbs API, which accesses network hardware directly—traffic to `rocep1s0f0` or `rocep1s0f1` does not traverse `bond0`. NIXL and any application using `libibverbs` bypass bonding entirely, requiring direct access to the physical RoCE interfaces.

### Bonding Modes

| Mode | Name | Description | Switch Required |
|------|------|-------------|-----------------|
| 0 | balance-rr | Round-robin packet distribution | No |
| 1 | active-backup | One active, others standby | No |
| 2 | balance-xor | Hash-based distribution (IP + port) | No |
| 4 | 802.3ad | LACP negotiation | Yes |

For direct-connected DGX Spark systems without a switch, use mode 2 (balance-xor). This mode distributes flows based on IP and port hash, ensuring packet ordering per connection while allowing parallel streams to use different links. Mode 0 (balance-rr) can cause TCP reordering issues.

### Configuration

Bonding configuration involves loading the bonding kernel module, creating a bond interface in balance-xor mode, adding physical interfaces as members, and setting MTU to 9000 (required—default MTU causes TCP congestion control to throttle throughput to near-zero).

The [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides complete configuration commands. Key parameters:
- Mode: balance-xor (mode 2)
- Hash policy: layer3+4 (distributes flows by IP + port)
- MTU: 9000 (critical for performance)
- Monitor interval: 100ms

### Performance

Balance-xor hashes each TCP connection to one interface, limiting single streams to one link. Application-level parallelism (multiple concurrent streams) distributes load across both links. See the [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) for iperf3 benchmarks and configuration commands.

---

## NIXL for Point-to-Point Transfers

RDMA memory registration fails when network interfaces are bonded. Remove any bond configuration before testing NIXL (see section 4.0 in the [tutorial notebook](02_Multi_Rail_Tutorial.ipynb)). NIXL provides direct memory transfers between specific node pairs (NCCL handles many-to-many collectives; NIXL handles one-to-one transfers). Distributed inference patterns that require point-to-point transfers include KV-cache movement (prefill node sends cache to decode node), tensor shard movement (specific layers between nodes), and disaggregated serving (separating compute stages across machines). NIXL provides direct RDMA transfers with a unified API for CPU memory, GPU memory, and storage.

### Installation

NIXL installs via pip: `pip install nixl[cu13]`

**UCX with CUDA Support:** The default UCX may lack CUDA support. Building UCX from source with `--with-cuda` enables GPU memory registration. The [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides complete build instructions.

### GPUDirect RDMA Limitation on DGX Spark

DGX Spark uses a unified memory architecture where GPU-allocated pinned memory cannot be coherently accessed by PCIe devices, preventing GPUDirect RDMA support.

```
CPU Memory Path (80-100 Gbps):
┌─────────┐                    ┌──────────────┐
│ Remote  │  ─── RDMA READ ───> │  Host DRAM   │
│   NIC   │     (rc_mlx5)       │  (direct)    │
└─────────┘                    └──────────────┘

GPU Memory Path (lower throughput):
┌─────────┐         ┌──────────────┐         ┌──────────┐
│ Remote  │  ─────> │ Host Bounce  │  ─────> │   GPU    │
│   NIC   │  RDMA   │   Buffer     │  copy   │  Memory  │
└─────────┘         └──────────────┘         └──────────┘
                     (rc_mlx5)              (cuda_copy)
```

The CPU memory path achieves higher throughput because RDMA READ operates directly into host DRAM—the NIC writes directly to registered memory in a single step. The GPU memory path shows lower throughput because RDMA READ must go into a host bounce buffer, then UCX `cuda_copy` transport stages data to GPU memory with additional synchronization and memory registration overhead. CPU being faster than GPU confirms that GPU buffers are not on a zero-copy RDMA path—this is expected behavior on DGX Spark.

**Workaround for DGX Spark:** Use host-allocated pinned memory instead of GPU-allocated memory for RDMA transfers. `cudaMalloc` allocates memory on the GPU itself (inaccessible to PCIe devices on DGX Spark), while `cudaHostAlloc` allocates pinned host memory that both the CPU and GPU can access. Allocate buffers with `cudaHostAlloc` and register them with the RDMA stack to achieve full 80-100 Gbps throughput while maintaining CUDA compatibility. Applications should query platform capabilities programmatically to detect whether zero-copy GPU RDMA is available and fall back to host memory when necessary.

### Configuration

NIXL uses a target/initiator pattern where the target node exports memory descriptors and the initiator performs RDMA read/write operations. The [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides complete Python implementations with timing measurements and data verification.

Multi-rail configuration: Set `UCX_NET_DEVICES=rocep1s0f0:1,rocep1s0f1:1` to enable both RoCE ports. UCX automatically stripes large transfers across available devices.

---

## Performance Comparison

The throughput benchmarks use single multi-gigabyte transfers and report aggregate bandwidth. Latency benchmarks require thousands of small transfers with per-iteration timing. Both are provided in the [tutorial notebook](02_Multi_Rail_Tutorial.ipynb).

**Latency percentiles (CPU memory, 4 KB transfers, 1000 iterations):**

| Configuration | Average | P50 | P95 |
|---------------|---------|-----|-----|
| NIXL single-rail | 17.4 μs | 16.2 μs | 20.8 μs |
| NIXL dual-rail | 58.6 μs | 11.1 μs | 166.6 μs |

These measurements include Python interpreter and user-space scheduling overhead. Single-rail and dual-rail latency differ because they use different transport lanes and wireup paths.

---

## Decision Guide

Use NIXL for KV-cache transfers, tensor shard movement, and disaggregated inference workloads that require point-to-point RDMA. Use NCCL for collective operations (see [first tutorial](01_InfiniBand_Tutorial.ipynb)). Use bonding mode 1 (active-backup) for SSH and management traffic, or mode 0 (balance-rr) for NFS storage.

### Coexistence

Bonding and NIXL are not mutually exclusive. Configure bonding for IP traffic and let NIXL use the raw RoCE devices for RDMA. The two paths do not interfere: RDMA verbs access `rocep1s0f0` and `rocep1s0f1` directly without traversing `bond0`.

---

## Disaggregated Inference Implications

Disaggregated inference architectures separate prefill and decode stages across nodes, with KV-cache transfers as the primary data movement pattern.

**Transfer time for 1 GB KV-cache:**
- TCP single stream (33.7 Gbps): 237 ms
- NIXL dual-rail (93.4 Gbps): 86 ms
- NIXL single-rail (81.8 Gbps): 98 ms

For tensor parallelism across two Spark systems, bonding provides no benefit—NCCL uses RDMA directly. Point-to-point patterns (KV-cache movement, tensor shard transfers) use NIXL for application-level RDMA.

---

## Summary

TCP bonding requires application-level parallelism to utilize both links. NIXL provides direct RDMA for point-to-point transfers with 81.8-93.4 Gbps throughput and 17-59 μs latency. The two approaches coexist: bonding for IP traffic, NIXL for RDMA workloads. See the Decision Guide above for workload recommendations.

Given the GPUDirect RDMA limitation on DGX Spark, how does the host memory staging overhead affect inference latency budgets when KV-cache transfers compete with compute on the same memory bus?

---

## References

- [NIXL GitHub Repository](https://github.com/ai-dynamo/nixl)
- [Linux Kernel Bonding Documentation](https://www.kernel.org/doc/Documentation/networking/bonding.txt)
- [UCX Documentation](https://openucx.readthedocs.io/)
- [NCCL Tutorial](01_InfiniBand_Tutorial.ipynb)
