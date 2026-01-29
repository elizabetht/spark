# DGX Spark Dual 100G Links: TCP Bonding vs RDMA

## The Challenge

DGX Spark systems include two physical 100 Gigabit Ethernet ports running RoCE (RDMA over Converged Ethernet). The hardware supports 200 Gbps aggregate bandwidth. The question: how to utilize both links for point-to-point data transfers?

Two approaches exist:

1. **Linux bonding** - Standard kernel networking that aggregates multiple interfaces into one logical interface
2. **RDMA (Remote Direct Memory Access)** - Hardware-level data transfers that bypass the operating system kernel

## Measured Performance

The [first article](01_Infiniband_Tutorial.md) established baseline RDMA performance: 93.4 Gbps on a single link with 1-2 microsecond latency. This article adds TCP bonding and NIXL (NVIDIA Inference Xfer Library) measurements:

| Configuration | Throughput | Description |
|---------------|------------|-------------|
| TCP single stream (bonded) | 33.7 Gbps | One connection uses one physical link |
| TCP 4 parallel streams (bonded) | 93.0 Gbps | Multiple connections distribute across both links |
| NIXL single-rail (CPU memory) | 81.8 Gbps | Application-level RDMA with one port |
| NIXL dual-rail (CPU memory) | 93.4 Gbps | Application-level RDMA with both ports |

**Key findings:**

1. **TCP bonding limitations** - Single streams are limited to one physical link (33.7 Gbps) despite 200 Gbps aggregate capacity. Four parallel streams achieve 93.0 Gbps by distributing across both links, matching single-port RDMA performance but requiring application-level parallelism.

2. **NIXL single-rail (81.8 Gbps)** - Application-level RDMA using one port. The ~12% reduction compared to raw RDMA (see [first article](01_Infiniband_Tutorial.md)) comes from Python interpreter overhead and the UCX (Unified Communication X) abstraction layer.

3. **NIXL dual-rail (93.4 Gbps)** - Configured to use both ports simultaneously, but measured throughput matches single-port performance. UCX automatically stripes large transfers across both links, but coordination overhead (managing two transport lanes, synchronization) negates bandwidth gains for the tested transfer sizes.

4. **Latency characteristics** - RDMA hardware achieves 1-2 microseconds per transfer, while TCP ranges from 50-200 microseconds. NIXL single-rail adds Python layer overhead (17.4 μs average), while dual-rail coordination increases latency to 58.6 μs average. The Python interpreter and user-space scheduling account for the difference between raw RDMA and NIXL measurements.

## Scope

This article documents point-to-point transfer configuration and benchmarks relevant to disaggregated inference architectures where specific node pairs exchange data (prefill sending KV-cache to decode nodes, for example). Collective operations across many nodes (all-reduce, all-gather) are covered in the [first article](01_Infiniband_Tutorial.md) using NCCL (NVIDIA Collective Communications Library). The accompanying [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides executable benchmarking code.

---

## Configuration Challenge

DGX Spark provides two 100G RoCE ports. The [first article](01_Infiniband_Tutorial.md) covers baseline RDMA performance testing with `ib_write_bw`, which shows 11,679 MB/sec (93.4 Gbps) on a single link. Two approaches exist for aggregating both links in point-to-point transfers:

- **Linux bonding** - Kernel-level interface aggregation for TCP/IP traffic
- **NIXL (NVIDIA Inference Xfer Library)** - Application-level RDMA for point-to-point transfers

---

## Linux Bonding

Bonding aggregates multiple network interfaces into a single logical interface. The kernel distributes packets across member interfaces based on the bonding mode.

### When Bonding Helps

| Use Case | Example |
|----------|---------|
| High availability | Automatic failover on link failure |
| Management traffic | SSH, monitoring, logs |
| IP-based storage | NFS mounts |
| Non-RDMA applications | Standard TCP services |

### When Bonding Does Not Help

Bonding operates at the IP layer. RDMA applications use the verbs API, which accesses the network hardware directly. Traffic to `mlx5_0` or `mlx5_1` does not traverse `bond0`.

NIXL and any application using `libibverbs` bypass bonding entirely.

### Bonding Modes

| Mode | Name | Description | Switch Required |
|------|------|-------------|-----------------|
| 0 | balance-rr | Round-robin packet distribution | No |
| 1 | active-backup | One active, others standby | No |
| 2 | balance-xor | Hash-based distribution (IP + port) | No |
| 4 | 802.3ad | LACP negotiation | Yes |

For direct-connected DGX Spark systems without a switch, use mode 2 (balance-xor). This mode distributes flows based on IP and port hash, ensuring packet ordering per connection while allowing parallel streams to use different links. Mode 0 (balance-rr) can cause TCP reordering issues.

### Configuration

Bonding configuration involves loading the bonding kernel module, creating a bond interface in balance-xor mode, adding physical interfaces as slaves, and setting MTU to 9000 (required—default MTU causes TCP congestion control to throttle throughput to near-zero).

The [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides complete configuration commands. Key parameters:
- Mode: balance-xor (mode 2)
- Hash policy: layer3+4 (distributes flows by IP + port)
- MTU: 9000 (critical for performance)
- Monitor interval: 100ms

### Bonding Performance

**Measured results with balance-xor mode:**

- Single stream: **33.7 Gbps** - Balance-xor hashes each connection to one interface. A single TCP flow cannot utilize both links.
- Four parallel streams (`-P 4`): **93.0 Gbps** - Multiple streams hash to different interfaces, distributing load across both 100G links.

**Key observation:** Single streams are limited to one link (~34 Gbps), but parallel streams achieve ~93 Gbps total throughput. This matches RDMA single-link performance but requires application-level parallelism.

See the [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) for iperf3 commands and detailed testing procedures.

---

## Remove Bond Before NIXL Testing

RDMA memory registration fails when network interfaces are enslaved to a bond. The verbs API requires direct access to the physical device, but bonded interfaces associate with `bond0` instead of the underlying hardware.

**Symptoms when bond is active:**
- `ibv_reg_mr` failures during memory registration
- NIXL `register_memory()` returns empty descriptors or raises exceptions
- `show_gids` shows GID entries pointing to `bond0` instead of `rocep1s0f0`/`rocep1s0f1`

**Why this happens:** When interfaces join a bond, the kernel reassigns their identity. RDMA operations that worked on `enp1s0f0np0` now fail because the GID table references `bond0`, which has no RDMA capability.

### Remove Bond on Both Nodes

Bond removal requires bringing down the bond interface, removing slaves, deleting the bond, and restoring IP addresses to physical interfaces. The [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides complete commands.

### Verify RDMA Devices Are Accessible

Use `ibdev2netdev` to verify both RoCE devices show status "Up" and `show_gids` to confirm GID entries point to physical interfaces (not bond0). Once the bond is removed and interfaces are up with IP addresses, NIXL memory registration will succeed.

---

## NIXL for Point-to-Point Transfers

NIXL provides direct memory transfers between specific node pairs. NCCL handles many-to-many collectives; NIXL handles one-to-one transfers.

### Use Cases

Distributed inference involves patterns that do not map to collectives:

| Pattern | Description |
|---------|-------------|
| KV-cache transfer | Prefill node sends cache to decode node |
| Tensor shard movement | Moving specific layers between nodes |
| Disaggregated serving | Separating compute stages across machines |

These patterns require point-to-point transfers. NIXL provides direct RDMA transfers with a unified API for CPU memory, GPU memory, and storage.

### Installation

NIXL installs via pip: `pip install nixl[cu12]`

**UCX with CUDA Support:** The default UCX may lack CUDA support. Building UCX from source with `--with-cuda` enables GPU memory registration. The [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides complete build instructions.

**DGX Spark limitation:** GPUDirect RDMA is not supported. The unified memory architecture prevents coherent GPU memory access from PCIe devices. RDMA registration of CUDA buffers fails; UCX stages GPU transfers through host memory.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Inference Framework                          │
│                  (vLLM, TensorRT-LLM, Dynamo)                   │
├─────────────────────────────────────────────────────────────────┤
│                         NIXL Agent                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Memory    │  │  Transfer   │  │  Metadata   │             │
│  │  Sections   │  │  Backends   │  │   Handler   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│              UCX Backend (RDMA/RoCE/InfiniBand)                 │
└─────────────────────────────────────────────────────────────────┘
```

NIXL abstracts memory types and transfer mechanisms with single memory registration across transport backends.

### NIXL Transfer Pattern

NIXL implements point-to-point RDMA transfers through a target/initiator pattern:

**Target node:**
1. Creates NIXL agent with metadata server enabled
2. Allocates and registers memory buffers
3. Exports memory descriptors
4. Waits for connection and sends descriptors to initiator

**Initiator node:**
1. Creates NIXL agent
2. Allocates local buffers
3. Connects to target's metadata server
4. Retrieves remote memory descriptors
5. Initiates RDMA read/write operations
6. Measures transfer performance

The [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides complete Python implementations for both target and initiator nodes, including timing measurements and data verification.

### Multi-Rail Configuration

NIXL uses UCX for transport. Setting `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1` enables both RoCE ports. UCX automatically stripes large transfers across available devices. See the [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) for environment configuration.

### Latency Measurement vs Throughput Scripts

The dual-rail throughput scripts measure bulk transfer rates with single multi-gigabyte transfers. This methodology is unsuitable for latency measurement:

- Single large transfer timing reports aggregate bandwidth, not per-transfer latency
- Large transfers use rendezvous and pipelining, measuring sustained throughput rather than one-way latency
- Latency measurement requires thousands of small transfers with per-iteration timing and percentile statistics

---

## Performance Comparison

**Latency percentiles (CPU memory, 4 KB transfers, 1000 iterations):**

| Configuration | Average | P50 | P95 |
|---------------|---------|-----|-----|
| NIXL single-rail | 17.4 μs | 16.2 μs | 20.8 μs |
| NIXL dual-rail | 58.6 μs | 11.1 μs | 166.6 μs |

**Note:** These measurements include Python interpreter and user-space scheduling overhead. Single-rail and dual-rail latency differ because they use different transport lanes and wireup paths.

---

## Decision Guide

| Workload | Recommended Approach |
|----------|----------------------|
| KV-cache transfer | NIXL |
| Tensor shard movement | NIXL |
| Disaggregated inference | NIXL |
| Collective operations | NCCL (see [first tutorial](01_InfiniBand_Tutorial.ipynb)) |
| SSH and management | Bonding (mode 1) |
| NFS storage | Bonding (mode 0) |

### Coexistence

Bonding and NIXL are not mutually exclusive. Configure bonding for IP traffic and let NIXL use the raw RoCE devices for RDMA. The two paths do not interfere: RDMA verbs access `mlx5_0` and `mlx5_1` directly without traversing `bond0`.

---

## Disaggregated Inference Implications

Disaggregated inference architectures separate prefill and decode stages across nodes, with KV-cache transfers as the primary data movement pattern.

**Transfer time for 1 GB KV-cache:**
- TCP single stream (33.7 Gbps): 237 ms
- NIXL dual-rail (93.4 Gbps): 86 ms
- NIXL single-rail (81.8 Gbps): 98 ms

For tensor parallelism across two Spark systems, bonding provides no benefit—NCCL uses RDMA directly. Point-to-point patterns (KV-cache movement, tensor shard transfers) use NIXL for application-level RDMA.

---

## Troubleshooting

### UCX Version Mismatch Between Nodes

**Symptom:** NIXL connection fails with `NIXL_ERR_NOT_FOUND`

**Cause:** UCX requires matching versions on both endpoints. Check with `ucx_info -v` and ensure both nodes use identical UCX versions (e.g., both 1.21.0).

**Fix:** Build matching UCX versions on both nodes from the same git tag.

### P2P Interface IP Conflict

**Symptom:** UCX connects but uses wrong interface

**Cause:** DGX Spark P2P interface (`enP2p1s0f0np0`) may have IP in same subnet as RoCE interfaces.

**Fix:** Remove conflicting IP with `ip addr del` or restrict UCX with `UCX_NET_DEVICES=rocep1s0f0:1`

### Interface Down

**Symptom:** `ibdev2netdev` shows interface as `(Down)`  
**Fix:** `sudo ip link set <interface> up`

### NIXL GPU Registration Fails

**Error:** `ibv_reg_mr failed: Bad address`  
**Cause:** DGX Spark does not support GPUDirect RDMA (`nvidia-peermem` fails to load)  
**Workaround:** Use CPU pinned memory fallback. UCX stages GPU transfers through host memory.

### TCP Throughput Near Zero

**Cause:** MTU mismatch—default 1500 causes TCP throttling  
**Fix:** Set MTU 9000 on all interfaces

### NIXL Does Not Use Both Rails

**Fix:** Set `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1` and verify with `ucx_info -d | grep mlx5`

### Bond Not Forming

**Fix:** Load bonding module (`modprobe bonding`), verify with `cat /proc/net/bonding/bond0`

### Transfer Falls Back to TCP

**Checks:** Verify RDMA devices active (`ibstat`), UCX has verbs support (`ucx_info -v | grep verbs`), `libibverbs` installed

---

## Summary

TCP bonding and NIXL serve different purposes on DGX Spark dual 100G RoCE:

- **TCP bonding** - Single streams use one link (~34 Gbps), parallel streams distribute across both (~93 Gbps). Suitable for management traffic, SSH, NFS.
- **NIXL** - Application-level RDMA for point-to-point transfers. Single-rail achieves 81.8 Gbps with lower latency (17.4 μs avg), dual-rail achieves 93.4 Gbps with higher coordination overhead (58.6 μs avg).

Bonding and RDMA coexist without conflict—bonding handles IP traffic while RDMA applications access hardware directly.

Collective operations (gradient sync, tensor parallelism) using NCCL are covered in the [first tutorial](01_InfiniBand_Tutorial.ipynb).

---

## References

- [NIXL GitHub Repository](https://github.com/ai-dynamo/nixl)
- [Linux Kernel Bonding Documentation](https://www.kernel.org/doc/Documentation/networking/bonding.txt)
- [UCX Documentation](https://openucx.readthedocs.io/)
- [NCCL Tutorial](01_InfiniBand_Tutorial.ipynb)
