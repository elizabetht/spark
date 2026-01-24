# DGX Spark Network Benchmarks: RDMA Performance Over RoCE

This article covers RDMA (Remote Direct Memory Access) benchmarking between two direct-connected DGX Spark systems. One notable finding: DGX Spark uses RoCE (RDMA over Converged Ethernet), not native InfiniBand, despite the similar cabling and tooling.

The results show why this distinction matters less than expected for most workloads.

---

## RDMA Overview

Traditional networking involves CPU overhead on every packet: interrupt handling, kernel buffer copies, and context switches. At high packet rates, the CPU becomes a bottleneck.

RDMA eliminates this overhead. The network card writes directly to application memory, bypassing the kernel entirely. Measured latency: 1-2 microseconds vs 50-200μs for standard Ethernet. That's a 50-100x improvement.

For multi-GPU inference, this matters. GPUs (Graphics Processing Units) constantly exchange KV-caches and attention states. Network latency directly impacts token generation throughput.

## RoCE vs InfiniBand

Running `ibv_devinfo` on DGX Spark shows devices named `roceP2p1s0f0`. The "roce" prefix indicates RoCE (RDMA over Converged Ethernet), not native InfiniBand.

Both technologies provide RDMA capabilities: zero-copy transfers, kernel bypass, and low latency. The difference is the physical layer: RoCE runs over standard Ethernet, InfiniBand uses a dedicated fabric.

Note: The ConnectX ports on DGX Spark are configured in Ethernet mode with no firmware option to switch to native InfiniBand. This means DGX Spark cannot connect to InfiniBand switches (e.g., Mellanox SB7800). RoCE works for back-to-back Spark connections and Ethernet-based clustering. See [this NVIDIA forum discussion](https://forums.developer.nvidia.com/t/connecting-dgx-spark-to-mellanox-infiniband-sb7800/355444) for details.

**Practical differences:**

| Factor | InfiniBand | RoCE |
|--------|------------|------|
| Switches | Requires InfiniBand switches | Works with standard Ethernet switches |
| Flow control | Lossless by default | Requires PFC (Priority Flow Control) / ECN (Explicit Congestion Notification) configuration |
| Scale | More predictable at datacenter scale | Well-suited for rack-level deployments |

Major cloud providers deploy native InfiniBand on high-end GPU instances (H100/B200 clusters). DGX Spark uses RoCE, which is appropriate for its target deployment scale.

---

## Test Environment

Configuration: Two DGX Spark systems, direct-connected without a switch.

Note on interface naming: Modern kernels use predictable naming. Interfaces appear as `enp1s0f0np0` and `enp1s0f1np1` rather than the traditional `ib0`/`ib1`. The `ibv_devinfo` output confirms RoCE operation. All standard tooling works unchanged.

For IP address configuration on the RoCE interfaces, refer to NVIDIA's official guide: [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks)

---

## Benchmark Results

**Test 1: RDMA Bandwidth**

```bash
# Server
ib_write_bw

# Client
ib_write_bw 192.168.100.11
```

Result: ~12,000 MB/sec (`BW average[MB/sec]` column), equivalent to ~96 Gbps.

Note: Benchmarks report in Bytes/sec; network specifications use bits/sec. Divide MB/sec by 125 to convert to Gbps.

**RDMA operation types:** `ib_write_bw` tests one-sided RDMA writes (sender writes directly to receiver memory without receiver CPU involvement). `ib_send_bw` tests two-sided operations (receiver posts buffers, receives completion notification). For hardware validation, `ib_write_bw` is the standard test.

**Test 2: TCP Comparison**

Running iperf3 over the same link: 35 Gbps.

This is expected behavior. iperf3 uses TCP/IP (Transmission Control Protocol/Internet Protocol), which traverses the kernel networking stack and IPoIB (IP over InfiniBand) translation layer. The overhead consumes 60-70% of available bandwidth.

Production inference frameworks (vLLM, TensorRT-LLM, TGI) use NCCL (NVIDIA Collective Communications Library) with native RDMA, achieving the full 96 Gbps.

**Test 3: Latency**

```bash
# Server
ib_write_lat

# Client
ib_write_lat 192.168.100.11
```

Result: 1-2 microseconds (`t_avg` column). Standard Ethernet ping latency: 50-200μs.

**Test 4: Dual Link Configuration**

DGX Spark has two RoCE ports. Adding a second cable doubles aggregate bandwidth to ~24,000 MB/sec (~192 Gbps). Per-link latency remains unchanged.

NCCL automatically utilizes both links without configuration changes.

**Test 5: NCCL Collectives**

The above tests measure raw network performance. Production code uses NCCL, which warrants direct testing with the nccl-tests suite. See NVIDIA's [NCCL Stacked Sparks playbook](https://build.nvidia.com/spark/nccl/stacked-sparks) for setup instructions.

**Available nccl-tests benchmarks:**

| Test | Operation | Use Case |
|------|-----------|----------|
| `all_gather_perf` | Collect data from all GPUs to all GPUs | Tensor parallelism (gathering sharded outputs) |
| `all_reduce_perf` | Sum/reduce across all GPUs | Gradient synchronization, tensor parallelism |
| `broadcast_perf` | One GPU sends to all others | Distributing inputs, model weights |
| `reduce_scatter_perf` | Reduce then scatter results | Data parallelism, ZeRO optimization |
| `sendrecv_perf` | Point-to-point transfers | Pipeline parallelism, KV-cache movement |

For this benchmark, `all_gather_perf` was used:

`mpirun` launches the test across both nodes from a single command. It SSHes to each host, starts the NCCL test process, and coordinates execution. Run from one machine only (e.g., spark-01); mpirun handles the remote process on spark-02.

```bash
mpirun -np 2 -H 192.168.200.12:1,192.168.200.13:1 \
    --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
    -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=INFO \
    $HOME/src/github.com/NVIDIA/nccl-tests/build/all_gather_perf
```

**Key indicators from NCCL_DEBUG output:** The output shows `NET/IB` with RoCE interfaces, confirming RDMA is active. Channels appear as `via NET/IB/4` and `via NET/IB/5`, indicating multiple RDMA paths are in use. Bus bandwidth reaches ~16 GB/s for 32 MB messages and ~22 GB/s for 16 GB messages. If `NET/Socket` appears instead of `NET/IB`, NCCL is falling back to TCP—troubleshooting required.

**Large message test (16 GB):**

For maximum throughput measurement, testing with larger messages reduces per-message overhead:

```bash
mpirun -np 2 -H 192.168.200.12:1,192.168.200.13:1 \
    --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
    -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=INFO \
    $HOME/src/github.com/NVIDIA/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

Result:
```
#       size      time   algbw   busbw
#        (B)      (us)  (GB/s)  (GB/s)
 17179869184   387603   44.32   22.16
# Avg bus bandwidth    : 21.9736
```

With 16 GB messages, busbw reaches ~22 GB/s, demonstrating the full potential of dual 100G RoCE links working together.

---

## Implications for LLM Inference

**Tensor parallelism:** Large models (e.g., Llama 70B) require distribution across multiple GPUs. Each forward pass executes all-reduce operations. At 1-2μs per network hop, communication overhead is negligible. At 200μs, latency accumulates to milliseconds per token.

**KV-cache management:** Long context windows produce large KV-caches (Key-Value caches). Continuous batching requires frequent cache transfers. Network bandwidth becomes a limiting factor with slow interconnects.

**Disaggregated serving:** Architectures separating prefill and decode phases across nodes require KV-cache transfer between them. RDMA enables microsecond-level transfers; standard Ethernet adds tens of milliseconds to TTFT (Time-To-First-Token).

---

## Summary

RDMA bandwidth: 12,000 MB/sec (~96 Gbps). TCP over the same link: ~4,400 MB/sec (35 Gbps). The difference reflects kernel bypass, not hardware capability.

For LLM (Large Language Model) inference, latency is the critical metric. RDMA delivers 1-2μs; standard Ethernet delivers 50-200μs. With tensor parallelism requiring multiple all-reduce operations per token, this difference directly impacts generation throughput.

Whether native InfiniBand or RoCE, the RDMA capability is what matters for inference performance. These benchmarks provide a baseline for evaluating network configuration.

---

#InfiniBand #RoCE #RDMA #LLMInference #AIInfrastructure #NVIDIA #DGX
