# Replication vs Disaggregation: Benchmarking Two-Node LLM Inference with vLLM and NIXL

Disaggregated serving is the architecture behind high-throughput LLM inference at scale. Companies like Deepseek, along with frameworks such as AI Dynamo, vLLM, and Mooncake, use it in production. The idea: instead of one GPU handling both prompt processing (prefill) and token generation (decode), split them across dedicated hardware connected by RDMA.

But does it actually help? And under what conditions?

This project connects two DGX Spark nodes via RoCE (RDMA over Converged Ethernet), runs Llama-3.1-8B-Instruct, and measures everything. No frameworks hiding the details, just vLLM, NIXL, and a Python proxy.

## The Setup

Two NVIDIA DGX Spark systems, each with a GB10 GPU. Connected over RoCE at 100 Gbps link speed. A third CPU-only node runs the proxy that routes requests.

Three configurations were tested using the same memory budget (0.3 `gpu-memory-utilization`):

| Configuration | Hardware | Architecture |
|---------------|----------|-------------|
| Single node | 1 GPU | Full pipeline on one node |
| Replicated | 2 GPUs | Two independent instances, round-robin proxy |
| Disaggregated | 2 GPUs | Prefill on node 1, decode on node 2, KV cache over NIXL/RDMA |

The replicated baseline is the comparison most benchmarks skip. Two GPUs running independent replicas is the simplest way to use additional hardware. Disaggregation has to beat that to justify its complexity.

## The Results

### Single Request Performance

| Architecture | Latency | Throughput | Throughput vs Baseline |
|-------------|---------|------------|----------------------|
| Single node | 6,874 ms | 14.5 tok/s | — |
| Replicated | 7,304 ms | 13.7 tok/s | 0.94x |
| Disaggregated | 8,925 ms | 11.2 tok/s | 0.77x |

For a single request, disaggregation is the slowest option. This makes sense: the request passes through the proxy to the prefill node, the KV cache transfers over RDMA to the decode node, and only then does token generation begin. Two network hops with no offsetting parallelism.

Replication adds minimal overhead (6.3%), which is the cost of the proxy forwarding the request.

### Batch Performance (8 Concurrent Requests)

| Architecture | Throughput | Avg Latency | Throughput vs Baseline |
|-------------|-----------|-------------|----------------------|
| Single node | 122.5 tok/s | 817 ms | — |
| Replicated | 26.5 tok/s | 18,633 ms | 0.22x |
| Disaggregated | 101.9 tok/s | 7,782 ms | 0.83x |

Under concurrent load, the story changes. Disaggregated serving delivered 101.9 tok/s, reaching 83% of the single-node throughput, while the replicated setup struggled at 26.5 tok/s.

The disaggregated pipeline overlaps: while the decode node generates tokens for request N, the prefill node is already processing request N+1. Two GPUs working in parallel on different phases of different requests.

## Key Observations

**1. The fair baseline matters.** Most disaggregation benchmarks compare against a single node. That is a 2-GPU architecture vs a 1-GPU architecture, and two GPUs will look better. Using the same two GPUs in a replicated configuration reveals whether disaggregation provides an architectural advantage or just a hardware advantage.

**2. KV cache transfer is real overhead.** For Llama-3.1-8B with a 512-token prompt, the KV cache is approximately 32 MB (32 layers × 8 KV heads × 128 head_dim × 512 tokens × 2 bytes × 2 for K and V). Over RDMA, that transfers in under a millisecond. Over TCP, it would take 10-100x longer, potentially negating any benefit.

**3. The proxy is the orchestrator.** vLLM's `kv_transfer_params` mechanism is straightforward: the proxy sends the prompt to prefill with `do_remote_decode: true`, gets back cache location metadata (`remote_engine_id`, `remote_block_ids`), and forwards that to the decode instance. The decode node pulls the KV cache over NIXL/RDMA and generates the response. Understanding this protocol is essential before adopting frameworks that abstract it away.

**4. Disaggregation is not always the answer.** For uniform, low-concurrency workloads, simple replication is simpler and faster per-request. Disaggregation wins when you have high concurrency, asymmetric workloads (long prefills with short decodes, or vice versa), or need to scale prefill and decode capacity independently.

## The Architecture, Visually

**Replicated:** each GPU handles everything independently.

```
  Client → Proxy (round-robin) → Instance A (prefill + decode)
                                → Instance B (prefill + decode)
```

**Disaggregated:** GPUs specialize by phase.

```
  Client → Proxy → Prefill Node (prompt processing)
                        ↓ KV cache via NIXL/RDMA
                    Decode Node (token generation) → Response
```

The tradeoff: replication avoids transfer overhead but cannot specialize hardware. Disaggregation pays the transfer cost but enables independent scaling of compute-bound prefill and memory-bandwidth-bound decode.

## What This Leads To

These notebooks cover the manual approach: raw vLLM, a Python proxy, and NixlConnector. Production systems add:

- **KV-aware routing**: direct follow-up requests to nodes that already hold the KV cache (cache reuse across requests)
- **Dynamic scaling**: add prefill workers when prompt processing is the bottleneck, add decode workers when generation queues back up
- **Service discovery**: workers register and deregister automatically

This is what NVIDIA AI Dynamo provides. But understanding the underlying mechanics first makes evaluating those abstractions possible.

## The Notebooks

Five Jupyter notebooks, designed to run in order on DGX Spark hardware:

1. **Environment Setup** — Verify GPU, network, RDMA, and software dependencies
2. **Local Inference Baseline** — Single-node vLLM performance (the "before" measurement)
3. **Understanding KV Cache** — Calculate cache dimensions and transfer costs from model architecture constants
4. **Replicated Serving** — Two independent instances behind a round-robin proxy (the fair comparison)
5. **Disaggregated Serving** — Prefill/decode split with NIXL/RDMA KV cache transfer

Each notebook produces metrics that feed into the next. The progression is intentional: you cannot evaluate an optimization without establishing what you are optimizing from.

The full source is on GitHub: [github.com/elizabetht/spark](https://github.com/elizabetht/spark)

## What's Next in This Series

This article covers Part 1: understanding disaggregated serving from first principles with vLLM and NIXL. The following installments will build on this foundation:

**Part 2: Production Benchmarking with guidellm / vllm bench** — The manual benchmarks in this series are designed for learning, not for production evaluation. The next installment will run the same three configurations (single node, replicated, disaggregated) through standardized benchmarking tools. Expect TTFT, TPOT, P50/P95/P99 percentile distributions, and load sweeps across varying request rates.

**Part 3: Disaggregated Serving with SGLang** — vLLM is not the only engine supporting prefill/decode disaggregation. SGLang implements a different approach to KV cache management and transfer. Running the same benchmarks on the same hardware with a different engine reveals which performance characteristics are architectural and which are engine-specific.

**Part 4: AI Dynamo, the Hard Way** — AI Dynamo adds KV-aware routing, dynamic scaling, and service discovery on top of disaggregated serving. Instead of deploying it as a black box, this installment will build up each layer manually: etcd for service registry, the planner for autoscaling decisions, and the KV-aware router. The goal is to understand what each component does before trusting the orchestration to handle it.

**Part 5: MicroK8s Cluster Deployment** — Moving from bare-metal SSH commands to Kubernetes-orchestrated inference. Deploy vLLM workers as pods, expose RDMA devices through the container runtime, and manage the prefill/decode topology through Kubernetes-native tooling.
---

*Built on two NVIDIA DGX Spark systems connected via RoCE. Model: Llama-3.1-8B-Instruct. Inference engine: vLLM 0.13.0 (cu130). KV transfer: NIXL over RDMA.*
