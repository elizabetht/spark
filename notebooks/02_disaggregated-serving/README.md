# Disaggregated LLM Serving on DGX Spark

Split prefill and decode across two DGX Spark nodes using vLLM and NIXL. Understand the architecture before using frameworks like AI Dynamo that abstract it away.

## Hardware

| Node | IP | GPU | Role |
|------|----|-----|------|
| spark-01 | 192.168.100.10 | 1x NVIDIA GB10 | Prefill |
| spark-02 | 192.168.100.11 | 1x NVIDIA GB10 | Decode |

Connected via RoCE (RDMA over Converged Ethernet).

## Learning Modules

### Module 00: [Environment Setup](vllm-native/00_Environment_Setup.ipynb)
Verify GPU, network, and software configuration on both nodes.
**Time**: 20-30 minutes

### Module 01: [Local Inference Baseline](vllm-native/01_Local_Inference_Baseline.ipynb)
Measure single-node vLLM performance with continuous batching. This is the bar that disaggregation must beat.
**Time**: 30-40 minutes

### Module 02: [Understanding KV Cache](vllm-native/02_Understanding_KV_Cache.ipynb)
Calculate KV cache dimensions from model architecture constants. Compare transfer cost over TCP vs RDMA without loading the model.
**Time**: 15-20 minutes

### Module 03: [Replicated Serving](vllm-native/03_Replicated_Serving.ipynb)
Run two independent vLLM instances (one per node, 0.3 `gpu-memory-utilization` each) behind a round-robin proxy. Same hardware footprint as disaggregated serving, providing a fair comparison baseline.
**Time**: 30-40 minutes

### Module 04: [Disaggregated Serving](vllm-native/04_Disaggregated_Serving.ipynb)
Run prefill on spark-01 and decode on spark-02 using vLLM's `NixlConnector` for GPU-to-GPU KV cache transfer over RDMA. Compare against both the single-node baseline (Module 01) and replicated baseline (Module 03).
**Time**: 45-60 minutes

## What Comes Next: AI Dynamo

These notebooks cover the manual approach: two vLLM instances, a proxy, and NixlConnector. This is enough to understand what disaggregated serving does and measure its overhead.

AI Dynamo adds:
- **Service discovery** (etcd): workers register automatically
- **KV-aware routing**: route follow-up requests to nodes that already hold the cache
- **Dynamic scaling**: add/remove prefill and decode workers at runtime
- **Production orchestration**: health checks, failover, metrics

The `ai-dynamo/` directory will contain notebooks for that layer, building on the understanding established here.

## Prerequisites

- Two DGX Spark nodes with SSH access between them
- RDMA-capable network (RoCE or InfiniBand)
- Python virtual environment with vLLM 0.13.0+, NIXL, PyTorch 2.9.0+
- `meta-llama/Llama-3.1-8B-Instruct` cached on both nodes

## Quick Start

```bash
# Activate the virtual environment
source ~/src/github.com/elizabetht/spark/bin/activate

# Start with environment verification
cd notebooks/02_disaggregated-serving/vllm-native
jupyter notebook 00_Environment_Setup.ipynb
```

Work through notebooks in order: 00 → 01 → 02 → 03 → 04.