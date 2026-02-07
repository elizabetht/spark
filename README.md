# DGX Spark Network Benchmarks and LLM Serving

Hands-on tutorials for RDMA networking, disaggregated LLM serving, and Kubernetes cluster setup on NVIDIA DGX Spark systems. All notebooks run real code on real hardware: no simulated benchmarks, no hardcoded numbers.

## Hardware

| Node | GPU | Network |
|------|-----|---------|
| spark-01 | 1x NVIDIA GB10 | RoCE (RDMA over Converged Ethernet) |
| spark-02 | 1x NVIDIA GB10 | RoCE (RDMA over Converged Ethernet) |

Direct-connected via dual 100 Gbps links.

## Tutorials

### 01: [RDMA Networking Deep Dive](notebooks/01_infiniband-tutorial/)

Measure RDMA performance between two direct-connected DGX Spark nodes. Covers single-link benchmarks, multi-rail bonding, and NIXL GPU-to-GPU transfers.

| Notebook | Description |
|----------|-------------|
| [01_InfiniBand_Tutorial](notebooks/01_infiniband-tutorial/01_InfiniBand_Tutorial.ipynb) | RDMA basics, `ib_write_bw` vs `iperf3`, single-link bandwidth and latency measurements |
| [02_Multi_Rail_Tutorial](notebooks/01_infiniband-tutorial/02_Multi_Rail_Tutorial.ipynb) | Dual-link performance: Linux bonding vs NIXL multi-rail RDMA comparison |
| [03_NixlBench](notebooks/01_infiniband-tutorial/03_NixlBench.ipynb) | Systematic `nixlbench` benchmarking for GPU-to-GPU RDMA transfer throughput |

### 02: [Disaggregated LLM Serving](notebooks/02_disaggregated-serving/)

Split prefill and decode across two DGX Spark nodes using vLLM and NIXL. Builds understanding progressively from single-node baseline through replicated serving to disaggregated inference.

| Notebook | Description |
|----------|-------------|
| [00_Environment_Setup](notebooks/02_disaggregated-serving/vllm-native/00_Environment_Setup.ipynb) | Verify GPU, network, and software configuration on both nodes |
| [01_Local_Inference_Baseline](notebooks/02_disaggregated-serving/vllm-native/01_Local_Inference_Baseline.ipynb) | Single-node vLLM performance with continuous batching (the bar to beat) |
| [02_Understanding_KV_Cache](notebooks/02_disaggregated-serving/vllm-native/02_Understanding_KV_Cache.ipynb) | KV cache dimensions from model architecture constants, TCP vs RDMA transfer cost |
| [03_Replicated_Serving](notebooks/02_disaggregated-serving/vllm-native/03_Replicated_Serving.ipynb) | Two independent vLLM instances behind a round-robin proxy (fair comparison baseline) |
| [04_Disaggregated_Serving](notebooks/02_disaggregated-serving/vllm-native/04_Disaggregated_Serving.ipynb) | Prefill on spark-01, decode on spark-02 via vLLM's `NixlConnector` for GPU-to-GPU KV cache transfer |
| [05_Production_Benchmarking](notebooks/02_disaggregated-serving/vllm-native/05_Production_Benchmarking.ipynb) | `guidellm` sweeps across all three configurations with TTFT/TPOT/ITL breakdowns and P50/P95/P99 distributions |

### 03: [MicroK8s Cluster Setup](notebooks/03_microk8s-cluster-setup/)

Deploy a 3-node Kubernetes cluster (1 CPU controller + 2 DGX Spark GPU workers) using MicroK8s, with GPU Operator configuration for containerized inference.

| Notebook | Description |
|----------|-------------|
| [01_MicroK8s_Cluster_Setup](notebooks/03_microk8s-cluster-setup/01_MicroK8s_Cluster_Setup.ipynb) | Cluster formation, GPU Operator install, vLLM deployment on Kubernetes |

## Prerequisites

- Two DGX Spark nodes with SSH access between them
- RDMA-capable network (RoCE or InfiniBand)
- Python virtual environment with vLLM 0.13.0+, NIXL, PyTorch 2.9.0+
- `meta-llama/Llama-3.1-8B-Instruct` cached on both nodes
