# Distributed LLM Inference on a Home Lab Kubernetes Cluster

I deployed vLLM on a 3-node Kubernetes cluster with two DGX Spark GPU workers. This documents the architecture, GPU Operator setup, and path to distributed inference with tensor parallelism.

## Why This Matters

Most teams run inference on single GPUs or cloud platforms. This setup demonstrates:
- Multi-node GPU orchestration with production tooling (NVIDIA GPU Operator)
- Infrastructure for distributed inference with tensor parallelism
- Direct connection to RDMA performance (96 Gbps InfiniBand vs 35 Gbps TCP/IP)

The goal: measure whether high-speed interconnects justify the complexity for distributed LLM serving.

## Cluster Architecture

| Node | Role | Hardware | IP Address | Purpose |
|------|------|----------|------------|---------|
| controller | Control Plane | CPU-only | 192.168.1.75 | K8s API, scheduler |
| spark-01 | Worker | DGX Spark (GPU) | 192.168.1.76 | Inference workload |
| spark-02 | Worker | DGX Spark (GPU) | 192.168.1.77 | Inference workload |

The control plane separation keeps 100% of GPU memory available for model weights and inference.

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Kubernetes | MicroK8s 1.31 | Lightweight, simple cluster formation |
| GPU Support | NVIDIA GPU Operator | Production-grade stack (driver, toolkit, device plugin, DCGM) |
| Inference | vLLM | Industry standard for LLM serving |
| Monitoring | Prometheus + DCGM Exporter | GPU utilization and inference metrics |

MicroK8s is appropriate for a 3-node development cluster. Production would use kubeadm or managed K8s.

## Prerequisites

Before starting:

1. Ubuntu 22.04 or later on all nodes
2. SSH key-based authentication configured between nodes
3. Network connectivity (WiFi or Ethernet)
4. `nvidia` user with sudo privileges

The SSH setup is worth getting right first. Every command in this tutorial runs over SSH from the controller to the worker nodes.

```bash
# Test connectivity before proceeding
ssh -o BatchMode=yes nvidia@192.168.1.235 hostname
ssh -o BatchMode=yes nvidia@192.168.1.71 hostname
```

If these commands prompt for a password, run `ssh-copy-id` first.

## Step 1: Install MicroK8s on All Nodes

Install MicroK8s on each node. The installation is identical across control plane and workers.

```bash
# On each node (controller, spark-01, spark-02)
sudo snap install microk8s --classic --channel=1.31/stable
sudo usermod -a -G microk8s $USER
sudo microk8s status --wait-ready
```

The `--channel=1.31/stable` pins to Kubernetes 1.31. This ensures consistency across nodes and avoids surprise upgrades.

## Step 2: Form the Cluster

From the controller, generate a join token:

```bash
# On controller
microk8s add-node
```

This outputs a join command with a one-time token. Run it on each worker:

```bash
# On spark-01 and spark-02
microk8s join 192.168.1.75:25000/<token> --worker
```

The `--worker` flag is important. It tells MicroK8s this node should only run workloads, not participate in control plane decisions.

Verify the cluster:

```bash
# On controller
microk8s kubectl get nodes
```

Expected output:

```
NAME         STATUS   ROLES    AGE   VERSION
controller   Ready    <none>   10m   v1.31.0
spark-01     Ready    <none>   5m    v1.31.0
spark-02     Ready    <none>   3m    v1.31.0
```

## GPU Operator vs Simple Device Plugin

The GPU Operator deploys the full NVIDIA software stack:

| Component | Function |
|-----------|----------|
| NVIDIA Driver | GPU device driver (optional if pre-installed) |
| Container Toolkit | Enables GPU access in containers |
| Device Plugin | Exposes `nvidia.com/gpu` as schedulable resource |
| DCGM Exporter | Exports GPU metrics to Prometheus |
| GPU Feature Discovery | Labels nodes with GPU properties |

A simple device plugin only handles resource scheduling. The GPU Operator is what production inference platforms run.

Installation via Helm:

```bash
# Add NVIDIA Helm repository
microk8s helm3 repo add nvidia https://helm.ngc.nvidia.com/nvidia

# Install with pre-installed drivers
microk8s helm3 install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=false \
  --set toolkit.enabled=true
```

### Why `driver.enabled=false`?

DGX Spark nodes ship with NVIDIA drivers pre-installed on the host OS. The GPU Operator supports two driver deployment modes:

**Mode 1: Containerized Drivers** (`driver.enabled=true`)
- Operator deploys drivers as privileged pods (`nvidia-driver-daemonset`)
- Installs/loads kernel modules from inside containers
- Use when starting with bare nodes without drivers

**Mode 2: Pre-installed Drivers** (`driver.enabled=false`)
- Operator uses existing host drivers
- Skips nvidia-driver-daemonset entirely
- Use when drivers are already installed (DGX Spark case)

Setting `driver.enabled=true` on nodes with existing drivers causes conflicts:
```
modprobe: ERROR: could not insert 'nvidia': File exists
```

Even with `driver.enabled=false`, the operator installs:
- NVIDIA Container Toolkit (maps GPUs into containers)
- Device Plugin (exposes `nvidia.com/gpu` to Kubernetes)
- DCGM Exporter (GPU metrics for Prometheus)  
- GPU Feature Discovery (automatic node labeling)

Verify pre-installed drivers on your nodes:
```bash
nvidia-smi           # Should show GPU details
ls /dev/nvidia*      # Should list GPU device files
modinfo nvidia       # Should show loaded driver module
```

### Runtime Configuration: The "nvidia" vs "nvidia-container-runtime" Issue

**Problem:** GPU Operator pods stuck in `Init:0/1` status for extended periods.

**Root Cause:** The GPU Operator expects a containerd runtime named `nvidia`, but MicroK8s on DGX Spark systems configures a runtime named `nvidia-container-runtime`. This naming mismatch prevents init containers from validating GPU availability.

**Diagnosis:**
```bash
# Check containerd configuration on GPU nodes
sudo grep -A 3 'runtimes.nvidia' /var/snap/microk8s/current/args/containerd-template.toml

# Problematic output shows:
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia-container-runtime]
  runtime_type = "io.containerd.runc.v2"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia-container-runtime.options]
    BinaryName = "nvidia-container-runtime"

# GPU Operator init containers look for "nvidia", not "nvidia-container-runtime"
```

**Solution:** Add a `nvidia` runtime entry alongside the existing `nvidia-container-runtime` config:

```bash
# Add this to /var/snap/microk8s/current/args/containerd-template.toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
    BinaryName = "/usr/bin/nvidia-container-runtime"

# Then restart containerd
sudo snap restart microk8s.daemon-containerd
```

Both runtime names can coexist. The existing `nvidia-container-runtime` configuration remains functional, and the new `nvidia` entry provides GPU Operator compatibility. After restarting containerd, init containers validate successfully and pods reach Running state.

## Distributed Inference with Tensor Parallelism

The interesting part: deploy a model across both nodes using tensor parallelism.

Tensor parallelism splits model layers across multiple GPUs. Each forward pass requires synchronization between GPUs—this is where the InfiniBand link matters.

**Deployment approach:**

1. Use KubeRay operator to manage a Ray cluster across worker nodes
2. Deploy vLLM with `--tensor-parallel-size=2`
3. Configure NCCL to use InfiniBand for GPU-to-GPU communication

**Critical NCCL configuration:**

```yaml
env:
- name: NCCL_IB_DISABLE
  value: "0"
- name: NCCL_SOCKET_IFNAME
  value: "enp1s0f0np0"  # InfiniBand interface
- name: NCCL_DEBUG
  value: "INFO"
```

This enables RDMA transport for NCCL collective operations. Without this, NCCL falls back to TCP/IP over IPoIB—exactly the 35 Gbps vs 96 Gbps difference measured in the InfiniBand benchmarks.

**vLLM command:**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000
```

A 70B model requires tensor parallelism to fit in memory and benefit from distributed execution.

## Why RDMA Matters for Inference

From previous benchmarks:
- `ib_write_bw` (RDMA): 96 Gbps
- `iperf3` (TCP/IP): 35 Gbps

Tensor parallelism requires constant communication between GPUs. Every forward pass involves:
1. Splitting input tensors across GPUs
2. Computing partial results
3. All-reduce to combine results
4. Broadcasting to all GPUs

At 100 tokens/sec output, that synchronization happens 100 times per second. The 2.7x bandwidth difference translates directly to latency.

**Questions to answer with benchmarks:**
- What is the latency penalty for tensor parallelism?
- Does RDMA reduce that penalty enough to justify multi-node?
- At what model size does distribution break even?

## Single-Node vLLM Baseline

Deploy vLLM on a single GPU as baseline:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-single
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    command:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
      - --model
      - meta-llama/Llama-3.1-8B-Instruct
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
    resources:
      limits:
        nvidia.com/gpu: 1
```

This exposes an OpenAI-compatible API endpoint. Benchmark with concurrent requests to measure throughput:

```python
response = requests.post(
    "http://vllm-single-svc:8000/v1/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "Explain distributed systems:",
        "max_tokens": 100
    }
)
```

Baseline metrics needed:
- Tokens/sec at various batch sizes
- P50, P95, P99 latency
- GPU utilization during serving

## Monitoring with DCGM and Prometheus

The GPU Operator includes DCGM Exporter, which exposes metrics like:

```
DCGM_FI_DEV_GPU_UTIL: GPU utilization (%)
DCGM_FI_DEV_GPU_TEMP: GPU temperature
DCGM_FI_DEV_MEM_COPY_UTIL: Memory bandwidth utilization
DCGM_FI_PROF_PCIE_RX_BYTES: PCIe receive throughput
DCGM_FI_PROF_NVLINK_RX_BYTES: NVLink receive throughput
```

Enable Prometheus in MicroK8s:

```bash
microk8s enable prometheus
```

DCGM metrics are automatically scraped. Build dashboards in Grafana to visualize:
- GPU utilization during inference
- Memory bandwidth saturation
- NVLink/PCIe throughput (proxy for inter-GPU communication)
- Tokens/sec correlated with GPU metrics

This data makes the performance story concrete.

## What Gets Measured

| Metric | Single-Node Baseline | Distributed (TP=2) |
|--------|---------------------|-------------------|
| Throughput (tokens/sec) | TBD | TBD |
| P95 Latency (ms) | TBD | TBD |
| GPU Utilization (%) | TBD | TBD |
| Inter-GPU Bandwidth | N/A | TBD |

The compelling article requires these numbers. Without them, it is theoretical.

## Architecture Decisions

| Choice | Rationale |
|--------|-----------|
| CPU control plane | 100% of GPU memory available for models |
| GPU Operator | Production-grade stack, not just device plugin |
| MicroK8s | Lightweight for 3-node dev cluster |
| InfiniBand direct connect | Eliminate switch hop, measure best-case RDMA |
| vLLM | Industry standard for LLM serving |

## Next: Get the Data

The infrastructure is documented. The path to distributed inference is clear. The remaining work:

1. Deploy distributed vLLM with KubeRay
2. Configure NCCL for RDMA transport
3. Run benchmarks: single-node vs distributed
4. Measure latency distribution under load
5. Correlate DCGM metrics with inference performance

Then the article writes itself: "Llama 3.1 70B on 2 DGX Spark nodes: X tokens/sec at Y ms P95 latency. Here is why the 96 Gbps InfiniBand link matters."

## Troubleshooting: Complete Cluster Reset

If the cluster becomes corrupted or you encounter version skew issues, reset all nodes and start fresh.

**Problem: Version Mismatch**

Kubernetes requires the control plane and worker nodes to be within ±1 minor version. If you see:
- `NotReady` nodes in `kubectl get nodes`
- GPU Operator pods in `CrashLoopBackOff` or `ContainerCreating` state
- Different containerd or Kubernetes versions across nodes

Reset is the cleanest path forward.

**Reset Procedure**

```bash
# Step 1: Remove worker nodes from cluster (on controller)
ssh nvidia@192.168.1.75
microk8s remove-node spark-01
microk8s remove-node spark-02

# Step 2: Leave cluster (on each worker)
ssh nvidia@192.168.1.76 'sudo microk8s leave'
ssh nvidia@192.168.1.77 'sudo microk8s leave'

# Step 3: Purge MicroK8s from all nodes
ssh nvidia@192.168.1.75 'sudo snap remove microk8s --purge'
ssh nvidia@192.168.1.76 'sudo snap remove microk8s --purge'
ssh nvidia@192.168.1.77 'sudo snap remove microk8s --purge'
```

The `--purge` flag removes all configuration, data, and cluster state. After purging, reinstall MicroK8s using **the same channel** on all nodes:

```bash
# Install on all three nodes
sudo snap install microk8s --classic --channel=1.32/stable
```

Then proceed through the cluster formation steps from the notebook.

---

*This documents the setup. The companion notebook contains executable steps. See the InfiniBand benchmarking article for context on why RDMA matters.*
