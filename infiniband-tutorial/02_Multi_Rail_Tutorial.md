# DGX Spark Dual 100G Links: TCP Bonding vs RDMA

## The Challenge

DGX Spark systems include two physical 100 Gigabit Ethernet ports running RoCE (RDMA over Converged Ethernet). The hardware supports 200 Gbps aggregate bandwidth. The question: how to utilize both links for point-to-point data transfers?

Two approaches exist:

1. **Linux bonding** - Standard kernel networking that aggregates multiple interfaces into one logical interface
2. **RDMA (Remote Direct Memory Access)** - Hardware-level data transfers that bypass the operating system kernel

## Measured Performance

Testing between two directly-connected DGX Spark nodes revealed the following throughput characteristics:

| Configuration | Throughput | Description |
|---------------|------------|-------------|
| TCP single stream (bonded) | 33.7 Gbps | One connection uses one physical link |
| TCP 4 parallel streams (bonded) | 93.0 Gbps | Multiple connections distribute across both links |
| RDMA single link | 93.4 Gbps | Direct hardware access, kernel bypass |
| NIXL single-rail | 81.8 Gbps | Application-level RDMA with one port |
| NIXL dual-rail | 93.4 Gbps | Application-level RDMA with both ports |

**Key observation:** Standard TCP networking with bonding requires multiple parallel connections to utilize both links. RDMA achieves similar throughput on a single link by eliminating kernel overhead. NIXL (NVIDIA Inference Xfer Library) provides RDMA capabilities through a Python API.

**Latency characteristics:**
- TCP: 50-200 microseconds per transfer
- RDMA hardware: 1-2 microseconds
- NIXL single-rail: 17.4 microseconds average (Python layer)
- NIXL dual-rail: 58.6 microseconds average (Python layer)

## Scope

This article documents point-to-point transfer configuration and benchmarks relevant to disaggregated inference architectures where specific node pairs exchange data (prefill sending KV-cache to decode nodes, for example). Collective operations across many nodes (all-reduce, all-gather) are covered in the [first article](01_Infiniband_Tutorial.md) using NCCL (NVIDIA Collective Communications Library). The accompanying [tutorial notebook](02_Multi_Rail_Tutorial.ipynb) provides executable benchmarking code.

**Note on GPU memory:** DGX Spark does not support GPUDirect RDMA due to its unified memory architecture. GPU buffer transfers require host staging through CPU memory. Benchmarks in this article use CPU memory allocation.

---

## Configuration Challenge

DGX Spark provides two 100G RoCE ports. Running `ib_write_bw` on a single link shows 11,679 MB/sec (93.4 Gbps). Two approaches exist for aggregating both links in point-to-point transfers:

| Approach | Traffic Type | Max Throughput | Latency |
|----------|--------------|----------------|---------|
| Linux bonding | TCP/IP | 34-93 Gbps | 50-200 μs |
| NIXL single-rail (CPU memory) | Point-to-point RDMA | 81.8 Gbps | 17.4 μs (avg) |
| NIXL dual-rail (CPU memory) | Point-to-point RDMA | 93.4 Gbps | 58.6 μs (avg) |

The throughput difference comes from the data path. TCP/IP traverses the kernel networking stack with buffer copies, interrupt handling, and context switches. RDMA bypasses the kernel entirely.

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

**For this tutorial:** Commands were executed on spark-02 only. The configuration is transient (ip commands) and does not persist after reboot.

**MTU 9000 is required:** Default MTU (1500) causes TCP congestion control to throttle connections, resulting in near-zero throughput.

**On spark-02 (192.168.100.11) - executed:**

```bash
sudo modprobe bonding
sudo ip link add bond0 type bond mode balance-xor
sudo ip link set bond0 type bond miimon 100
sudo ip link set bond0 type bond xmit_hash_policy layer3+4

sudo ip link set enp1s0f0np0 down
sudo ip link set enp1s0f0np0 master bond0
sudo ip link set enp1s0f0np0 up

sudo ip link set enp1s0f1np1 down
sudo ip link set enp1s0f1np1 master bond0
sudo ip link set enp1s0f1np1 up

sudo ip addr add 192.168.100.11/24 dev bond0
sudo ip link set bond0 mtu 9000
sudo ip link set bond0 up
```

**For production (both nodes):** Replace `192.168.100.11` with `192.168.100.10` on spark-01. Configuration can be verified with `cat /proc/net/bonding/bond0`.

### Bonding Performance

**Measured results with balance-xor mode:**

```bash
# Server (spark-01)
iperf3 -s -B 192.168.100.10

# Client (single stream)
iperf3 -c 192.168.100.10 -t 10
```

Result: **33.7 Gbps**. Balance-xor hashes each connection to one interface. A single TCP flow cannot utilize both links.

```bash
# Client (four parallel streams)
iperf3 -c 192.168.100.10 -t 10 -P 4
```

Result: **93.0 Gbps**. Multiple streams hash to different interfaces, distributing load across both 100G links and achieving near line-rate performance.

**Key observation:** With balance-xor bonding, single streams are limited to ~34 Gbps (one link), but parallel streams can achieve ~93 Gbps total throughput. This matches RDMA single-link performance but requires application-level parallelism.

### Making Configuration Persistent (Optional)

For Ubuntu systems, create `/etc/netplan/60-roce-bond.yaml`:

```yaml
network:
  version: 2
  renderer: networkd

  ethernets:
    enp1s0f0np0:
      dhcp4: false
    enp1s0f1np1:
      dhcp4: false

  bonds:
    bond0:
      interfaces:
        - enp1s0f0np0
        - enp1s0f1np1
      addresses:
        - 192.168.100.10/24
      mtu: 9000
      parameters:
        mode: balance-xor
        mii-monitor-interval: 100
        transmit-hash-policy: layer3+4
```

Apply with `sudo netplan apply`.

---

## Remove Bond Before NIXL Testing

RDMA memory registration fails when network interfaces are enslaved to a bond. The verbs API requires direct access to the physical device, but bonded interfaces associate with `bond0` instead of the underlying hardware.

**Symptoms when bond is active:**
- `ibv_reg_mr` failures during memory registration
- NIXL `register_memory()` returns empty descriptors or raises exceptions
- `show_gids` shows GID entries pointing to `bond0` instead of `rocep1s0f0`/`rocep1s0f1`

**Why this happens:** When interfaces join a bond, the kernel reassigns their identity. RDMA operations that worked on `enp1s0f0np0` now fail because the GID table references `bond0`, which has no RDMA capability.

### Remove Bond on Both Nodes

```bash
# Check if bond exists
cat /proc/net/bonding/bond0 2>/dev/null

# Remove bond (run on each node)
sudo ip link set bond0 down
sudo ip link set enp1s0f0np0 nomaster
sudo ip link set enp1s0f1np1 nomaster
sudo ip link delete bond0

# Bring interfaces back up
sudo ip link set enp1s0f0np0 up
sudo ip link set enp1s0f1np1 up

# Restore IP addresses (adjust for each node)
# On spark-01:
sudo ip addr add 192.168.100.10/24 dev enp1s0f0np0

# On spark-02:
sudo ip addr add 192.168.100.11/24 dev enp1s0f0np0
```

### Verify RDMA Devices Are Accessible

```bash
# Should show both RoCE devices as Up
ibdev2netdev

# Expected output:
# rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
# rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)

# Verify GIDs point to physical interfaces (not bond0)
show_gids | grep -E "rocep1s0f0|rocep1s0f1"
```

Once the bond is removed and interfaces are up with IP addresses, NIXL memory registration will succeed.

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

```bash
pip install nixl[cu12]
python -c "from nixl._api import nixl_agent; print('NIXL installed')"
```

**UCX with CUDA Support (Required for GPU memory transfers):**

The default UCX installation may not include CUDA support. To enable GPU memory registration:

```bash
# Install build dependencies
sudo apt-get install -y autoconf automake libtool m4 pkg-config \
    build-essential libibverbs-dev librdmacm-dev libnuma-dev

# Clone and build UCX with CUDA
git clone https://github.com/openucx/ucx.git
cd ucx
./autogen.sh
./configure \
    --prefix=/usr/local/ucx \
    --with-cuda=/usr/local/cuda \
    --with-verbs \
    --with-rdmacm \
    --enable-mt
make -j$(nproc)
sudo make install
sudo ldconfig

# Update environment
export LD_LIBRARY_PATH=/usr/local/ucx/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/ucx/bin:$PATH

# Verify CUDA support
ucx_info -d | grep -i cuda
```

Without CUDA-enabled UCX, NIXL will fall back to CPU memory, which still works for the tutorial but limits production GPU-to-GPU transfer performance.

**DGX Spark limitation:** GPUDirect RDMA is not supported on DGX Spark. The platform uses a unified memory architecture where GPU-allocated pinned memory is not coherently accessible to the CPU complex or PCIe devices. As a result, `nvidia-peermem`, dma-buf, and GDRCopy do not work on this platform, and RDMA registration of CUDA buffers fails. UCX provides GPU transfers via `cuda_copy` and `cuda_ipc`, but those paths stage through host memory. RDMA-style transfers require host-pinned buffers allocated with `cudaHostAlloc` and registered with verbs (`ibv_reg_mr`).

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

### Target Node (holds source data)

```python
import torch
from nixl._api import nixl_agent, nixl_agent_config

config = nixl_agent_config(
    enable_progress_thread=True,
    enable_tcp_meta_server=True,
    tcp_meta_server_port=5555
)

agent = nixl_agent("target", config)

# Allocate and register GPU memory
torch.set_default_device("cuda:0")
tensor = torch.ones((4096, 4096), dtype=torch.float32)  # 64 MB
agent.register_memory(tensor)

# Export descriptors for remote access
target_descs = agent.get_xfer_descs([tensor])
desc_str = agent.get_serialized_descs(target_descs)

# Wait for initiator, send descriptors
while not agent.check_remote_metadata("initiator"):
    pass
agent.send_notif("initiator", desc_str)

# Wait for transfer completion
while True:
    notifs = agent.get_new_notifs()
    if "initiator" in notifs and b"done" in notifs["initiator"]:
        break
```

### Initiator Node (reads from target)

```python
import torch
import time
from nixl._api import nixl_agent, nixl_agent_config

config = nixl_agent_config(
    enable_progress_thread=True,
    enable_tcp_meta_server=True,
    tcp_meta_server_port=0
)

agent = nixl_agent("initiator", config)

# Allocate local buffer
torch.set_default_device("cuda:0")
local_tensor = torch.zeros((4096, 4096), dtype=torch.float32)
agent.register_memory(local_tensor)

# Connect to target
agent.fetch_remote_metadata("target", "192.168.100.10", 5555)
agent.send_local_metadata("192.168.100.10", 5555)

# Get remote descriptors
notifs = agent.get_new_notifs()
while len(notifs) == 0:
    notifs = agent.get_new_notifs()

remote_descs = agent.deserialize_descs(notifs["target"][0])
local_descs = agent.get_xfer_descs([local_tensor])

while not agent.check_remote_metadata("target"):
    pass

# Measure transfer time
start = time.perf_counter()

xfer_handle = agent.initialize_xfer(
    "READ", local_descs, remote_descs, "target", "done"
)
agent.transfer(xfer_handle)

while agent.check_xfer_state(xfer_handle) != "DONE":
    pass

elapsed = time.perf_counter() - start
size_mb = local_tensor.numel() * 4 / 1e6
throughput_gbps = (size_mb * 8) / (elapsed * 1000)

print(f"Transfer: {size_mb:.1f} MB in {elapsed*1000:.2f} ms")
print(f"Throughput: {throughput_gbps:.1f} Gbps")

# Verify
expected = torch.ones((4096, 4096), dtype=torch.float32, device="cuda:0")
assert torch.allclose(local_tensor, expected)
```

### Multi-Rail Configuration

NIXL uses UCX (Unified Communication X) for transport. Configure UCX to use both devices:

```bash
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1
export UCX_TLS=rc_verbs,rc_mlx5
```

For large transfers, UCX automatically stripes data across both ports.

### Latency Measurement vs Throughput Scripts

The dual-rail throughput scripts measure bulk transfer rates with single multi-gigabyte transfers. This methodology is unsuitable for latency measurement:

- Single large transfer timing reports aggregate bandwidth, not per-transfer latency
- Large transfers use rendezvous and pipelining, measuring sustained throughput rather than one-way latency
- Latency measurement requires thousands of small transfers with per-iteration timing and percentile statistics

---

## Performance Comparison

**Measured results from DGX Spark testing:**

| Configuration | Throughput | Latency | Notes |
|---------------|------------|---------|-------|
| RDMA single link (`ib_write_bw`) | 93.4 Gbps | 1-2 μs | Kernel bypass, near line rate |
| TCP single stream (bonded) | 33.7 Gbps | 50-200 μs | One link due to hash policy |
| TCP 4 parallel streams (bonded) | 93.0 Gbps | 50-200 μs | Distributed across both links |
| NIXL single rail (CPU memory) | 81.8 Gbps | 17.4 μs (avg) | Python-based latency test, 4 KB CPU buffers |
| NIXL dual rail (CPU memory) | 93.4 Gbps | 58.6 μs (avg) | Python-based latency test, 4 KB CPU buffers |

**Key findings:**

1. **RDMA achieves 2.8x the throughput** of single-stream TCP over the same link (93.4 vs 33.7 Gbps)
2. **TCP bonding with parallel streams matches RDMA single-link performance** (93 Gbps), but requires application-level parallelism
3. **Single TCP streams are limited to ~34 Gbps** despite 200 Gbps aggregate capacity
4. **NIXL dual-rail achieves 93.4 Gbps** for point-to-point transfers with CPU memory, matching RDMA single-link performance
5. **NIXL single-rail achieves 81.8 Gbps**, slightly lower than dual-rail but with significantly better latency (17.4 vs 58.6 μs average)

The throughput difference comes from the data path. RDMA bypasses kernel TCP/IP stack overhead (system calls, buffer copies, interrupt handling, context switches), achieving both higher bandwidth and 50-100x lower latency.

**Measured NIXL latency (CPU, 4 KB, 1000 iterations):** Avg 58.6 μs, P50 11.1 μs, P95 166.6 μs. These numbers include Python and user-space scheduling overhead and should be used for relative comparison only.

**Single-rail latency (CPU, 4 KB, 1000 iterations):** Avg 17.4 μs, P50 16.2 μs, P95 20.8 μs. Single-rail and dual-rail latency are not interchangeable because transport lanes and wireup paths differ.

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

**Measured latency characteristics:**
- TCP: 50-200 μs per transfer
- RDMA single-link: 1-2 μs
- NIXL dual-rail (Python, 4 KB): 58.6 μs average
- NIXL single-rail (Python, 4 KB): 17.4 μs average

For tensor parallelism across two Spark systems, bonding provides no benefit—NCCL uses RDMA directly. Point-to-point patterns (KV-cache movement, tensor shard transfers) achieve full RDMA performance with NIXL.

---

## Troubleshooting

### UCX Version Mismatch Between Nodes

**Symptom:** NIXL connection fails with `NIXL_ERR_NOT_FOUND` even when both nodes have NIXL installed.

**Cause:** UCX requires matching versions on both endpoints for RDMA connections. A node with UCX 1.16.0 (system) cannot connect to a node with UCX 1.21.0 (custom build).

**Diagnosis:**
```bash
# Check UCX version on each node
ucx_info -v
# Look for the version line, e.g., "# UCX version=1.16.0"

# Check which library is loaded
ldd $(which ucx_info) | grep libucs
```

**Fix:** Build matching UCX versions on both nodes:
```bash
cd ~
git clone https://github.com/openucx/ucx.git
cd ucx
git checkout v1.21.0  # Use same tag on both nodes
./autogen.sh
./configure --prefix=/usr/local/ucx --with-cuda=/usr/local/cuda --with-verbs --with-rdmacm --enable-mt
make -j$(nproc)
sudo make install
export LD_LIBRARY_PATH=/usr/local/ucx/lib:$LD_LIBRARY_PATH
```

### P2P Interface IP Conflict

**Symptom:** UCX connects but uses wrong interface, or connection times out despite correct IP addresses.

**Cause:** DGX Spark has a P2P interface (`enP2p1s0f0np0`) that may have an IP address in the same subnet as the RoCE interfaces. UCX picks the first matching interface, which may be incorrect.

**Diagnosis:**
```bash
# Check all interfaces for IPs in your subnet
ip addr | grep "192.168.100"

# Example problematic output:
#   inet 192.168.100.11/24 brd 192.168.100.255 scope global enp1s0f0np0
#   inet 192.168.100.15/24 brd 192.168.100.255 scope global enP2p1s0f0np0  <-- conflict!
```

**Fix:**
```bash
# Remove the conflicting IP
sudo ip addr del 192.168.100.15/24 dev enP2p1s0f0np0

# Or restrict UCX to the correct interface (add to scripts BEFORE imports):
import os
os.environ["UCX_NET_DEVICES"] = "rocep1s0f0:1"
```

### Interface Down

**Symptom:** `ibdev2netdev` shows one or more interfaces as `(Down)`.

**Diagnosis:**
```bash
ibdev2netdev
# Expected: rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
# Problem:  rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
```

**Fix:**
```bash
sudo ip link set enp1s0f1np1 up
```

### NIXL GPU Registration Fails

**Error:** `ibv_reg_mr failed: Bad address`

**Cause:** GPU memory registration over RDMA requires GPUDirect RDMA kernel modules. On DGX Spark, the `nvidia-peermem` module may fail to load.

**Diagnosis:**
```bash
# Check if nvidia-peermem is loaded
lsmod | grep nvidia_peermem

# Try loading it
sudo modprobe nvidia-peermem
# May fail with: modprobe: ERROR: could not insert 'nvidia_peermem': Invalid argument
```

**Workaround:** CPU memory fallback:
```python
try:
    tensor = torch.ones((4096, 4096), dtype=torch.float32, device="cuda:0")
    agent.register_memory(tensor)
except Exception as e:
    print(f"GPU registration failed: {e}")
    tensor = torch.ones((4096, 4096), dtype=torch.float32, device="cpu")
    tensor = tensor.pin_memory()
    agent.register_memory(tensor)
```

UCX `cuda_copy` transport stages GPU data through host memory. Performance is lower than native GPUDirect RDMA, though specific benchmarks were not performed on DGX Spark due to GPU registration failures.

### TCP Throughput Near Zero

**Symptom:** `iperf3` shows very low throughput despite successful connection.

**Cause:** MTU mismatch causes TCP congestion control to throttle.

**Diagnosis:**
```bash
# Check MTU on both endpoints
cat /sys/class/net/bond0/mtu  # Should be 9000
cat /sys/class/net/enp1s0f0np0/mtu

# Look for Cwnd stuck at ~1.4 KB in iperf3 output
```

**Fix:**
```bash
sudo ip link set enp1s0f0np0 mtu 9000
sudo ip link set enp1s0f1np1 mtu 9000
sudo ip link set bond0 mtu 9000
```

### NIXL Does Not Use Both Rails

```bash
# Check UCX device detection
ucx_info -d | grep mlx5

# Explicitly set devices
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1
export UCX_TLS=rc_verbs,rc_mlx5
```

### Bond Not Forming

```bash
# Check if bonding module is loaded
lsmod | grep bonding
sudo modprobe bonding

# Verify bond status
cat /proc/net/bonding/bond0

# Check for errors
dmesg | grep -i bond
```

### Transfer Falls Back to TCP

If NIXL debug output shows socket-based transport instead of RDMA:
1. Verify RDMA devices are active: `ibstat`
2. Check UCX installation includes RDMA: `ucx_info -v | grep verbs`
3. Ensure `libibverbs` is installed: `dpkg -l | grep libibverbs`

---

## Summary

Measured performance on DGX Spark with dual 100G RoCE:

- NIXL single-rail RDMA: 81.8 Gbps (CPU memory)
- NIXL dual-rail RDMA: 93.4 Gbps (CPU memory)
- TCP bonding with 4 parallel streams: 93.0 Gbps
- TCP single stream: 33.7 Gbps
- Latency: RDMA 1-2 μs (hardware), 17.4 μs (NIXL single-rail), 58.6 μs (NIXL dual-rail) vs TCP 50-200 μs

Bonding and RDMA coexist without conflict. Bonding serves management networks, IP-based storage, and non-RDMA services. RDMA (via NIXL or NCCL) provides direct hardware access for point-to-point and collective operations.

Collective operations (gradient sync, tensor parallelism) using NCCL are covered in the [first tutorial](01_InfiniBand_Tutorial.ipynb).

---

## References

- [NIXL GitHub Repository](https://github.com/ai-dynamo/nixl)
- [Linux Kernel Bonding Documentation](https://www.kernel.org/doc/Documentation/networking/bonding.txt)
- [UCX Documentation](https://openucx.readthedocs.io/)
- [NCCL Tutorial](01_InfiniBand_Tutorial.ipynb)
