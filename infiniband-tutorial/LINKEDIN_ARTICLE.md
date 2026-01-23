# RoCE vs InfiniBand: Benchmarking Two DGX Sparks

Two DGX Spark boxes on the bench. Connected with what looked like InfiniBand cables. Ran the tests everyone talks about but few actually show. Plot twist: it's actually RoCE. Here's what came out—and why that distinction matters less than you'd think.

---

## First, What's the Deal with RDMA?

Here's the thing about regular networking—Ethernet, WiFi, whatever—the CPU handles every single packet. Data comes in, CPU gets interrupted, CPU copies it to memory, CPU tells your application about it. Do this a few thousand times a second and your CPU is spending more time playing mailman than doing actual work.

InfiniBand flips this around. The network card writes directly into your application's memory. No CPU involved. They call this RDMA—Remote Direct Memory Access. Sounds like marketing speak until you see the latency numbers: 1-2 microseconds versus 50-200 for Ethernet. That's not 2x faster. That's 50-100x.

For inference at scale, this matters more than you'd think. Multi-GPU serving means GPUs constantly passing KV-caches, attention states, and intermediate activations between each other. Every request, multiple times. Every microsecond the network adds shows up directly in your latency metrics.

## Plot Twist: DGX Spark Uses RoCE, Not InfiniBand

Here's something confusing at first. Running `ibv_devinfo` showed device names like `roceP2p1s0f0`. That `roce` prefix is the giveaway—this is **RoCE (RDMA over Converged Ethernet)**, not native InfiniBand.

Same RDMA benefits (zero-copy, kernel-bypass, microsecond latency), but running over Ethernet hardware instead of an InfiniBand fabric. The ConnectX adapters support both modes.

**Why does this matter?**

Honestly, for most workloads—it doesn't. The `ib_write_bw` and NCCL tests work identically. The performance is the same. NCCL doesn't care whether the underlying transport is InfiniBand or RoCE; it just sees RDMA.

**Where it does matter:**

- **Switching**: InfiniBand requires InfiniBand switches. RoCE uses standard Ethernet switches you probably already have.
- **Configuration**: InfiniBand has lossless flow control built in. RoCE needs Priority Flow Control (PFC) and ECN configured on your switches. Get this wrong, and performance tanks.
- **Scale**: Native InfiniBand fabrics can be more predictable at massive scale. RoCE is fine for rack-scale deployments.

Cloud providers typically offer native InfiniBand on the serious GPU instances (H100/B200 clusters) and RoCE on everything else. DGX Spark falls into the RoCE category—which is totally fine for most inference workloads.

---

## The Setup

Two DGX Sparks, one cable between them. No switch, no special configuration. Just a direct connection.

One thing worth noting—the interfaces don't show up as `ib0` like the old documentation says. Modern kernels use predictable naming, so they appeared as `enp1s0f0np0` and `enp1s0f1np1`. And the device names in `ibv_devinfo` showed `roceP2p1s0f0`—confirming this is RoCE, not native InfiniBand. Same tools work though.

---

## What the Numbers Actually Look Like

**Test 1: Raw RDMA bandwidth**

```bash
# One machine runs the server
ib_write_bw

# Other machine connects to it  
ib_write_bw 192.168.100.11
```

The output shows `BW average[MB/sec]`. Results showed around 12,000 MB/sec, which works out to roughly 96 Gbps. (Divide by 125 to convert—benchmarks use Bytes, specs use bits.)

**Quick note on RDMA test types:** There's `ib_write_bw`, `ib_send_bw`, and `ib_read_bw`. The difference is how RDMA moves the data. Write is one-sided—the sender writes directly to the receiver's memory without the receiver's CPU knowing. Send is two-sided—the receiver has to post receive buffers first and gets notified when data arrives. For validating that InfiniBand is working, `ib_write_bw` is the standard choice. It shows raw hardware capability without any protocol overhead. Both should hit near line-rate anyway.

**Test 2: TCP over the same link**

Running iperf3 over the same link showed 35 Gbps.

Wait, what? Same cable, same hardware, but a third of the speed?

This is actually correct. iperf3 uses TCP/IP, which means it goes through the kernel's networking stack, then through an IPoIB (IP-over-InfiniBand) translation layer, then finally to the hardware. All that overhead costs 60-70% of your bandwidth.

The important part: inference frameworks like vLLM, TensorRT-LLM, and TGI use NCCL for GPU communication, and NCCL uses native RDMA. Actual serving workloads see the full 96 Gbps, not the 35.

**Test 3: Latency**

```bash
ib_write_lat  # server
ib_write_lat 192.168.100.11  # client
```

Look for `t_avg` in the output. Should be 1-2 microseconds. Ethernet ping on the same network? 50-200 microseconds.

**Test 4: Adding a second cable**

Starting with one cable between ports, DGX Spark has two RoCE ports, so plugging in a second cable doubles the available bandwidth.

Each link still runs at ~12,000 MB/sec individually. But now workloads can use both simultaneously—aggregate bandwidth doubles to ~24,000 MB/sec. Latency stays the same; individual messages aren't faster, just more parallel traffic is possible.

NCCL figures this out automatically. No code changes needed.

**Test 5: NCCL All-Reduce (the real test)**

All the tests above measure raw network performance. But inference frameworks don't talk to the network directly—they use NCCL (NVIDIA Collective Communications Library). Testing NCCL directly shows what actual workloads will achieve.

```bash
# Multi-node NCCL test with mpirun
mpirun -np 4 --host node1:2,node2:2 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_DISABLE=0 \
    all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

**What to look for:**

First, check the NCCL_DEBUG output. The line you want to see:
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB
```

That `NET/IB` means NCCL found the RDMA interface and is using it. If you see `NET/Socket` instead, it's falling back to TCP—something's wrong.

Second, look at the `busbw` column in the results. For 128MB messages over RDMA, expect 20-40 GB/s. Stuck below 5 GB/s even for large messages? NCCL isn't using the RDMA link.

The confusing part about these tests: small messages show low bandwidth. That's normal—they're latency-bound. The bandwidth numbers only matter for messages above 1MB or so.

---

## Why This Matters for LLM Inference

**Tensor parallelism** - Large models like Llama 70B or Mixtral don't fit on one GPU. Split them across 4 or 8 GPUs, and suddenly every forward pass requires all-reduce operations. At 1-2 microseconds per hop, the network is invisible. At 200 microseconds, milliseconds get added to every token generated.

**KV-cache movement** - Long context windows mean large KV-caches. Continuous batching means these caches need to move around as requests get scheduled. Slow interconnect = cache transfer becomes the bottleneck.

**Disaggregated serving** - Separating prefill (compute-heavy) from decode (memory-bound) across different node pools is becoming standard practice. The KV-cache has to move between them after prefill completes. With RDMA, this transfer takes microseconds. With Ethernet, tens of milliseconds get added before the first token even starts generating.

**Time-to-first-token (TTFT)** - Users notice latency. Every hop through the serving stack adds up. RDMA makes inter-GPU communication essentially free from a latency perspective.

---

## Common Gotchas

**iperf3 wouldn't start** - "Address already in use." The default port was taken. Adding `-p 5202` fixes it.

**Interface names were wrong** - Looking for `ib0` when modern systems name them differently. `ibstat` shows the actual port status regardless of what the interface is called.

**iperf3 results looked broken** - 35 Gbps on a 100G link felt wrong. It's not. Different protocol, different overhead. The RDMA tests show true hardware capability.

---

## Commands Worth Remembering

`ibstat` - Shows port status. Look for "State: Active"

`ib_write_bw` / `ib_write_lat` - Native RDMA tests. These show real InfiniBand performance.

`iperf3` - TCP bandwidth test. Useful for comparison but doesn't reflect what NCCL will achieve.

---

## Bottom Line

Running `ib_write_bw` showed 12,000 MB/sec sustained. The same test over regular TCP: lucky to hit 1,200 MB/sec. That's not a small difference. That's the difference between the interconnect being a bottleneck and being invisible.

For LLM inference, the latency gap is what kills you. 1-2 microseconds versus 50-200 microseconds. Running a 70B model across 8 GPUs with tensor parallelism means every token generation involves multiple all-reduce operations. That latency multiplies fast.

Whether it's native InfiniBand or RoCE (like on DGX Spark), the RDMA magic is what matters. Building inference infrastructure and wondering whether RDMA networking is worth it? Run these tests. The numbers speak for themselves.

---

#InfiniBand #RoCE #RDMA #LLMInference #AIInfrastructure #NVIDIA #DGX
