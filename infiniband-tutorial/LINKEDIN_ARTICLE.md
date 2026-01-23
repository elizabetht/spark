# Benchmarking Two DGX Sparks: RoCE ≠ InfiniBand

Two DGX Spark boxes on the bench. Cables that *look* like InfiniBand. Benchmarks that everyone talks about but few actually run. And a plot twist: those cables? RoCE, not InfiniBand.

Here's what the numbers look like—and why that distinction matters less than you'd think.

---

## Wait, What's RDMA Again?

Traditional networking (Ethernet, WiFi, you name it) has an inefficiency baked right in. Every packet that arrives? CPU has to stop what it's doing, copy the data, tell your app about it. Rinse and repeat thousands of times per second. The CPU becomes a glorified mail sorter.

RDMA flips that model. The network card writes directly into your app's memory. CPU never knows it happened. The latency difference is significant—1-2 microseconds vs 50-200μs for regular Ethernet. That's not a 2x improvement, that's 50-100x.

Why care? Well, multi-GPU inference means GPUs constantly shuffling KV-caches and attention states around. Every. Single. Request. Those microseconds add up fast.

## The Plot Twist Nobody Mentions

Poking around, running `ibv_devinfo`, and the devices are named `roceP2p1s0f0`. Huh. That "roce" bit is the tell—this is **RoCE (RDMA over Converged Ethernet)**, not actual InfiniBand.

Same RDMA goodness (zero-copy, kernel-bypass, crazy low latency), just running over Ethernet instead of a dedicated InfiniBand fabric. The ConnectX cards can do both.

**Does it actually matter though?**

For day-to-day stuff? Nah. The `ib_write_bw` tests work the same. NCCL doesn't care—it just sees RDMA and goes "cool."

**When it DOES matter:**

- **Switches**: InfiniBand needs InfiniBand switches (expensive). RoCE works with the Ethernet switches you already own.
- **Config headaches**: InfiniBand has lossless flow control baked in. RoCE needs PFC and ECN configured on your switches. Mess it up and performance tanks.
- **Scale**: At massive scale (think datacenter-wide), native InfiniBand is more predictable. RoCE is fine for a rack or two.

Big cloud providers give you real InfiniBand on the beefy GPU instances (H100/B200 clusters). Everything else gets RoCE. DGX Spark is in the "everything else" bucket, which is honestly fine.

---

## The Setup

Dead simple: two DGX Sparks, one cable between them. No switch. No fancy config. Direct connection.

One gotcha—the interfaces aren't called `ib0` like all the old docs say. Newer kernels use "predictable naming" so they show up as `enp1s0f0np0` and `enp1s0f1np1`. Not exactly intuitive. The `ibv_devinfo` output with `roceP2p1s0f0` confirms it's RoCE. Tools still work the same though.

---

## Actual Numbers (The Fun Part)

**Test 1: Raw RDMA bandwidth**

```bash
# Machine 1 (server)
ib_write_bw

# Machine 2 (client)
ib_write_bw 192.168.100.11
```

Result: ~12,000 MB/sec in the `BW average[MB/sec]` column. That's roughly 96 Gbps. (Divide by 125 to convert—benchmarks use Bytes, marketing uses bits, because of course they do.)

**Nerdy sidebar on RDMA tests:** There's `ib_write_bw`, `ib_send_bw`, and `ib_read_bw`. Write is one-sided—sender writes directly to receiver's memory, receiver's CPU has no clue. Send is two-sided—receiver posts buffers first, gets notified when data lands. For "is this thing working?" tests, `ib_write_bw` is what everyone uses.

**Test 2: TCP over the same cable**

Ran iperf3 on the same link. Got 35 Gbps.

Wait, what? Same cable. Same hardware. A third of the speed?!

Turns out this is totally normal. iperf3 uses TCP/IP, which means kernel networking stack → IPoIB translation layer → finally the hardware. All that overhead eats 60-70% of your bandwidth.

Good news: vLLM, TensorRT-LLM, TGI—they all use NCCL, and NCCL uses raw RDMA. Real workloads see the full 96 Gbps.

**Test 3: Latency**

```bash
ib_write_lat  # server
ib_write_lat 192.168.100.11  # client
```

Check `t_avg` in the output. Should see 1-2 microseconds. A regular ping over Ethernet? 50-200μs. Yeah.

**Test 4: Adding a second cable**

DGX Spark has two RoCE ports. Plugging in a second cable doubles the available bandwidth.

Each link still does ~12,000 MB/sec on its own. But now there's double the aggregate bandwidth (~24,000 MB/sec). Latency stays the same—individual messages aren't faster, just more lanes available.

NCCL figures this out automatically. No code changes needed.

**Test 5: NCCL All-Reduce (what actually matters)**

All the above tests the raw network. But real inference code uses NCCL. So testing that directly:

```bash
# Multi-node NCCL test
mpirun -np 4 --host node1:2,node2:2 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_DISABLE=0 \
    all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

**What you're looking for:**

In the NCCL_DEBUG output, find this line:
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB
```

`NET/IB` = good, it found RDMA. `NET/Socket` = bad, it's using TCP as fallback. Go fix something.

Then check the `busbw` column. For 128MB messages you want 20-40 GB/s. Stuck under 5 GB/s? RDMA isn't being used.

Small messages showing low bandwidth? That's normal—they're latency-bound. Bandwidth only matters for big messages (1MB+).

---

## Why This Matters (LLM Stuff)

**Tensor parallelism** - Llama 70B doesn't fit on one GPU. Split it across 4 or 8, and now every forward pass needs all-reduce ops. At 1-2μs per hop, the network is basically invisible. At 200μs? That's milliseconds added per token. Ouch.

**KV-cache shuffling** - Long contexts = big KV-caches. Continuous batching = caches moving around constantly. Slow network = bottleneck city.

**Disaggregated serving** - Separating prefill from decode on different nodes is becoming standard. KV-cache has to move between them. RDMA = microseconds. Ethernet = tens of milliseconds before the first token even starts. Users notice.

**TTFT** - Time-to-first-token is what users actually feel. Every hop adds up. RDMA makes GPU-to-GPU comms basically free from a latency perspective.

---

## Common Gotchas

**iperf3 wouldn't start** - "Address already in use." Something else was on the default port. `-p 5202` fixed it.

**Couldn't find `ib0`** - Because it doesn't exist anymore. Modern systems name interfaces differently. `ibstat` shows port status regardless of the weird interface names.

**iperf3 numbers seemed broken** - 35 Gbps on a "100G" link felt wrong. It's not. Different protocol, different overhead. The RDMA tests show what the hardware can actually do.

---

## Cheat Sheet

`ibstat` - Is this thing on? Look for "State: Active"

`ib_write_bw` / `ib_write_lat` - Raw RDMA performance tests

`iperf3` - TCP test. Useful for comparison but not what NCCL uses

---

## TL;DR

`ib_write_bw` showed 12,000 MB/sec. Same test over TCP: maybe 1,200 MB/sec on a good day. That's not a small gap—that's the difference between the network being a problem and being invisible.

For LLM inference, the latency gap hurts more. 1-2μs vs 50-200μs. Running a 70B model across 8 GPUs with tensor parallelism means every token gen does multiple all-reduces. Those microseconds stack up real quick.

Hit up the comments if you're also messing around with DGX Spark—always good to compare notes.

Whether it's native InfiniBand or RoCE (like on DGX Spark), the RDMA magic is what matters. Building inference infrastructure and wondering whether RDMA networking is worth it? Run these tests. The numbers speak for themselves.

---

#InfiniBand #RoCE #RDMA #LLMInference #AIInfrastructure #NVIDIA #DGX
