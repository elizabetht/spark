# AI Agent Instructions for Disaggregated Serving Notebooks

> **Reference**: See [README.md](README.md) for module structure and learning path.

## Learner Profile

**Background**: Systems engineering (not ML expert)
**Existing Knowledge**:
- LLM serving basics
- KV cache fundamentals
- Strong in: distributed systems, networking, infrastructure

**Learning Style**: Build progressively, understand each component before moving to the next.

---

## Core Principles

1. **Every notebook runs real code on real hardware.** No simulated benchmarks, no `time.sleep()`, no hardcoded latency numbers pretending to be measurements.
2. **Use vLLM's built-in disaggregated serving** with `NixlConnector`. Do not build custom TCP+pickle transfer code.
3. **Arithmetic over model loading** when possible. KV cache dimensions are model architecture constants. Loading 16 GB of weights to read config values wastes time and GPU memory.
4. **Baseline before optimization.** Notebook 01 establishes single-node performance. Notebook 03 must compare against it.

---

## Module Layout (4 Notebooks)

| # | Notebook | Purpose |
|---|----------|---------|
| 00 | Environment Setup | Hardware/network/software verification |
| 01 | Local Inference Baseline | Single-node vLLM performance (the bar to beat) |
| 02 | Understanding KV Cache | Cache structure, size math, transfer cost analysis |
| 03 | Disaggregated Serving | Split prefill/decode across spark-01 and spark-02 |

### Why 04-06 Were Removed

The original notebooks 04 (NIXL Integration), 05 (KV-Aware Routing), and 06 (Full Dynamo Integration) were entirely simulated. They used `time.sleep()`, hardcoded latency values, and classes like `RDMAKVTransfer` and `DynamoOrchestrator` that did not perform actual RDMA or orchestration. This violates the core principle: every notebook must run real code.

NIXL integration is handled by vLLM's `NixlConnector` in Notebook 03. KV-aware routing and full orchestration belong in the `ai-dynamo/` directory, which will use NVIDIA Dynamo's actual framework.

---

## Concept Mappings

| Concept | Systems Engineering Equivalent |
|---------|-------------------------------|
| KV Cache | In-memory session state (like Redis per-request cache) |
| Prefill Phase | Request parsing + cache warmup (compute-bound) |
| Decode Phase | Response streaming (memory-bound, latency-sensitive) |
| NixlConnector | vLLM's RDMA transport for GPU-to-GPU KV cache transfer |
| Disaggregated Serving | Splitting web tier (prefill) from API tier (decode) |
| Proxy Server | Reverse proxy routing requests to prefill or decode |

---

## What Does NOT Belong

- Simulated benchmarks (all measurements must come from real execution)
- Manual KV cache serialization (TCP + pickle). vLLM handles this via NixlConnector.
- Loading the full model just to inspect config values. Use arithmetic.
- Notebooks that require `ai-dynamo` components. Those go in `ai-dynamo/`.
- Hardcoded performance numbers presented as if they were measured

---

## Hardware Configuration

- **spark-01**: 192.168.100.10, 1x NVIDIA GB10 GPU, RoCE interfaces
- **spark-02**: 192.168.100.11, 1x NVIDIA GB10 GPU, RoCE interfaces
- **Network**: RDMA over Converged Ethernet (RoCE), direct-connected
- **Model**: meta-llama/Llama-3.1-8B-Instruct (cached on both nodes)

---

## Notebook Cell Structure

Each notebook follows this pattern:

1. **Title + Learning Objectives** (markdown)
2. **Prerequisite Check** (code: load config, verify environment)
3. **Concept Explanation** (markdown: with systems analogy)
4. **Working Code** (code: minimal, real execution)
5. **Measurement + Comparison** (code: compare against baseline)
6. **Key Takeaways** (markdown: what we learned, what's next)

---

## Code Style

```python
# State facts directly
print(f"KV cache per token: {per_token_mb:.4f} MB")

# Not this
print("Amazing! The KV cache is only {per_token_mb:.4f} MB per token!")
```

- No emoji in output
- No superlatives ("incredible", "revolutionary")
- Include units with all measurements
- Show the command that was actually run, not a simplified version

---

## Troubleshooting Patterns

| Symptom | Likely Cause | Debug Command |
|---------|-------------|---------------|
| vLLM OOM | Model too large for GPU memory | `nvidia-smi` during load |
| NIXL transfer fails | RDMA misconfigured | `ibstat`, `ibv_devinfo` |
| Connection refused | vLLM instance not started | `ss -tlnp \| grep <port>` |
| Proxy timeout | Prefill/decode not responding | `curl http://<ip>:<port>/health` |
| SSH fails | Key not configured | `ssh -v nvidia@<ip>` |