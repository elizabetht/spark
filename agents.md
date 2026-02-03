# AGENTS.md — Guidelines for AI Assistants

This document provides instructions for AI agents working on this repository. The goal is to maintain a professional, technical tone and ensure content is accessible to the target audience.

> **Reference**: See [README.md](README.md) for architecture diagrams, learning modules, and hardware setup details.

---

## Core Philosophy

> Build understanding progressively. No magic abstractions. Understand the "why" before the "how".

When helping with this project:
1. **Map ML concepts to systems concepts** (e.g., "KV cache is like a session cache in web servers")
2. **Start with working minimal code**, then add complexity incrementally
3. **Explain distributed systems aspects deeply**, this is the target audience's strength
4. **Demystify ML jargon** with plain systems engineering language
5. **Establish baselines before showing optimizations**, you cannot appreciate "after" without "before"

---

## Baseline Comparison Pedagogy

> You cannot appreciate an optimization without seeing what you're optimizing from.

### Why Baselines Matter

This is how systems engineers evaluate any infrastructure change:
- "We added RDMA → 10x faster than TCP"
- "We used NIXL multi-rail → 176 Gbps vs 96 Gbps single-rail"
- "We implemented KV-aware routing → 2x cache hit rate"

### Making the Baseline Fair

To ensure honest comparison, the baseline should be the *best you can do* without the optimization:
- Use proper MTU settings (not default 1500)
- Use appropriate GID index for RoCE (not link-local)
- Tune buffer sizes appropriately

This way, when the optimization wins, it's because of **architectural advantages**, not because the baseline was poorly configured.

---

## Project Overview

**DGX Spark Network Benchmarks** documents RDMA (Remote Direct Memory Access) performance testing between two direct-connected DGX Spark systems using RoCE (RDMA over Converged Ethernet).

### Content Focus
- RDMA vs TCP/IP performance comparison
- RoCE vs native InfiniBand clarification
- NIXL multi-rail benchmarks
- Practical implications for LLM inference

### Project Structure
```
spark/
├── infiniband-tutorial/
│   ├── 01_InfiniBand_Tutorial.ipynb  # RDMA basics, single-link benchmarks
│   ├── 02_Multi_Rail_Tutorial.ipynb  # Bonding vs NIXL comparison
│   └── README.md
├── microk8s-cluster-setup/
│   └── 01_MicroK8s_Cluster_Setup.ipynb
└── agents.md              # This file (AI guidelines)
```

---

## Learner Profile

**Background**: Systems engineering (not ML expert)

**Existing Knowledge**:
- LLM serving basics
- KV cache fundamentals
- Strong in: distributed systems, networking, infrastructure

**Learning Style**: Build progressively, understand each component before moving to the next.

---

## Target Audience

### Who We're Writing For

**Hiring managers, technical leadership, and software engineers with infrastructure/backend backgrounds.**

They are:
- Experienced with systems, networking, and distributed infrastructure
- Familiar with: load balancers, caching, databases, APIs, Kubernetes
- Evaluating candidates or making technology investment decisions
- Technical leaders at LLM inference companies (TensorRT-LLM, vLLM, etc.)
- People who recognize marketing fluff instantly

### What They Care About
- Does this actually work?
- What are the real numbers?
- What tradeoffs should I know about?
- Can this person execute?

### What They Don't Care About
- Buzzwords and superlatives
- Vague promises
- Over-explained basics they already know

---

## Concept Mappings

When explaining RDMA and networking components, use these systems engineering equivalents:

| Concept | Systems Engineering Equivalent |
|---------|--------------------------------|
| RDMA | DMA but across network (NIC writes directly to remote memory) |
| RoCE | RDMA running over standard Ethernet instead of InfiniBand fabric |
| Verbs API | Low-level interface to RDMA hardware (like raw sockets for TCP) |
| Queue Pair (QP) | Bidirectional communication channel (like a TCP connection) |
| GID | Global Identifier, like an IP address for RDMA endpoints |
| MTU | Maximum payload per packet (jumbo frames = 9000 bytes) |
| NIXL | NVIDIA's library for point-to-point RDMA transfers |
| UCX | Unified Communication X, transport abstraction layer |
| NCCL | Collective operations library (all-reduce, all-gather) |
| KV Cache | In-memory cache per request (like Redis session storage) |
| Prefill Phase | Request parsing + cache warmup (CPU/memory intensive) |
| Decode Phase | Response generation, streaming (latency-sensitive) |

---

## Writing Style Guidelines

### 1. Lead with Data, Not Claims

**Bad:**
> InfiniBand offers incredible performance improvements that will revolutionize your infrastructure.

**Good:**
> `ib_write_bw` showed 12,000 MB/sec (~96 Gbps). iperf3 over the same link hit 35 Gbps. The difference? RDMA bypasses the kernel.

### 2. Explain the "Why" Once, Then Move On

Readers at this level don't need repeated explanations. State the concept, give a concrete example, continue.

**Bad:**
> RDMA is very important because it allows direct memory access. This direct memory access is significant because it bypasses the CPU. Bypassing the CPU matters because...

**Good:**
> RDMA bypasses the CPU entirely—one machine writes directly to another's memory. That's why `ib_write_bw` hits 96 Gbps while TCP-based `iperf3` caps at 35 Gbps over the same hardware.

### 3. Acknowledge Complexity and Tradeoffs

Real practitioners know nothing is perfect. Acknowledging limitations builds credibility.

**Include:**
- When something didn't work as expected
- Configuration gotchas you hit
- Situations where the technology isn't the right choice

**Example:**
> If you get ~35 Gbps on iperf3 and expect 100+, that's normal. iperf3 uses TCP/IP over IPoIB—kernel overhead limits it. Your actual ML workloads use NCCL with native RDMA, which hits the full link speed.

### 4. Use Specific Examples from Actual Work

Generic descriptions read as theoretical. Specific details signal hands-on experience.

**Generic:**
> InfiniBand interfaces can have different naming conventions depending on your system configuration.

**Specific:**
> My interfaces showed up as `enp1s0f0np0` and `enp1s0f1np1`—predictable naming, not the traditional `ib0`/`ib1`. The `np` suffix indicates InfiniBand network ports.

### 5. Structure for Scanning

Senior people scan first, read second. Make it easy.

**Use:**
- Tables for comparisons
- Code blocks for commands
- Short paragraphs (3-4 sentences max)
- Headers that summarize the section

**Avoid:**
- Long unbroken paragraphs
- Burying key information mid-paragraph
- Headers that require reading to understand

### 6. Skip the Preamble

Don't spend three paragraphs explaining why the topic matters. They clicked on the article—they already care.

**Bad opening:**
> In today's rapidly evolving AI landscape, infrastructure decisions have never been more critical. As organizations race to deploy large language models...

**Good opening:**
> I connected two DGX Spark boxes with InfiniBand and measured what actually happens. Here's the data.

### 7. Be Direct About What You Don't Know

Uncertainty stated clearly is more credible than false confidence.

**Good:**
> I haven't tested this with NCCL directly—these are raw fabric benchmarks. The actual distributed training numbers would require a separate test with real model training.

---

## Formatting Conventions

### Numbers and Units

- Always include units: `12,000 MB/sec`, not `12000`
- Show conversions when relevant: `12,000 MB/sec (~96 Gbps)`
- Use consistent precision: don't mix `96.4 Gbps` and `~100 Gbps`

### Commands and Code

- Use code blocks for anything someone might copy
- Include the actual command you ran, not a simplified version
- Note any flags or options that matter

```bash
# What you actually type
iperf3 -c 192.168.100.11 -p 5202 -t 10

# Not this
iperf3 -c <server_ip>
```

### Tables vs Prose

Use tables when comparing more than 2 items on more than 2 dimensions. Otherwise, prose is fine.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Fix |
|--------------|--------------|-----|
| "Revolutionary/game-changing" | Meaningless superlatives | State the measured improvement |
| "In this article, we will..." | Wastes reader time | Just start |
| "As we all know..." | Condescending | Delete entirely |
| "It's important to note that..." | Filler | State the fact directly |
| Excessive emoji | Unprofessional for this audience | Use sparingly or not at all |
| Rhetorical questions | Often feel manipulative | Make statements instead |
| "Simple" or "just" | Dismissive of complexity | Describe what actually happens |

---

## Tone Guidelines

**Voice:** Professional and technical. Write for engineers and technical leadership.

**Do:**
- Use clear, precise technical language
- State facts directly without hedging
- Let data and measurements speak for themselves
- Write complete sentences with proper structure
- Maintain a formal but readable tone

**Avoid:**
- Colloquialisms and casual expressions ("Nah", "Yeah", "cool", "gotcha")
- Rhetorical questions as section headers ("Wait, What's RDMA Again?")
- Exclamation marks and dramatic phrasing
- Profanity or crude language
- Emdashes (use colons, periods, or commas instead)
- Emoji in technical content
- Filler phrases ("Turns out", "The fun part", "Ouch")
- Sentence fragments for effect

**Example transformation:**

Before (casual):
> Wait, what? Same cable. Same hardware. A third of the speed?! Turns out this is totally normal.

After (professional):
> This is expected behavior. iperf3 uses TCP/IP, which traverses the kernel networking stack and IPoIB translation layer. The overhead consumes 60-70% of available bandwidth.

**The goal:** Sound like a technical whitepaper or engineering blog from a respected company. Clear, precise, informative.

---

## Acronym Usage

**Expand acronyms on first use.** The first time an acronym appears, include the full term in parentheses.

**Examples:**
- RDMA (Remote Direct Memory Access)
- RoCE (RDMA over Converged Ethernet)
- GPU (Graphics Processing Unit)
- NCCL (NVIDIA Collective Communications Library)
- TCP/IP (Transmission Control Protocol/Internet Protocol)

After the first expansion, use the acronym alone. This respects readers who know the terms while ensuring clarity for those encountering them for the first time.

---

## Checklist Before Publishing

- [ ] Does the opening sentence contain actual information?
- [ ] Are all claims backed by specific data or commands?
- [ ] Did I acknowledge at least one limitation or gotcha?
- [ ] Can someone skim the headers and get the main points?
- [ ] Would I share this with a skeptical colleague?
- [ ] Did I cut every sentence that's just "filling space"?
- [ ] Is the language clean and professional throughout?
- [ ] Are acronyms expanded on first use?

---

## Example Transformation

**Before (AI-sounding):**
> InfiniBand technology represents a paradigm shift in high-performance computing infrastructure. By leveraging RDMA capabilities, organizations can achieve unprecedented levels of performance that were previously thought impossible. This article will walk you through the transformative journey of implementing InfiniBand in your AI infrastructure.

**After (Human-sounding):**
> I ran `ib_write_bw` between two DGX Spark nodes and got 12,000 MB/sec sustained. Same test over Ethernet: 1,200 MB/sec. That 10x difference is RDMA—the NIC writes directly to remote memory without touching the CPU.

The second version:
- Opens with data
- Uses specific tools and numbers
- Explains the mechanism in one sentence
- Doesn't promise or hype—just states what happened
