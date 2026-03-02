# GPU Performance Optimization Wisdom

Hard-won principles for GPU performance measurement and optimization.
Applies to this project (Nadrin/PBR Lux variant) and beyond.

---

## Core Principles

### 1. GPU Timers Are the Primary Source of Truth

Instruction counts, binary sizes, and static analysis are useful proxies but do NOT reliably predict runtime performance. Always measure actual GPU time with:

- **Vulkan timestamp queries** (`vkCmdWriteTimestamp`) — per-pass GPU timing with nanosecond precision
- **RenderDoc GPU counters** — per-draw timing, ALU utilization, texture fetch stalls
- **Vendor profilers** (NSight, RGP) — for deeper hardware-specific analysis

### 2. Top-Down Analysis

Always start from the highest level and work down:

- **Level 0: Frame structure** — how many passes, how many draws, what's the bottleneck?
- **Level 1: Algorithm** — is the math itself optimal? Redundant work? Wrong approach?
- **Level 2: Data flow** — memory access patterns, cache misses, bandwidth usage
- **Level 3: Instruction-level** — only after levels 0-2 are addressed

### 3. Isolate Changes

One optimization at a time, always with before/after measurement. Multiple simultaneous changes make it impossible to attribute improvements.

### 4. Measure at Statistical Significance

Run enough frames (1000+) to get stable mean. Report min/max/p95. GPU timing has variance from thermal throttling, scheduling, memory contention.

### 5. Visual Parity Is Non-Negotiable

Every optimization must pass PSNR > 40 dB regression check. Faster code that produces wrong pixels is a bug, not an optimization.

### 6. Instruction Count != Performance

Fewer instructions can be slower (more register pressure, worse occupancy). More instructions can be faster (better scheduling, less divergence). The GPU's scheduler, cache hierarchy, and wavefront occupancy dominate over raw instruction count.

### 7. Know Your Bottleneck

Before optimizing anything, determine if you're:

- **ALU-bound**: shader math is the bottleneck -> optimize arithmetic
- **Bandwidth-bound**: texture/buffer reads are the bottleneck -> optimize access patterns
- **Fill-rate-bound**: too many fragments -> reduce overdraw, use early-Z
- **CPU-bound**: driver overhead -> batch draws, reduce state changes

---

## Measurement Tools Reference

| Tool | What It Measures | When to Use |
|------|-----------------|-------------|
| `vkCmdWriteTimestamp` | Per-pass GPU time (ns precision) | Always — primary automated metric |
| RenderDoc (`renderdoccmd`) | Per-draw timing, pipeline state, textures | Deep-dive analysis, visual debugging |
| SPIR-V instruction count | Static code complexity | Quick proxy, NOT a substitute for timing |
| Binary size | SPIR-V binary bytes | Correlates with I-cache pressure, not runtime |

---

## Timestamp Query Layout (Nadrin/PBR)

5 timestamps per frame, 4 measured intervals:

```
TS[0] — frame start (top of command buffer)
  Main Render Pass:
    Draw skybox
    Draw PBR model
TS[1] — after main render pass ends
  Pipeline barrier (color attachment -> shader read)
TS[2] — after barrier
  Tonemap Render Pass:
    Draw fullscreen triangle
TS[3] — after tonemap render pass ends
TS[4] — frame end (before submit)
```

| Interval | Start->End | What it measures |
|----------|-----------|------------------|
| Main pass | TS[0]->TS[1] | Skybox + PBR combined |
| Barrier | TS[1]->TS[2] | Layout transition overhead |
| Tonemap | TS[2]->TS[3] | Tonemap fullscreen pass |
| Total frame | TS[0]->TS[4] | Full GPU frame time |

---

## Experiment Protocol

**Principle: One change at a time, measure before AND after.**

For each experiment:
1. Deploy variant A shaders, build, run `--benchmark 1000`, record timestamps
2. Deploy variant B shaders, build, run `--benchmark 1000`, record timestamps
3. Capture RenderDoc frame for both A and B
4. Compare: timing delta (ms), visual parity (PSNR > 40 dB)
5. Record results

---

## Known Optimization Opportunities

### Shader (pbr.lux)

| # | Issue | Fix | Expected Impact |
|---|-------|-----|-----------------|
| 1 | `pow(1.0 - cosTheta, 5.0)` in Fresnel (called 4x/fragment) | Multiply chain: `t*t*t*t*t` or `t2*t2*t` | Eliminates 4 `OpExtInst Pow` |
| 2 | `texture_levels()` per fragment | Hoist to uniform/specialization constant | Eliminates per-fragment `OpImageQueryLevels` |

### C++ Renderer (vulkan.cpp)

| # | Issue | Fix | Expected Impact |
|---|-------|-----|-----------------|
| 1 | Per-frame descriptor set update | Pre-allocate, only update when changed | Reduces CPU overhead |
| 2 | Single large UBO + dynamic offsets | Fewer descriptor updates, better cache | Minor CPU improvement |

### Low-Level (Only After Measuring)

- Specialization constants for known-at-compile-time values
- Half-precision for intermediate PBR calculations
- Reduce redundant texture fetches
