# GPU Performance Optimization

## Instruction Budget Guidelines
- Fragment: <200 ALU for 60fps 1080p, <100 for 90fps VR
- Vertex: <50 ALU ops
- Texture samples: <4 mobile, <8 desktop
- Branches: <3 in fragment (divergent execution)
- VGPR pressure: <30 low, 30-60 medium, >60 high (affects occupancy)

## Common Optimizations
- `pow(x, 2.0)` → `x * x` (strength reduction)
- `pow(x, 0.5)` → `sqrt(x)` (cheaper instruction)
- `length(v) * length(v)` → `dot(v, v)` (avoid sqrt)
- Avoid `OpFDiv` where possible — multiply by reciprocal
- CSE repeated `normalize()` and `dot()` calls

## Quality Gates
- PSNR >= 40 dB: imperceptible difference
- SSIM >= 0.99: structural lossless
- PSNR >= 30 dB, SSIM >= 0.95: acceptable for real-time

## Learned Patterns
<!-- Auto-populated by tools/skill_updater.py after experiments -->

### Compute-shader threadgroup tiling (from optimizing lux's fused-GPU UNet)

Retiling a naive "one thread per (pixel, output channel)" convolution kernel
into "one threadgroup per spatial tile, staged once into `shared` memory"
is the single biggest win available for small-kernel (3x3-ish) conv-style
compute shaders — it turns Cout-way redundant global reads into one shared
load per tile. But two follow-on traps ate most of the actual win on Apple
Silicon and are easy to miss if you only look at "did it get faster than
before":

- **A tile sized for the worst-case channel count caps EVERY caller's
  occupancy at the worst caller's requirement.** One generic kernel reused
  across layers of very different channel counts (a UNet's stem vs. its
  bottleneck, say) needs its `shared` array sized for the largest `cin`
  any caller passes at runtime — but that reservation is paid by every
  call, including the tiny-channel/huge-resolution ones where it's pure
  waste. Concretely: an 8x8-tile, 64-channel-max conv3x3 kernel spent
  25.6KB/32KB of threadgroup memory on EVERY dispatch, capping how many
  threadgroups could be co-resident per GPU core even for an 18-channel
  layer at full resolution — that layer alone got SLOWER after "optimizing"
  it (91.8ms baseline -> 183.9ms tiled) because occupancy collapsed further
  than the memory-traffic win recovered. Fix: compile multiple kernel
  variants bucketed by channel-count ceiling (a "lo" and "hi" variant, tile
  size traded against channel-max), and have the host pick per-layer by
  comparing the layer's actual `cin`/`cout` against each variant's ceiling.
  This alone took the UNet from 183.9ms back under the original 91.8ms.
- **A tile with only a handful of threads (one thread per output pixel,
  looping over the full output-channel count internally) has WORSE
  occupancy than the untiled design it's replacing, even though it does
  far less redundant memory traffic.** The threadgroup itself doing less
  total work isn't the goal; keeping the GPU saturated is. Fix: add a
  channel-parallel dimension to the threadgroup (3D dispatch, Z = e.g. 8)
  so `cout` is split across Z-many threads that all read the SAME staged
  tile — this recovers thread count/occupancy without adding any shared-
  memory pressure (the tile's size depends only on the spatial extent and
  `cin`, not on how many threads consume it). Only the Z==0 slice needs to
  do the cooperative tile load; every thread must still hit the `barrier()`
  uniformly regardless of Z. Don't assume more Z is strictly better past
  the point where the untiled design's thread count is matched — one
  kernel measured WORSE going from Z=8 (256-512 threads/threadgroup) to
  Z=16 (512-1024), likely register/occupancy pressure from the bigger
  threadgroup outweighing the extra parallelism.
- Apple GPUs (measured on M4 Max) have a hard 32768-byte-per-threadgroup
  `shared`/`threadgroup` memory limit — not just a "recommended" one.
  Exceeding it fails Metal pipeline creation outright ("Threadgroup memory
  size exceeds the maximum threadgroup memory allowed") rather than
  silently falling back to something slower, so it's a hard constraint to
  size tiles against, not a soft one to tune.
- Bake zero-padding into the tile LOAD (fill halo cells with 0 for
  out-of-bounds source positions) rather than checking bounds per tap in
  the inner conv loop. This moves the branch from "once per (output
  channel, tap)" to "once per tile cell" — a big win once the inner loop is
  looping over many output channels per thread.
- When profiling a compute pipeline of several dispatches with very
  different (resolution, channel-count) shapes, don't trust a single
  end-to-end number to tell you where time goes — per-dispatch GPU
  timestamps (here: Metal's `MTL::CommandBuffer::GPUStartTime/GPUEndTime`
  per dispatch) are essential; the layer that "should" be cheapest
  arithmetically (e.g. the highest-resolution, lowest-channel-count layer)
  is often the actual bottleneck once occupancy effects dominate over
  raw FLOP count.
