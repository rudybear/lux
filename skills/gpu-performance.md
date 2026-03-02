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
