I've always dreamed of creating my own shading language.

As someone who's spent years working with GPU rendering, I've always felt that shader languages like GLSL and HLSL carry too much ceremony. Layout qualifiers, manual descriptor binding, stage wiring — all plumbing that gets in the way of the actual rendering math.

I wanted a language where you just write the math: surfaces, materials, lighting — and the compiler handles the rest. But like many side projects, it kept getting postponed. Never enough spare time.

Well, now my dream has come true.

Meet Lux — a math-first shader language that compiles to SPIR-V for Vulkan.

Instead of writing `layout(set=0, binding=1) uniform sampler2D...`, you write:

```
import brdf;

surface CopperMetal {
    brdf: pbr(vec3(0.95, 0.64, 0.54), 0.3, 0.9),
}

pipeline PBRForward {
    geometry: StandardMesh,
    surface: CopperMetal,
}
```

The compiler generates fully typed vertex + fragment SPIR-V automatically.

What makes this project special is that AI helped bring it to life. What would have taken months of evenings and weekends came together remarkably fast with AI as a development partner. The entire compiler — parser, type checker, SPIR-V codegen, standard library, automatic differentiation, ray tracing support — along with native C++ and Rust GPU playgrounds, was built with AI assistance.

Here's what Lux can do today:
- Declarative materials with surface/geometry/pipeline blocks
- 70+ stdlib functions (PBR BRDF, SDF primitives, procedural noise, colorspace)
- Automatic differentiation (@differentiable generates gradient functions)
- Full Vulkan ray tracing pipeline (raygen, closest-hit, miss stages)
- GLSL-to-Lux transpiler
- AI-powered shader generation from natural language
- Native C++ and Rust Vulkan renderers with headless and interactive modes
- 260+ tests, all passing

This is a fun side project, but I plan to keep evolving it — especially in the direction of leveraging AI even more:
- Extracting material properties from 3D models and reference images
- AI-driven shader optimization and quality/performance schedule selection
- Generating complete material pipelines from photographs
- Using differentiable rendering for material parameter fitting

The full source is open: https://github.com/rudybear/lux

If you're into GPU programming, rendering, or the intersection of AI and graphics — I'd love to hear your thoughts.

#ShaderProgramming #GPU #Vulkan #SPIRV #RayTracing #AI #OpenSource #ComputerGraphics #Rendering #SideProject
