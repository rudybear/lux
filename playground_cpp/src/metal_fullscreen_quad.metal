#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle vertex shader (no vertex buffer needed)
// Generates 3 vertices that cover the entire screen when rendered as a triangle.
vertex VertexOut fullscreen_vert(uint vid [[vertex_id]]) {
    VertexOut out;
    float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    float2 uvs[3] = {float2(0.0, 1.0), float2(2.0, 1.0), float2(0.0, -1.0)};
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}
