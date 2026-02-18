#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        color = color + vec3(0.25);
    }
    outColor = vec4(color, 1.0);
}
