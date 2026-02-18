#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragPos;
layout(location = 0) out vec4 outColor;

const vec3 lightDir = vec3(0.0, 1.0, 0.0);
const vec3 lightColor = vec3(1.0, 1.0, 1.0);
const vec3 albedo = vec3(0.8, 0.2, 0.1);

void main() {
    vec3 n = normalize(fragNormal);
    vec3 l = normalize(lightDir);
    float ndotl = max(dot(n, l), 0.0);
    vec3 diffuse = albedo * ndotl;
    outColor = vec4(diffuse, 1.0);
}
