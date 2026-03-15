/**
 * Scene data structures for meshes, materials, and lights.
 */

export interface Mesh {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer | null;
  indexCount: number;
  vertexCount: number;
  vertexStride: number;
}

export interface Material {
  baseColor: [number, number, number, number];
  metallic: number;
  roughness: number;
  emissive: [number, number, number];
  textures: Map<string, GPUTextureView>;
  sampler: GPUSampler;
}

export interface Light {
  type: 'directional' | 'point' | 'spot';
  color: [number, number, number];
  intensity: number;
  position: [number, number, number];
  direction: [number, number, number];
  range: number;
  innerCone: number;
  outerCone: number;
}

export interface SceneNode {
  mesh?: Mesh;
  material?: Material;
  transform: Float32Array; // mat4
  children: SceneNode[];
}

export interface Scene {
  nodes: SceneNode[];
  meshes: Mesh[];
  materials: Material[];
  lights: Light[];
}
