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

  // Alpha
  alphaMode: 'OPAQUE' | 'BLEND' | 'MASK';
  alphaCutoff: number;
  doubleSided: boolean;

  // IOR
  ior: number;

  // Emissive strength (KHR_materials_emissive_strength)
  emissiveStrength: number;

  // Unlit (KHR_materials_unlit)
  isUnlit: boolean;

  // Clearcoat (KHR_materials_clearcoat)
  hasClearcoat: boolean;
  clearcoatFactor: number;
  clearcoatRoughnessFactor: number;

  // Sheen (KHR_materials_sheen)
  hasSheen: boolean;
  sheenColorFactor: [number, number, number];
  sheenRoughnessFactor: number;

  // Transmission (KHR_materials_transmission)
  hasTransmission: boolean;
  transmissionFactor: number;

  // Original glTF material index
  materialIndex: number;
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

export interface DrawRange {
  indexOffset: number;
  indexCount: number;
  vertexOffset: number;
  vertexCount: number;
  materialIndex: number;
  worldTransform: Float32Array;  // mat4
  /** Index into Scene.meshes[] — used by gltf-pbr to find vertex/index buffers. */
  meshIndex: number;
}

export interface Scene {
  nodes: SceneNode[];
  meshes: Mesh[];
  materials: Material[];
  lights: Light[];
  boundsMin: [number, number, number];
  boundsMax: [number, number, number];
  drawRanges: DrawRange[];
}
