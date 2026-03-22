/**
 * MaterialUBO: 144-byte uniform buffer matching the C++ std140 layout.
 *
 * Offset map:
 *   0: vec4 baseColorFactor
 *  16: vec3 emissiveFactor
 *  28: float metallicFactor
 *  32: float roughnessFactor
 *  36: float emissiveStrength
 *  40: float ior
 *  44: float clearcoatFactor
 *  48: float clearcoatRoughnessFactor
 *  52: float sheenRoughnessFactor
 *  56: float transmissionFactor
 *  60: float _pad
 *  64: vec3 sheenColorFactor
 *  76: float _pad0
 *  80: vec4 baseColorUvSt
 *  96: vec4 normalUvSt
 * 112: vec4 mrUvSt
 * 128: float baseColorUvRot
 * 132: float normalUvRot
 * 136: float mrUvRot
 * 140: float _pad1
 *
 * Total: 144 bytes.
 */

import type { Material } from './scene';

/** Size of the material uniform buffer in bytes. */
export const MATERIAL_UBO_SIZE = 144;

/**
 * Pack a Material into a 144-byte ArrayBuffer with std140 layout.
 *
 * Fields not present on the Material interface (UV transforms) fall back to
 * identity defaults (offset=[0,0], scale=[1,1], rotation=0).
 */
export function fillMaterialUBO(mat: Material): ArrayBuffer {
  const buf = new ArrayBuffer(MATERIAL_UBO_SIZE);
  const f = new Float32Array(buf);

  // offset 0: vec4 baseColorFactor (indices 0-3)
  f[0] = mat.baseColor[0];
  f[1] = mat.baseColor[1];
  f[2] = mat.baseColor[2];
  f[3] = mat.baseColor[3];

  // offset 16: vec3 emissiveFactor (indices 4-6)
  f[4] = mat.emissive[0];
  f[5] = mat.emissive[1];
  f[6] = mat.emissive[2];

  // offset 28: float metallicFactor (index 7)
  f[7] = mat.metallic;

  // offset 32: float roughnessFactor (index 8)
  f[8] = mat.roughness;

  // offset 36: float emissiveStrength (index 9)
  f[9] = mat.emissiveStrength;

  // offset 40: float ior (index 10)
  f[10] = mat.ior;

  // offset 44: float clearcoatFactor (index 11)
  f[11] = mat.clearcoatFactor;

  // offset 48: float clearcoatRoughnessFactor (index 12)
  f[12] = mat.clearcoatRoughnessFactor;

  // offset 52: float sheenRoughnessFactor (index 13)
  f[13] = mat.sheenRoughnessFactor;

  // offset 56: float transmissionFactor (index 14)
  f[14] = mat.transmissionFactor;

  // offset 60: float _pad (index 15)
  f[15] = 0.0;

  // offset 64: vec3 sheenColorFactor (indices 16-18)
  f[16] = mat.sheenColorFactor[0];
  f[17] = mat.sheenColorFactor[1];
  f[18] = mat.sheenColorFactor[2];

  // offset 76: float _pad0 (index 19)
  f[19] = 0.0;

  // offset 80: vec4 baseColorUvSt — default [0, 0, 1, 1] (indices 20-23)
  const bcUvSt = (mat as UvTransformMaterial).baseColorUvSt ?? [0, 0, 1, 1];
  f[20] = bcUvSt[0];
  f[21] = bcUvSt[1];
  f[22] = bcUvSt[2];
  f[23] = bcUvSt[3];

  // offset 96: vec4 normalUvSt — default [0, 0, 1, 1] (indices 24-27)
  const nUvSt = (mat as UvTransformMaterial).normalUvSt ?? [0, 0, 1, 1];
  f[24] = nUvSt[0];
  f[25] = nUvSt[1];
  f[26] = nUvSt[2];
  f[27] = nUvSt[3];

  // offset 112: vec4 mrUvSt — default [0, 0, 1, 1] (indices 28-31)
  const mrUvSt = (mat as UvTransformMaterial).mrUvSt ?? [0, 0, 1, 1];
  f[28] = mrUvSt[0];
  f[29] = mrUvSt[1];
  f[30] = mrUvSt[2];
  f[31] = mrUvSt[3];

  // offset 128: float baseColorUvRot (index 32)
  f[32] = (mat as UvTransformMaterial).baseColorUvRot ?? 0;

  // offset 132: float normalUvRot (index 33)
  f[33] = (mat as UvTransformMaterial).normalUvRot ?? 0;

  // offset 136: float mrUvRot (index 34)
  f[34] = (mat as UvTransformMaterial).mrUvRot ?? 0;

  // offset 140: float _pad1 (index 35)
  f[35] = 0.0;

  return buf;
}

/**
 * Extended material interface for UV transform fields.
 * These are optional extensions that may be added to Material by a glTF loader
 * supporting KHR_texture_transform.
 */
interface UvTransformMaterial extends Material {
  baseColorUvSt?: [number, number, number, number];
  normalUvSt?: [number, number, number, number];
  mrUvSt?: [number, number, number, number];
  baseColorUvRot?: number;
  normalUvRot?: number;
  mrUvRot?: number;
}
