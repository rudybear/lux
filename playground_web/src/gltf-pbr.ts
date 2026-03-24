/**
 * glTF PBR rendering pipeline using Lux compiler-produced WGSL shaders.
 *
 * Uses gltf_pbr_web shader (compiled from examples/gltf_pbr_web.lux) which has:
 *   - ALL texture bindings (base_color, normal, metallic_roughness, occlusion, emissive, IBL)
 *   - Material UBO for live editing (base_color_factor, metallic/roughness factors, emissive)
 *   - Light UBO with light_dir, view_pos, light_color
 *
 * Layout (set 1):
 *   b0:  Light UBO (48 bytes)
 *   b1:  Material UBO (48 bytes)
 *   b2-3:  base_color_tex (sampler + texture)
 *   b4-5:  normal_tex
 *   b6-7:  metallic_roughness_tex
 *   b8-9:  occlusion_tex
 *   b10-11: emissive_tex
 *   b12-13: env_specular (cubemap)
 *   b14-15: env_irradiance (cubemap)
 *   b16-17: brdf_lut
 */

import { loadShader } from './shader-loader';
import { ReflectedPipeline } from './reflected-pipeline';
import type { Scene, Material } from './scene';
import type { DrawCall, UniformBufferBinding } from './render-engine';
import type { ProceduralIBL } from './procedural-ibl';

// --------------------------------------------------------------------------
// Material UBO for gltf_pbr_web (48 bytes — simpler than the 144-byte layered)
// --------------------------------------------------------------------------

/** Size of the Material UBO for gltf_pbr_web shader. */
const WEB_MATERIAL_UBO_SIZE = 48;

/** Pack material into the 48-byte UBO matching gltf_pbr_web.lux Material block. */
function fillWebMaterialUBO(mat: Material): ArrayBuffer {
  // Layout (std140):
  //  0: vec4 base_color_factor (16 bytes)
  // 16: vec3 emissive_factor (12 bytes)
  // 28: float metallic_factor (4 bytes)
  // 32: float roughness_factor (4 bytes)
  // 36: float emissive_strength (4 bytes)
  // 40: 8 bytes padding to 48
  const buf = new ArrayBuffer(WEB_MATERIAL_UBO_SIZE);
  const f = new Float32Array(buf);
  f[0] = mat.baseColor[0];
  f[1] = mat.baseColor[1];
  f[2] = mat.baseColor[2];
  f[3] = mat.baseColor[3];
  f[4] = mat.emissive[0];
  f[5] = mat.emissive[1];
  f[6] = mat.emissive[2];
  f[7] = mat.metallic;
  f[8] = mat.roughness;
  f[9] = mat.emissiveStrength;
  // f[10], f[11] = padding
  return buf;
}

// Re-export for main.ts
export { fillWebMaterialUBO, WEB_MATERIAL_UBO_SIZE };

// --------------------------------------------------------------------------
// Pipeline creation
// --------------------------------------------------------------------------

export async function createGltfPbrPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
): Promise<ReflectedPipeline> {
  try {
    const [vertShader, fragShader] = await Promise.all([
      loadShader(device, 'shaders/gltf_pbr_web.vert'),
      loadShader(device, 'shaders/gltf_pbr_web.frag'),
    ]);
    return ReflectedPipeline.create(
      device, vertShader.module, fragShader.module,
      vertShader.reflection, fragShader.reflection,
      format, 'depth24plus', 'none',
    );
  } catch (e) {
    console.warn('gltf_pbr_web not available, falling back to gltf_pbr:', e);
    const [vertShader, fragShader] = await Promise.all([
      loadShader(device, 'shaders/gltf_pbr.vert'),
      loadShader(device, 'shaders/gltf_pbr.frag'),
    ]);
    return ReflectedPipeline.create(
      device, vertShader.module, fragShader.module,
      vertShader.reflection, fragShader.reflection,
      format, 'depth24plus', 'none',
    );
  }
}

// --------------------------------------------------------------------------
// Texture helpers
// --------------------------------------------------------------------------

export function create1x1Texture(
  device: GPUDevice,
  r: number, g: number, b: number, a: number,
): GPUTexture {
  const tex = device.createTexture({
    size: [1, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: `1x1_${r}_${g}_${b}_${a}`,
  });
  device.queue.writeTexture(
    { texture: tex },
    new Uint8Array([r, g, b, a]),
    { bytesPerRow: 4 },
    [1, 1],
  );
  return tex;
}

// --------------------------------------------------------------------------
// Build draw calls from glTF scene
// --------------------------------------------------------------------------

const ALPHA_ORDER: Record<string, number> = { OPAQUE: 0, MASK: 1, BLEND: 2 };

export function buildGltfSceneDrawCalls(
  device: GPUDevice,
  pipeline: ReflectedPipeline,
  scene: Scene,
  ibl: ProceduralIBL,
): {
  draws: DrawCall[];
  uniformBuffers: UniformBufferBinding[];
  mvpBindGroup: GPUBindGroup;
  materialUboBuffers: Map<number, GPUBuffer>;
} {
  // Default textures
  const whiteTex = create1x1Texture(device, 255, 255, 255, 255);
  const normalTex = create1x1Texture(device, 128, 128, 255, 255);
  const blackTex = create1x1Texture(device, 0, 0, 0, 255);
  const defaultSampler = device.createSampler({
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    addressModeU: 'repeat', addressModeV: 'repeat',
  });
  const whiteView = whiteTex.createView();
  const normalView = normalTex.createView();
  const blackView = blackTex.createView();

  // Light UBO (48 bytes: light_dir vec3, view_pos vec3, light_color vec3)
  const lightBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'light_ubo',
  });

  const uniformBuffers: UniformBufferBinding[] = [
    {
      buffer: lightBuffer,
      fields: [
        { name: 'light_dir', type: 'vec3', offset: 0, size: 12 },
        { name: 'view_pos', type: 'vec3', offset: 16, size: 12 },
        { name: 'light_color', type: 'vec3', offset: 32, size: 12 },
      ],
      size: 48,
    },
  ];

  // Per-material UBO buffers
  const materialUboBuffers = new Map<number, GPUBuffer>();

  function getOrCreateMaterialUbo(materialIndex: number): GPUBuffer {
    if (materialUboBuffers.has(materialIndex)) return materialUboBuffers.get(materialIndex)!;
    const mat = materialIndex >= 0 && materialIndex < scene.materials.length
      ? scene.materials[materialIndex] : null;
    const data = mat ? fillWebMaterialUBO(mat) : new ArrayBuffer(WEB_MATERIAL_UBO_SIZE);
    const buf = device.createBuffer({
      size: WEB_MATERIAL_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: `material_ubo_${materialIndex}`,
    });
    device.queue.writeBuffer(buf, 0, data);
    materialUboBuffers.set(materialIndex, buf);
    return buf;
  }

  // Per-material bind group cache
  const materialBindGroups = new Map<number, GPUBindGroup>();
  const matLayout = pipeline.bindGroupLayouts[1];

  function getOrCreateMaterialBindGroup(materialIndex: number): GPUBindGroup {
    if (materialBindGroups.has(materialIndex)) return materialBindGroups.get(materialIndex)!;
    const mat = materialIndex >= 0 && materialIndex < scene.materials.length
      ? scene.materials[materialIndex] : null;
    const materialUbo = getOrCreateMaterialUbo(materialIndex);

    const sampler = mat?.sampler ?? defaultSampler;
    const baseColorView = mat?.textures.get('baseColor') ?? whiteView;
    const normalMapView = mat?.textures.get('normal') ?? normalView;
    const mrView = mat?.textures.get('metallicRoughness') ?? whiteView;
    const occlusionView = mat?.textures.get('occlusion') ?? whiteView;
    const emissiveView = mat?.textures.get('emissive') ?? blackView;

    const bg = device.createBindGroup({
      layout: matLayout,
      entries: [
        // Light UBO (b0)
        { binding: 0, resource: { buffer: lightBuffer } },
        // Material UBO (b1)
        { binding: 1, resource: { buffer: materialUbo } },
        // base_color_tex (b2-3)
        { binding: 2, resource: sampler },
        { binding: 3, resource: baseColorView },
        // normal_tex (b4-5)
        { binding: 4, resource: sampler },
        { binding: 5, resource: normalMapView },
        // metallic_roughness_tex (b6-7)
        { binding: 6, resource: sampler },
        { binding: 7, resource: mrView },
        // occlusion_tex (b8-9)
        { binding: 8, resource: sampler },
        { binding: 9, resource: occlusionView },
        // emissive_tex (b10-11)
        { binding: 10, resource: sampler },
        { binding: 11, resource: emissiveView },
        // env_specular (b12-13)
        { binding: 12, resource: ibl.sampler },
        { binding: 13, resource: ibl.specularView },
        // env_irradiance (b14-15)
        { binding: 14, resource: ibl.sampler },
        { binding: 15, resource: ibl.irradianceView },
        // brdf_lut (b16-17)
        { binding: 16, resource: ibl.sampler },
        { binding: 17, resource: ibl.brdfLutView },
      ],
      label: `material_bg_${materialIndex}`,
    });
    materialBindGroups.set(materialIndex, bg);
    return bg;
  }

  // Build draws
  const draws: DrawCall[] = [];

  if (scene.drawRanges.length > 0) {
    for (let i = 0; i < scene.drawRanges.length; i++) {
      const dr = scene.drawRanges[i];
      const mesh = scene.meshes[dr.meshIndex];
      if (!mesh) continue;

      const mvpBuffer = device.createBuffer({
        size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `mvp_draw_${i}`,
      });
      const mvpBg = device.createBindGroup({
        layout: pipeline.bindGroupLayouts[0],
        entries: [{ binding: 0, resource: { buffer: mvpBuffer } }],
        label: `mvp_bg_${i}`,
      });

      const mat = dr.materialIndex >= 0 && dr.materialIndex < scene.materials.length
        ? scene.materials[dr.materialIndex] : null;

      draws.push({
        vertexBuffer: mesh.vertexBuffer,
        indexBuffer: mesh.indexBuffer,
        indexCount: mesh.indexCount,
        vertexCount: mesh.vertexCount,
        materialBindGroup: getOrCreateMaterialBindGroup(dr.materialIndex),
        mvpBindGroup: mvpBg,
        mvpBuffer,
        worldTransform: dr.worldTransform,
        alphaMode: mat?.alphaMode ?? 'OPAQUE',
      });
    }
  } else {
    // Fallback: walk node tree
    function collectDraws(nodes: Scene['nodes']): void {
      for (const node of nodes) {
        if (node.mesh) {
          const mat = node.material;
          const matIdx = mat?.materialIndex ?? -1;
          const mvpBuffer = device.createBuffer({
            size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          const mvpBg = device.createBindGroup({
            layout: pipeline.bindGroupLayouts[0],
            entries: [{ binding: 0, resource: { buffer: mvpBuffer } }],
          });
          draws.push({
            vertexBuffer: node.mesh.vertexBuffer,
            indexBuffer: node.mesh.indexBuffer,
            indexCount: node.mesh.indexCount,
            vertexCount: node.mesh.vertexCount,
            materialBindGroup: getOrCreateMaterialBindGroup(matIdx),
            mvpBindGroup: mvpBg,
            mvpBuffer,
            worldTransform: node.transform,
            alphaMode: mat?.alphaMode ?? 'OPAQUE',
          });
        }
        collectDraws(node.children);
      }
    }
    collectDraws(scene.nodes);
  }

  // Sort: OPAQUE first, MASK second, BLEND last
  draws.sort((a, b) =>
    (ALPHA_ORDER[a.alphaMode ?? 'OPAQUE'] ?? 0) - (ALPHA_ORDER[b.alphaMode ?? 'OPAQUE'] ?? 0),
  );

  // Shared MVP bind group (backward compat)
  const sharedMvpBuffer = device.createBuffer({
    size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'mvp_shared',
  });
  const mvpBindGroup = device.createBindGroup({
    layout: pipeline.bindGroupLayouts[0],
    entries: [{ binding: 0, resource: { buffer: sharedMvpBuffer } }],
    label: 'mvp_bg_shared',
  });
  uniformBuffers.unshift({
    buffer: sharedMvpBuffer,
    fields: [
      { name: 'model', type: 'mat4', offset: 0, size: 64 },
      { name: 'view', type: 'mat4', offset: 64, size: 64 },
      { name: 'projection', type: 'mat4', offset: 128, size: 64 },
    ],
    size: 192,
  });

  return { draws, uniformBuffers, mvpBindGroup, materialUboBuffers };
}
