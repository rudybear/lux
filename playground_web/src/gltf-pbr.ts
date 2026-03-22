/**
 * glTF PBR rendering pipeline using Lux compiler-produced WGSL shaders.
 *
 * Loads gltf_pbr.vert.wgsl + gltf_pbr.frag.wgsl (compiled from examples/gltf_pbr.lux)
 * and creates the pipeline via ReflectedPipeline from the reflection JSON.
 *
 * Layout (from compiler reflection):
 *   Set 0, binding 0: MVP uniform (model, view, projection — 192 bytes)
 *   Set 1, binding 0: Light uniform (light_dir, view_pos — 32 bytes)
 *   Set 1, binding 1-10: material textures (base_color, normal, metallic_roughness, occlusion, emissive)
 *   Set 1, binding 11-16: IBL textures (env_specular cubemap, env_irradiance cubemap, brdf_lut 2D)
 *
 * Supports per-draw model matrices, per-material bind groups, and alpha-mode draw ordering.
 */

import { loadShader } from './shader-loader';
import { ReflectedPipeline } from './reflected-pipeline';
import type { Scene, Material } from './scene';
import type { DrawCall, UniformBufferBinding } from './render-engine';
import type { ProceduralIBL } from './procedural-ibl';
import { fillMaterialUBO, MATERIAL_UBO_SIZE } from './material-ubo';
import {
  populateLightsFromScene, packLightsBuffer, packSceneLightUBO,
  LIGHT_BUFFER_STRIDE, MAX_LIGHTS,
} from './lights';

// --------------------------------------------------------------------------
// Pipeline creation (async — loads compiler-produced shaders)
// --------------------------------------------------------------------------

export async function createGltfPbrPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
): Promise<ReflectedPipeline> {
  const [vertShader, fragShader] = await Promise.all([
    loadShader(device, 'shaders/gltf_pbr.vert'),
    loadShader(device, 'shaders/gltf_pbr.frag'),
  ]);
  return ReflectedPipeline.create(
    device, vertShader.module, fragShader.module,
    vertShader.reflection, fragShader.reflection,
    format,
    'depth24plus',
    'none', // double-sided
  );
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

/** Alpha mode sort order: OPAQUE first, MASK second, BLEND last. */
const ALPHA_ORDER: Record<string, number> = { OPAQUE: 0, MASK: 1, BLEND: 2 };

export function buildGltfSceneDrawCalls(
  device: GPUDevice,
  pipeline: ReflectedPipeline,
  scene: Scene,
  ibl: ProceduralIBL,
): { draws: DrawCall[]; uniformBuffers: UniformBufferBinding[]; mvpBindGroup: GPUBindGroup; storageBuffers?: { buffer: GPUBuffer; data: Float32Array }[] } {
  // Default textures
  const whiteTex = create1x1Texture(device, 255, 255, 255, 255);
  const normalTex = create1x1Texture(device, 128, 128, 255, 255);
  const blackTex = create1x1Texture(device, 0, 0, 0, 255);
  const defaultSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  });

  const whiteView = whiteTex.createView();
  const normalView = normalTex.createView();
  const blackView = blackTex.createView();

  // Light uniform buffer (set 1, binding 0 — 32 bytes: light_dir + view_pos)
  const lightBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'light_uniforms',
  });

  const uniformBuffers: UniformBufferBinding[] = [
    {
      buffer: lightBuffer,
      fields: [
        { name: 'light_dir', type: 'vec3', offset: 0, size: 12 },
        { name: 'view_pos', type: 'vec3', offset: 16, size: 12 },
      ],
      size: 32,
    },
  ];

  // Per-material bind group cache (keyed by materialIndex)
  const materialBindGroups = new Map<number, GPUBindGroup>();
  const matLayout = pipeline.bindGroupLayouts[1];

  function getOrCreateMaterialBindGroup(materialIndex: number): GPUBindGroup {
    if (materialBindGroups.has(materialIndex)) return materialBindGroups.get(materialIndex)!;
    const mat = materialIndex >= 0 && materialIndex < scene.materials.length
      ? scene.materials[materialIndex] : null;
    const bg = createMaterialBindGroup(
      device, matLayout, mat, lightBuffer, defaultSampler, ibl,
      whiteView, normalView, blackView,
    );
    materialBindGroups.set(materialIndex, bg);
    return bg;
  }

  // Build draws
  const draws: DrawCall[] = [];

  if (scene.drawRanges.length > 0) {
    // ---- New path: use drawRanges with per-draw model matrices ----
    for (let i = 0; i < scene.drawRanges.length; i++) {
      const dr = scene.drawRanges[i];
      const mesh = scene.meshes[dr.meshIndex];
      if (!mesh) continue;

      // Per-draw MVP buffer (192 bytes: model + view + proj)
      const mvpBuffer = device.createBuffer({
        size: 192,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `mvp_draw_${i}`,
      });

      // Per-draw MVP bind group (set 0)
      const mvpBindGroup = device.createBindGroup({
        layout: pipeline.bindGroupLayouts[0],
        entries: [{ binding: 0, resource: { buffer: mvpBuffer } }],
        label: `mvp_bg_${i}`,
      });

      const materialBindGroup = getOrCreateMaterialBindGroup(dr.materialIndex);
      const mat = dr.materialIndex >= 0 && dr.materialIndex < scene.materials.length
        ? scene.materials[dr.materialIndex] : null;

      draws.push({
        vertexBuffer: mesh.vertexBuffer,
        indexBuffer: mesh.indexBuffer,
        indexCount: mesh.indexCount,
        vertexCount: mesh.vertexCount,
        materialBindGroup,
        mvpBindGroup,
        mvpBuffer,
        worldTransform: dr.worldTransform,
        alphaMode: mat?.alphaMode ?? 'OPAQUE',
      });
    }
  } else {
    // ---- Fallback path: walk node tree (backward compat for scenes without drawRanges) ----
    function collectDraws(nodes: Scene['nodes']): void {
      for (const node of nodes) {
        if (node.mesh) {
          const mat = node.material;
          const mvpBuffer = device.createBuffer({
            size: 192,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          const mvpBindGroup = device.createBindGroup({
            layout: pipeline.bindGroupLayouts[0],
            entries: [{ binding: 0, resource: { buffer: mvpBuffer } }],
          });
          const materialBindGroup = createMaterialBindGroup(
            device, matLayout, mat ?? null,
            lightBuffer, defaultSampler, ibl,
            whiteView, normalView, blackView,
          );
          draws.push({
            vertexBuffer: node.mesh.vertexBuffer,
            indexBuffer: node.mesh.indexBuffer,
            indexCount: node.mesh.indexCount,
            vertexCount: node.mesh.vertexCount,
            materialBindGroup,
            mvpBindGroup,
            mvpBuffer,
            worldTransform: node.transform,
            alphaMode: (mat?.alphaMode) ?? 'OPAQUE',
          });
        }
        collectDraws(node.children);
      }
    }
    collectDraws(scene.nodes);
  }

  // Sort draws: OPAQUE first, MASK second, BLEND last
  draws.sort((a, b) =>
    (ALPHA_ORDER[a.alphaMode ?? 'OPAQUE'] ?? 0) - (ALPHA_ORDER[b.alphaMode ?? 'OPAQUE'] ?? 0),
  );

  // Create a shared MVP bind group (needed for backward compat — won't be used
  // when per-draw mvpBindGroup exists, but callers expect it in the return value)
  const sharedMvpBuffer = device.createBuffer({
    size: 192,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'mvp_shared',
  });
  const mvpBindGroup = device.createBindGroup({
    layout: pipeline.bindGroupLayouts[0],
    entries: [{ binding: 0, resource: { buffer: sharedMvpBuffer } }],
    label: 'mvp_bg_shared',
  });

  // Add shared MVP to uniformBuffers for backward compat (engine writes model/view/proj)
  uniformBuffers.unshift({
    buffer: sharedMvpBuffer,
    fields: [
      { name: 'model', type: 'mat4', offset: 0, size: 64 },
      { name: 'view', type: 'mat4', offset: 64, size: 64 },
      { name: 'projection', type: 'mat4', offset: 128, size: 64 },
    ],
    size: 192,
  });

  return { draws, uniformBuffers, mvpBindGroup };
}

/**
 * Create a material bind group (set 1) matching the compiler reflection layout:
 *   binding 0: Light uniform buffer
 *   binding 1-2: base_color_tex (sampler + texture)
 *   binding 3-4: normal_tex (sampler + texture)
 *   binding 5-6: metallic_roughness_tex (sampler + texture)
 *   binding 7-8: occlusion_tex (sampler + texture)
 *   binding 9-10: emissive_tex (sampler + texture)
 *   binding 11-12: env_specular (sampler + cubemap)
 *   binding 13-14: env_irradiance (sampler + cubemap)
 *   binding 15-16: brdf_lut (sampler + texture)
 */
function createMaterialBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  mat: Material | null,
  lightBuffer: GPUBuffer,
  defaultSampler: GPUSampler,
  ibl: ProceduralIBL,
  defaultWhiteView: GPUTextureView,
  defaultNormalView: GPUTextureView,
  defaultBlackView: GPUTextureView,
): GPUBindGroup {
  const sampler = mat?.sampler ?? defaultSampler;
  const baseColorView = mat?.textures.get('baseColor') ?? defaultWhiteView;
  const normalMapView = mat?.textures.get('normal') ?? defaultNormalView;
  const mrView = mat?.textures.get('metallicRoughness') ?? defaultWhiteView;
  const occlusionView = mat?.textures.get('occlusion') ?? defaultWhiteView;
  const emissiveView = mat?.textures.get('emissive') ?? defaultBlackView;

  return device.createBindGroup({
    layout,
    entries: [
      // Light uniform
      { binding: 0, resource: { buffer: lightBuffer } },
      // Material textures (each has sampler + texture)
      { binding: 1, resource: sampler },
      { binding: 2, resource: baseColorView },
      { binding: 3, resource: sampler },
      { binding: 4, resource: normalMapView },
      { binding: 5, resource: sampler },
      { binding: 6, resource: mrView },
      { binding: 7, resource: sampler },
      { binding: 8, resource: occlusionView },
      { binding: 9, resource: sampler },
      { binding: 10, resource: emissiveView },
      // IBL cubemaps
      { binding: 11, resource: ibl.sampler },
      { binding: 12, resource: ibl.specularView },
      { binding: 13, resource: ibl.sampler },
      { binding: 14, resource: ibl.irradianceView },
      { binding: 15, resource: ibl.sampler },
      { binding: 16, resource: ibl.brdfLutView },
    ],
    label: 'material_bg',
  });
}
