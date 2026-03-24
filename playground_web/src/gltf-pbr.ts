/**
 * glTF PBR rendering pipeline using Lux compiler-produced WGSL shaders.
 *
 * Loads gltf_pbr_layered+emission+normal_map shaders (compiled from examples/gltf_pbr_layered.lux)
 * with fallback to gltf_pbr shaders, and creates the pipeline via ReflectedPipeline
 * from the reflection JSON.
 *
 * Layout (from compiler reflection — layered shader):
 *   Set 0, binding 0: MVP uniform (model, view, projection — 192 bytes)
 *   Set 1, binding 0: SceneLight UBO (view_pos vec3 + light_count int — 16 bytes)
 *   Set 1, binding 1: Material UBO (144 bytes)
 *   Set 1, binding 2-7: material textures (base_color, metallic_roughness, occlusion)
 *   Set 1, binding 8-13: IBL textures (env_specular cubemap, env_irradiance cubemap, brdf_lut 2D)
 *   Set 1, binding 14: lights storage buffer (SSBO)
 *
 * Supports per-draw model matrices, per-material bind groups with Material UBO for
 * live editing, and alpha-mode draw ordering.
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
  // Try layered shader first, fall back to basic
  try {
    const [vertShader, fragShader] = await Promise.all([
      loadShader(device, 'shaders/gltf_pbr_full.vert'),
      loadShader(device, 'shaders/gltf_pbr_full.frag'),
    ]);
    // Stride override: glTF loader interleaves at 48 bytes (pos3+nor3+uv2+tan4)
    // even though this shader only reads 32 bytes (pos3+nor3+uv2)
    return ReflectedPipeline.create(
      device, vertShader.module, fragShader.module,
      vertShader.reflection, fragShader.reflection,
      format, 'depth24plus', 'none', 48,
    );
  } catch {
    // Fallback to basic gltf_pbr (stride 48 matches its vertex layout)
    const [vertShader, fragShader] = await Promise.all([
      loadShader(device, 'shaders/gltf_pbr.vert'),
      loadShader(device, 'shaders/gltf_pbr.frag'),
    ]);
    return ReflectedPipeline.create(
      device, vertShader.module, fragShader.module,
      vertShader.reflection, fragShader.reflection,
      format, 'depth24plus', 'none', 48,
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

/** Alpha mode sort order: OPAQUE first, MASK second, BLEND last. */
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
  storageBuffers?: { buffer: GPUBuffer; data: Float32Array }[];
  materialUboBuffers: Map<number, GPUBuffer>;
} {
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

  // -- Lights --
  const lights = populateLightsFromScene(scene);
  const lightsData = packLightsBuffer(lights);

  // Lights SSBO (set 1, binding 14)
  const lightsBuffer = device.createBuffer({
    size: Math.max(lightsData.byteLength, 64),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'lights_ssbo',
  });
  device.queue.writeBuffer(lightsBuffer, 0, lightsData.buffer, lightsData.byteOffset, lightsData.byteLength);

  // SceneLight UBO (set 1, binding 0 — 16 bytes: view_pos + light_count)
  const sceneLightBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'scene_light_ubo',
  });

  const uniformBuffers: UniformBufferBinding[] = [
    {
      buffer: sceneLightBuffer,
      fields: [
        { name: 'view_pos', type: 'vec3', offset: 0, size: 12 },
        { name: 'light_count', type: 'int', offset: 12, size: 4 },
      ],
      size: 16,
    },
  ];

  // Per-material UBO buffers (for live editing)
  const materialUboBuffers = new Map<number, GPUBuffer>();

  function getOrCreateMaterialUboBuffer(materialIndex: number): GPUBuffer {
    if (materialUboBuffers.has(materialIndex)) return materialUboBuffers.get(materialIndex)!;
    const mat = materialIndex >= 0 && materialIndex < scene.materials.length
      ? scene.materials[materialIndex] : null;
    const uboData = mat ? fillMaterialUBO(mat) : new ArrayBuffer(MATERIAL_UBO_SIZE);
    const buf = device.createBuffer({
      size: MATERIAL_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: `material_ubo_${materialIndex}`,
    });
    device.queue.writeBuffer(buf, 0, uboData);
    materialUboBuffers.set(materialIndex, buf);
    return buf;
  }

  // Per-material bind group cache (keyed by materialIndex)
  const materialBindGroups = new Map<number, GPUBindGroup>();
  const matLayout = pipeline.bindGroupLayouts[1];

  function getOrCreateMaterialBindGroup(materialIndex: number): GPUBindGroup {
    if (materialBindGroups.has(materialIndex)) return materialBindGroups.get(materialIndex)!;
    const mat = materialIndex >= 0 && materialIndex < scene.materials.length
      ? scene.materials[materialIndex] : null;
    const materialUboBuffer = getOrCreateMaterialUboBuffer(materialIndex);
    const bg = createMaterialBindGroup(
      device, matLayout, mat,
      sceneLightBuffer, materialUboBuffer,
      defaultSampler, ibl, lightsBuffer,
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
          const matIndex = mat?.materialIndex ?? -1;
          const mvpBuffer = device.createBuffer({
            size: 192,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          const mvpBindGroup = device.createBindGroup({
            layout: pipeline.bindGroupLayouts[0],
            entries: [{ binding: 0, resource: { buffer: mvpBuffer } }],
          });
          const materialBindGroup = getOrCreateMaterialBindGroup(matIndex);
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

  return { draws, uniformBuffers, mvpBindGroup, materialUboBuffers };
}

/**
 * Create a material bind group (set 1) matching the layered shader reflection layout:
 *   binding 0: SceneLight UBO (16 bytes: view_pos vec3 + light_count int)
 *   binding 1: Material UBO (144 bytes)
 *   binding 2-3: base_color_tex (sampler + texture)
 *   binding 4-5: metallic_roughness_tex (sampler + texture)
 *   binding 6-7: occlusion_tex (sampler + texture)
 *   binding 8-9: env_specular (sampler + cubemap)
 *   binding 10-11: env_irradiance (sampler + cubemap)
 *   binding 12-13: brdf_lut (sampler + texture)
 *   binding 14: lights storage buffer (SSBO)
 */
function createMaterialBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  mat: Material | null,
  sceneLightBuffer: GPUBuffer,
  materialUboBuffer: GPUBuffer,
  defaultSampler: GPUSampler,
  ibl: ProceduralIBL,
  lightsBuffer: GPUBuffer,
  defaultWhiteView: GPUTextureView,
  _defaultNormalView: GPUTextureView,
  _defaultBlackView: GPUTextureView,
): GPUBindGroup {
  const sampler = mat?.sampler ?? defaultSampler;
  const baseColorView = mat?.textures.get('baseColor') ?? defaultWhiteView;
  const mrView = mat?.textures.get('metallicRoughness') ?? defaultWhiteView;
  const occlusionView = mat?.textures.get('occlusion') ?? defaultWhiteView;

  return device.createBindGroup({
    layout,
    entries: [
      // SceneLight UBO
      { binding: 0, resource: { buffer: sceneLightBuffer } },
      // Material UBO
      { binding: 1, resource: { buffer: materialUboBuffer } },
      // Material textures (each has sampler + texture)
      { binding: 2, resource: sampler },
      { binding: 3, resource: baseColorView },
      { binding: 4, resource: sampler },
      { binding: 5, resource: mrView },
      { binding: 6, resource: sampler },
      { binding: 7, resource: occlusionView },
      // IBL cubemaps
      { binding: 8, resource: ibl.sampler },
      { binding: 9, resource: ibl.specularView },
      { binding: 10, resource: ibl.sampler },
      { binding: 11, resource: ibl.irradianceView },
      { binding: 12, resource: ibl.sampler },
      { binding: 13, resource: ibl.brdfLutView },
      // Lights SSBO
      { binding: 14, resource: { buffer: lightsBuffer } },
    ],
    label: 'material_bg',
  });
}
