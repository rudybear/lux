/**
 * Shadow map infrastructure for WebGPU.
 *
 * Creates a depth-only render pass for shadow-casting lights,
 * producing a 2D depth array texture used by the main PBR pass.
 *
 * Matches the C++ engine's shadow mapping:
 *   - 2048x2048 depth texture array (up to 8 layers)
 *   - Per-light orthographic/perspective shadow matrix
 *   - Comparison sampler for PCF in fragment shader
 *   - ShadowEntry SSBO (80 bytes/entry: mat4 VP + bias params)
 */

import { mat4, vec3 } from 'gl-matrix';
import type { Light } from './scene';

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------

export const SHADOW_MAP_SIZE = 2048;
export const MAX_SHADOW_LAYERS = 8;
export const SHADOW_ENTRY_SIZE = 80; // bytes per shadow entry (mat4 + 4 floats)

// Minimal depth-only vertex shader (transforms position by lightVP)
const SHADOW_VERT_WGSL = `
struct ShadowUniforms {
  lightVP: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) tangent: vec4<f32>,
};

@vertex
fn main(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shadow.lightVP * vec4<f32>(input.position, 1.0);
}
`;

// --------------------------------------------------------------------------
// Shadow pass manager
// --------------------------------------------------------------------------

export interface ShadowResources {
  /** Depth texture array for shadow maps. */
  depthTexture: GPUTexture;
  /** Per-layer texture views (for render attachments). */
  layerViews: GPUTextureView[];
  /** Full array view (for shader sampling). */
  arrayView: GPUTextureView;
  /** Comparison sampler for PCF. */
  comparisonSampler: GPUSampler;
  /** Shadow VP matrices + bias params, packed as Float32Array. */
  shadowEntries: Float32Array;
  /** GPU buffer for shadow entries SSBO. */
  shadowBuffer: GPUBuffer;
  /** Shadow render pipeline (depth-only). */
  pipeline: GPURenderPipeline;
  /** Bind group layout for the shadow uniform. */
  bindGroupLayout: GPUBindGroupLayout;
  /** Per-layer uniform buffers (lightVP). */
  uniformBuffers: GPUBuffer[];
  /** Per-layer bind groups. */
  bindGroups: GPUBindGroup[];
  /** Number of active shadow layers. */
  activeLayers: number;
}

/**
 * Create shadow map resources.
 */
export function createShadowResources(device: GPUDevice): ShadowResources {
  // Depth texture array
  const depthTexture = device.createTexture({
    size: [SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, MAX_SHADOW_LAYERS],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    label: 'shadow_depth_array',
  });

  // Per-layer views for rendering
  const layerViews: GPUTextureView[] = [];
  for (let i = 0; i < MAX_SHADOW_LAYERS; i++) {
    layerViews.push(depthTexture.createView({
      dimension: '2d',
      baseArrayLayer: i,
      arrayLayerCount: 1,
      label: `shadow_layer_${i}`,
    }));
  }

  // Full array view for shader sampling
  const arrayView = depthTexture.createView({
    dimension: '2d-array',
    label: 'shadow_array_view',
  });

  // Comparison sampler for PCF
  const comparisonSampler = device.createSampler({
    compare: 'less',
    magFilter: 'linear',
    minFilter: 'linear',
    label: 'shadow_comparison_sampler',
  });

  // Shadow entries buffer (SSBO)
  const shadowEntries = new Float32Array(MAX_SHADOW_LAYERS * (SHADOW_ENTRY_SIZE / 4));
  const shadowBuffer = device.createBuffer({
    size: MAX_SHADOW_LAYERS * SHADOW_ENTRY_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'shadow_entries_ssbo',
  });

  // Shadow render pipeline (depth-only)
  const shadowModule = device.createShaderModule({
    code: SHADOW_VERT_WGSL,
    label: 'shadow_vert',
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'uniform' },
    }],
    label: 'shadow_bgl',
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
    label: 'shadow_pipeline_layout',
  });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shadowModule,
      entryPoint: 'main',
      buffers: [{
        arrayStride: 48, // 12 floats × 4 bytes (pos3 + nor3 + uv2 + tan4)
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },   // position
          { shaderLocation: 1, offset: 12, format: 'float32x3' },  // normal
          { shaderLocation: 2, offset: 24, format: 'float32x2' },  // uv
          { shaderLocation: 3, offset: 32, format: 'float32x4' },  // tangent
        ],
      }],
    },
    // No fragment shader — depth-only
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
      frontFace: 'ccw',
    },
    depthStencil: {
      format: 'depth32float',
      depthWriteEnabled: true,
      depthCompare: 'less',
      depthBias: 2,
      depthBiasSlopeScale: 1.5,
    },
    label: 'shadow_pipeline',
  });

  // Per-layer uniforms + bind groups
  const uniformBuffers: GPUBuffer[] = [];
  const bindGroups: GPUBindGroup[] = [];
  for (let i = 0; i < MAX_SHADOW_LAYERS; i++) {
    const buf = device.createBuffer({
      size: 64, // mat4
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: `shadow_uniform_${i}`,
    });
    uniformBuffers.push(buf);
    bindGroups.push(device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: buf } }],
      label: `shadow_bg_${i}`,
    }));
  }

  return {
    depthTexture,
    layerViews,
    arrayView,
    comparisonSampler,
    shadowEntries,
    shadowBuffer,
    pipeline,
    bindGroupLayout,
    uniformBuffers,
    bindGroups,
    activeLayers: 0,
  };
}

// --------------------------------------------------------------------------
// Shadow matrix computation
// --------------------------------------------------------------------------

/**
 * Compute shadow view-projection matrix for a directional light.
 * Uses an orthographic frustum sized to the scene bounds.
 */
export function computeDirectionalShadowMatrix(
  light: Light,
  boundsMin: [number, number, number],
  boundsMax: [number, number, number],
): mat4 {
  const center = vec3.fromValues(
    (boundsMin[0] + boundsMax[0]) * 0.5,
    (boundsMin[1] + boundsMax[1]) * 0.5,
    (boundsMin[2] + boundsMax[2]) * 0.5,
  );

  const dx = boundsMax[0] - boundsMin[0];
  const dy = boundsMax[1] - boundsMin[1];
  const dz = boundsMax[2] - boundsMin[2];
  const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5;

  const dir = vec3.fromValues(light.direction[0], light.direction[1], light.direction[2]);
  vec3.normalize(dir, dir);

  const lightPos = vec3.create();
  vec3.scaleAndAdd(lightPos, center, dir, -radius * 2);

  const lightView = mat4.create();
  mat4.lookAt(lightView, lightPos, center, [0, 1, 0]);

  const lightProj = mat4.create();
  mat4.ortho(lightProj, -radius, radius, -radius, radius, 0.1, radius * 4);

  const lightVP = mat4.create();
  mat4.multiply(lightVP, lightProj, lightView);
  return lightVP;
}

/**
 * Compute shadow view-projection matrix for a spot light.
 */
export function computeSpotShadowMatrix(light: Light): mat4 {
  const lightView = mat4.create();
  const pos = vec3.fromValues(light.position[0], light.position[1], light.position[2]);
  const dir = vec3.fromValues(light.direction[0], light.direction[1], light.direction[2]);
  const target = vec3.create();
  vec3.add(target, pos, dir);
  mat4.lookAt(lightView, pos, target, [0, 1, 0]);

  const lightProj = mat4.create();
  const fov = light.outerCone * 2;
  mat4.perspective(lightProj, fov, 1.0, 0.1, light.range > 0 ? light.range : 100);

  const lightVP = mat4.create();
  mat4.multiply(lightVP, lightProj, lightView);
  return lightVP;
}

/**
 * Update shadow resources for the current scene lights.
 * Returns the number of active shadow layers.
 */
export function updateShadowMatrices(
  device: GPUDevice,
  resources: ShadowResources,
  lights: Light[],
  boundsMin: [number, number, number],
  boundsMax: [number, number, number],
): number {
  let layerIdx = 0;
  const entryFloats = SHADOW_ENTRY_SIZE / 4; // 20 floats per entry

  for (const light of lights) {
    if (layerIdx >= MAX_SHADOW_LAYERS) break;

    let shadowVP: mat4;
    if (light.type === 'directional') {
      shadowVP = computeDirectionalShadowMatrix(light, boundsMin, boundsMax);
    } else if (light.type === 'spot') {
      shadowVP = computeSpotShadowMatrix(light);
    } else {
      continue; // Point light shadows not yet implemented
    }

    // Write lightVP to per-layer uniform buffer
    const vpData = new Float32Array(shadowVP as unknown as ArrayLike<number>);
    device.queue.writeBuffer(resources.uniformBuffers[layerIdx], 0, vpData);

    // Pack shadow entry: mat4 VP (16 floats) + bias + normalBias + resolution + lightSize
    const offset = layerIdx * entryFloats;
    resources.shadowEntries.set(vpData, offset);
    resources.shadowEntries[offset + 16] = 0.005;  // bias
    resources.shadowEntries[offset + 17] = 0.02;   // normalBias
    resources.shadowEntries[offset + 18] = SHADOW_MAP_SIZE; // resolution
    resources.shadowEntries[offset + 19] = 0.02;   // lightSize

    layerIdx++;
  }

  resources.activeLayers = layerIdx;

  // Upload shadow entries SSBO
  if (layerIdx > 0) {
    device.queue.writeBuffer(resources.shadowBuffer, 0, resources.shadowEntries.buffer);
  }

  return layerIdx;
}

/**
 * Record shadow render passes for all active layers.
 */
export function recordShadowPasses(
  encoder: GPUCommandEncoder,
  resources: ShadowResources,
  draws: Array<{ vertexBuffer: GPUBuffer; indexBuffer: GPUBuffer | null; indexCount: number; vertexCount: number }>,
): void {
  for (let layer = 0; layer < resources.activeLayers; layer++) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: resources.layerViews[layer],
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      label: `shadow_pass_${layer}`,
    });

    pass.setPipeline(resources.pipeline);
    pass.setBindGroup(0, resources.bindGroups[layer]);

    for (const draw of draws) {
      pass.setVertexBuffer(0, draw.vertexBuffer);
      if (draw.indexBuffer) {
        pass.setIndexBuffer(draw.indexBuffer, 'uint32');
        pass.drawIndexed(draw.indexCount);
      } else {
        pass.draw(draw.vertexCount);
      }
    }

    pass.end();
  }
}

/**
 * Clean up shadow resources.
 */
export function destroyShadowResources(resources: ShadowResources): void {
  resources.depthTexture.destroy();
  resources.shadowBuffer.destroy();
  for (const buf of resources.uniformBuffers) buf.destroy();
}
