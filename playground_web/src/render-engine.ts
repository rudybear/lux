/**
 * Core render loop and command encoding.
 *
 * Supports:
 *   - Reflection-driven uniform updates (reads field layout from reflection JSON)
 *   - Multi-node glTF rendering with per-node transforms
 *   - Push constant emulation buffer writes
 *   - Fallback rendering (no pipeline — just clear)
 */

import { mat4, vec3 } from 'gl-matrix';
import type { GPUContext } from './gpu-context';
import { createDepthTexture } from './gpu-context';
import { OrbitCamera } from './camera';
import type { ReflectedPipeline } from './reflected-pipeline';

/** A single draw call within the render state. */
export interface DrawCall {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer | null;
  indexCount: number;
  vertexCount: number;
  /** Per-draw material bind group (set 1) for glTF rendering. */
  materialBindGroup?: GPUBindGroup;
  /** Per-draw MVP bind group (set 0) — if set, used instead of shared bindGroups[0]. */
  mvpBindGroup?: GPUBindGroup;
  /** Per-draw MVP uniform buffer for model matrix update. */
  mvpBuffer?: GPUBuffer;
  /** World transform (model matrix) for this draw call. */
  worldTransform?: Float32Array;
  /** Alpha mode for draw ordering (OPAQUE rendered first). */
  alphaMode?: 'OPAQUE' | 'BLEND' | 'MASK';
}

/** A uniform buffer with its field layout from reflection. */
export interface UniformBufferBinding {
  buffer: GPUBuffer;
  fields: FieldLayout[];
  size: number;
}

export interface RenderState {
  pipeline: ReflectedPipeline | null;
  bindGroups: GPUBindGroup[];
  draws: DrawCall[];
  pushBuffer: GPUBuffer | null;
  /** All uniform buffers to update per-frame, populated from reflection. */
  uniformBuffers: UniformBufferBinding[];
  /** Storage buffers that need per-frame updates (e.g. lights SSBO). */
  storageBuffers?: { buffer: GPUBuffer; data: Float32Array }[];
}

/** Field layout entry from reflection JSON. */
interface FieldLayout {
  name: string;
  type: string;
  offset: number;
  size: number;
}

// Byte sizes for writing mat4/vec types into buffers
const TYPE_FLOAT_COUNT: Record<string, number> = {
  scalar: 1, int: 1, uint: 1,
  vec2: 2, vec3: 3, vec4: 4,
  mat2: 8, mat3: 12, mat4: 16,
};

export class RenderEngine {
  readonly gpu: GPUContext;
  readonly camera: OrbitCamera;
  private _depthTexture: GPUTexture;
  private _width: number;
  private _height: number;
  private _renderState: RenderState | null = null;

  // Per-frame properties (set by UI)
  private _time = 0;
  metallic = 0.0;
  roughness = 0.5;
  exposure = 1.0;
  lightDirY = 0.7;
  lightCount = 1;

  constructor(gpu: GPUContext) {
    this.gpu = gpu;
    this.camera = new OrbitCamera();
    this._width = gpu.canvas.width;
    this._height = gpu.canvas.height;
    this._depthTexture = createDepthTexture(gpu.device, this._width, this._height);
  }

  get width(): number { return this._width; }
  get height(): number { return this._height; }

  /** Get a depth texture view for external renderers (e.g., splat renderer). */
  getDepthView(): GPUTextureView {
    return this._depthTexture.createView();
  }

  setRenderState(state: RenderState): void {
    this._renderState = state;
  }

  resize(width: number, height: number): void {
    if (width === this._width && height === this._height) return;
    this._width = width;
    this._height = height;
    this._depthTexture.destroy();
    this._depthTexture = createDepthTexture(this.gpu.device, width, height);
  }

  /** Render a single frame. Returns false if nothing to render. */
  renderFrame(dt: number): boolean {
    this._time += dt;

    const { device, context } = this.gpu;
    const state = this._renderState;

    const textureView = context.getCurrentTexture().createView();
    const encoder = device.createCommandEncoder();

    const depthView = this._depthTexture.createView();

    // No pipeline — just clear
    if (!state || !state.pipeline) {
      encoder.beginRenderPass({
        colorAttachments: [{
          view: textureView,
          clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
        },
      }).end();
      device.queue.submit([encoder.finish()]);
      return false;
    }

    // Update uniforms from reflection
    this._updateUniforms(state);

    // Update per-draw MVP buffers with current view/proj + each draw's model matrix
    const view = mat4.create();
    const proj = mat4.create();
    this.camera.getViewMatrix(view);
    this.camera.getProjectionMatrix(proj, this._width / this._height);

    for (const draw of state.draws) {
      if (draw.mvpBuffer && draw.worldTransform) {
        const data = new Float32Array(48); // 3x mat4 = 192 bytes
        data.set(draw.worldTransform, 0);         // model at offset 0
        data.set(view as Float32Array, 16);        // view at offset 64 bytes = float index 16
        data.set(proj as Float32Array, 32);        // proj at offset 128 bytes = float index 32
        device.queue.writeBuffer(draw.mvpBuffer, 0, data);
      }
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
      },
    });

    pass.setPipeline(state.pipeline.renderPipeline);

    for (let i = 0; i < state.bindGroups.length; i++) {
      pass.setBindGroup(i, state.bindGroups[i]);
    }

    // Multi-draw: iterate over all draw calls
    for (const draw of state.draws) {
      if (draw.mvpBindGroup) {
        pass.setBindGroup(0, draw.mvpBindGroup);
      }
      if (draw.materialBindGroup) {
        pass.setBindGroup(1, draw.materialBindGroup);
      }
      pass.setVertexBuffer(0, draw.vertexBuffer);
      if (draw.indexBuffer) {
        pass.setIndexBuffer(draw.indexBuffer, 'uint32');
        pass.drawIndexed(draw.indexCount);
      } else {
        pass.draw(draw.vertexCount);
      }
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
    return true;
  }

  // --------------------------------------------------------------------------
  // Reflection-driven uniform update
  // --------------------------------------------------------------------------

  private _updateUniforms(state: RenderState): void {
    const aspect = this._width / this._height;
    const view = mat4.create();
    const proj = mat4.create();
    this.camera.getViewMatrix(view);
    this.camera.getProjectionMatrix(proj, aspect);
    const eye = this.camera.getEyePosition();

    // Named values available for writing into push/uniform fields
    const namedValues = this._buildNamedValues(view, proj, eye);

    // Write push emulated buffer
    if (state.pushBuffer && state.pipeline?.pushEmulated) {
      const pe = state.pipeline.pushEmulated;
      const data = new ArrayBuffer(pe.size);
      this._writeFieldsFromReflection(new Float32Array(data), pe.fields, namedValues);
      this.gpu.device.queue.writeBuffer(state.pushBuffer, 0, data);
    }

    // Write all uniform buffers from reflection-driven bindings
    for (const ub of state.uniformBuffers) {
      const data = new ArrayBuffer(ub.size);
      this._writeFieldsFromReflection(new Float32Array(data), ub.fields, namedValues);
      this.gpu.device.queue.writeBuffer(ub.buffer, 0, data);
    }

    // Write storage buffers (e.g., lights SSBO)
    if (state.storageBuffers) {
      for (const sb of state.storageBuffers) {
        this.gpu.device.queue.writeBuffer(sb.buffer, 0, sb.data.buffer, sb.data.byteOffset, sb.data.byteLength);
      }
    }
  }

  /** Build a map of named values for reflection-driven field writes. */
  private _buildNamedValues(
    view: mat4,
    proj: mat4,
    eye: vec3,
  ): Map<string, number[]> {
    const values = new Map<string, number[]>();

    // Matrices
    values.set('view', Array.from(view as Float32Array));
    values.set('view_matrix', Array.from(view as Float32Array));
    values.set('proj', Array.from(proj as Float32Array));
    values.set('projection', Array.from(proj as Float32Array));
    values.set('proj_matrix', Array.from(proj as Float32Array));

    const mvp = mat4.create();
    mat4.multiply(mvp, proj, view);
    values.set('mvp', Array.from(mvp as Float32Array));
    values.set('view_proj', Array.from(mvp as Float32Array));

    const invView = mat4.create();
    mat4.invert(invView, view);
    values.set('inv_view', Array.from(invView as Float32Array));

    const invProj = mat4.create();
    mat4.invert(invProj, proj);
    values.set('inv_proj', Array.from(invProj as Float32Array));

    // Vectors
    values.set('camera_position', [eye[0], eye[1], eye[2]]);
    values.set('eye', [eye[0], eye[1], eye[2]]);
    values.set('view_pos', [eye[0], eye[1], eye[2]]);
    values.set('eye_position', [eye[0], eye[1], eye[2]]);

    // Light (exposure scales the light color as intensity multiplier)
    const len = Math.sqrt(0.25 + this.lightDirY * this.lightDirY + 0.09);
    values.set('light_dir', [0.5 / len, this.lightDirY / len, 0.3 / len]);
    values.set('light_direction', [0.5 / len, this.lightDirY / len, 0.3 / len]);
    const e = this.exposure;
    values.set('light_color', [e, e, e]);

    // Material
    values.set('metallic', [this.metallic]);
    values.set('roughness', [this.roughness]);
    values.set('exposure', [this.exposure]);

    // Screen
    values.set('resolution', [this._width, this._height]);
    values.set('screen_size', [this._width, this._height]);
    values.set('time', [this._time]);

    // Light count (for multi-light shaders)
    values.set('light_count', [this.lightCount]);

    // Identity model matrix
    values.set('model', Array.from(mat4.create() as Float32Array));
    values.set('model_matrix', Array.from(mat4.create() as Float32Array));

    // Normal matrix (inverse transpose of model-view 3x3, stored as mat4)
    const normalMatrix = mat4.create();
    mat4.invert(normalMatrix, view);
    mat4.transpose(normalMatrix, normalMatrix);
    values.set('normal_matrix', Array.from(normalMatrix as Float32Array));

    return values;
  }

  /** Write named fields into a Float32Array at their reflection offsets. */
  private _writeFieldsFromReflection(
    f32: Float32Array,
    fields: FieldLayout[],
    namedValues: Map<string, number[]>,
  ): void {
    const dv = new DataView(f32.buffer, f32.byteOffset, f32.byteLength);
    for (const field of fields) {
      const vals = namedValues.get(field.name);
      if (!vals) continue;
      const idx = field.offset / 4;
      const count = TYPE_FLOAT_COUNT[field.type] ?? vals.length;
      const isInt = field.type === 'int' || field.type === 'uint';
      for (let i = 0; i < Math.min(count, vals.length); i++) {
        if (isInt) {
          dv.setInt32(field.offset + i * 4, Math.round(vals[i]), true);
        } else {
          f32[idx + i] = vals[i];
        }
      }
    }
  }
}
