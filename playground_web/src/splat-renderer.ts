/**
 * Gaussian splatting renderer for WebGPU.
 *
 * 4-stage pipeline:
 *   1. Preprocess compute — transform, cull, project, SH evaluate
 *   2. GPU radix sort — 4 bit-passes × 3 dispatches each (histogram → prefix sum → scatter)
 *   3. Instanced quad draw — 6 vertices × visibleCount (via indirect draw)
 *   4. Alpha blending — premultiplied: src=ONE, dst=ONE_MINUS_SRC_ALPHA
 *
 * Push constants emulated as uniform buffers.
 *
 * Sort constants (matching shaders/radix_sort/*.comp):
 *   TILE_SIZE = 3840 (256 threads × 15 elements)
 *   RADIX = 256
 *   PREFIX_BLOCK_SIZE = 2048 (1024 threads × 2 elements)
 */

import { mat4, vec3 } from 'gl-matrix';
import { loadShader, type ReflectionJSON } from './shader-loader';
import { createReflectedComputePipeline } from './reflected-pipeline';

// Sort algorithm constants (must match GLSL shaders)
const HISTOGRAM_WG_SIZE = 256;
const TILE_SIZE = 3840;
const PREFIX_WG_SIZE = 1024;
const PREFIX_BLOCK_SIZE = 2048;
const RADIX = 256;

export interface SplatBuffers {
  positions: GPUBuffer;       // vec4 (x,y,z,1), float32
  scales: GPUBuffer;          // vec4 (x,y,z,0), float32
  rotations: GPUBuffer;       // vec4 quat, float32
  opacities: GPUBuffer;       // float32 (scalar per splat)
  shCoeffs: GPUBuffer;        // vec4 (x,y,z,0), float32, degree-dependent
  count: number;
  shDegree: number;
}

interface SortResources {
  keysA: GPUBuffer;
  keysB: GPUBuffer;
  valsA: GPUBuffer;
  valsB: GPUBuffer;
  histogram: GPUBuffer;
  partitionSums: GPUBuffer;
  indirectDraw: GPUBuffer;
}

interface SortPipelines {
  histogram: GPUComputePipeline;
  histogramLayout: GPUBindGroupLayout;
  prefix: GPUComputePipeline;
  prefixLayout: GPUBindGroupLayout;
  scatter: GPUComputePipeline;
  scatterLayout: GPUBindGroupLayout;
}

export class SplatRenderer {
  private _device: GPUDevice;
  private _numSplats: number;

  // Preprocess compute
  private _preprocessPipeline: GPUComputePipeline | null = null;
  private _preprocessBindGroup: GPUBindGroup | null = null;
  private _preprocessPushBuffer: GPUBuffer | null = null;
  private _preprocessPushBindGroup: GPUBindGroup | null = null;

  // Sort pipelines
  private _sort: SortPipelines | null = null;
  private _sortPushBuffer: GPUBuffer | null = null;

  // Render pipeline
  private _renderPipeline: GPURenderPipeline | null = null;
  private _renderBindGroup: GPUBindGroup | null = null;
  private _renderPushBuffer: GPUBuffer | null = null;
  private _renderPushBindGroup: GPUBindGroup | null = null;
  // Fragment shader may use a different push emulated set
  private _renderFragPushBindGroup: GPUBindGroup | null = null;

  // GPU buffers
  private _sortRes: SortResources | null = null;
  private _projectedCenter: GPUBuffer | null = null;   // vec4: ndc_x, ndc_y, depth, radius
  private _projectedConic: GPUBuffer | null = null;     // vec4: inv_a, inv_b, inv_c, opacity
  private _projectedColor: GPUBuffer | null = null;     // vec4: r, g, b, opacity

  // Sort dispatch sizing
  private _histogramWorkgroups = 0;
  private _prefixPartitions = 0;
  private _totalHistogramEntries = 0;

  // SH degree for push constant
  private _shDegree = 0;

  constructor(device: GPUDevice, numSplats: number) {
    this._device = device;
    this._numSplats = numSplats;
  }

  /**
   * Initialize pipelines and buffers from pre-compiled WGSL shaders.
   *
   * @param shaderBasePath Base path for shaders (e.g. 'shaders/gaussian_splat')
   * @param splatBuffers   Input splat data buffers
   * @param colorFormat    Swap chain color format
   */
  async init(
    shaderBasePath: string,
    splatBuffers: SplatBuffers,
    colorFormat: GPUTextureFormat,
  ): Promise<void> {
    const device = this._device;
    const n = this._numSplats;
    this._shDegree = splatBuffers.shDegree;

    // Compute dispatch sizes
    this._histogramWorkgroups = Math.ceil(n / TILE_SIZE);
    this._totalHistogramEntries = RADIX * this._histogramWorkgroups;
    this._prefixPartitions = Math.ceil(this._totalHistogramEntries / PREFIX_BLOCK_SIZE);

    // ---- Allocate projected splat buffers ----
    this._projectedCenter = device.createBuffer({
      size: n * 16, label: 'projected_center',
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._projectedConic = device.createBuffer({
      size: n * 16, label: 'projected_conic',
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._projectedColor = device.createBuffer({
      size: n * 16, label: 'projected_color',
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ---- Sort resources ----
    const histSize = Math.max(this._totalHistogramEntries * 4, 4);
    const partSize = Math.max(this._prefixPartitions, 2048) * 4;
    this._sortRes = {
      keysA: device.createBuffer({ size: n * 4, label: 'sort_keys_a', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      keysB: device.createBuffer({ size: n * 4, label: 'sort_keys_b', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      valsA: device.createBuffer({ size: n * 4, label: 'sort_vals_a', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      valsB: device.createBuffer({ size: n * 4, label: 'sort_vals_b', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      histogram: device.createBuffer({ size: histSize, label: 'sort_histogram', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      partitionSums: device.createBuffer({ size: partSize, label: 'sort_partition_sums', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      indirectDraw: device.createBuffer({
        size: 16, label: 'indirect_draw',  // 4 x uint32: vertexCount, instanceCount, firstVertex, firstInstance
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    };

    // ---- Push constant emulation buffers ----
    // Preprocess push: 176 bytes (view_matrix, proj_matrix, cam_pos, screen_size, total_splats, focal_x, focal_y, sh_degree)
    this._preprocessPushBuffer = device.createBuffer({
      size: 256, label: 'preprocess_push',
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._sortPushBuffer = device.createBuffer({
      size: 16, label: 'sort_push',  // num_elements (u32) + bit_offset (u32) or num_entries + pass_id
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Render push: 16 bytes (screen_size vec2, visible_count u32, alpha_cutoff f32)
    this._renderPushBuffer = device.createBuffer({
      size: 16, label: 'render_push',
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ---- Load shaders and create pipelines ----
    await this._createPreprocessPipeline(shaderBasePath, splatBuffers);
    await this._createSortPipelines(shaderBasePath);
    await this._createRenderPipeline(shaderBasePath, colorFormat);

    // Write initial indirect draw buffer: 6 vertices (quad), N instances (all splats), 0, 0
    // Since we don't have a visible count from preprocess, draw all splats
    device.queue.writeBuffer(this._sortRes.indirectDraw, 0, new Uint32Array([6, n, 0, 0]));
  }

  private async _createPreprocessPipeline(basePath: string, splats: SplatBuffers): Promise<void> {
    const device = this._device;
    const n = this._numSplats;
    const sr = this._sortRes!;

    try {
      const shader = await loadShader(device, `${basePath}.comp`);

      // The compute shader layout from reflection:
      // group(0): bindings 0-10 = splat_pos, splat_rot, splat_scale, splat_opacity, splat_sh0,
      //           projected_center, projected_conic, projected_color, sort_keys, sorted_indices, visible_count
      // group(1): binding 0 = push uniform

      // Create bind group layout for group(0) — all storage buffers (read_write in WGSL)
      const dataLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // splat_pos
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // splat_rot
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // splat_scale
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // splat_opacity
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // splat_sh0
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // projected_center
          { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // projected_conic
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // projected_color
          { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // sort_keys
          { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // sorted_indices
          { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // visible_count
        ],
        label: 'preprocess_data_layout',
      });

      const pushLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
        label: 'preprocess_push_layout',
      });

      const pipeLayout = device.createPipelineLayout({
        bindGroupLayouts: [dataLayout, pushLayout],
        label: 'preprocess_pipeline_layout',
      });

      this._preprocessPipeline = device.createComputePipeline({
        layout: pipeLayout,
        compute: { module: shader.module, entryPoint: 'main' },
        label: 'preprocess_pipeline',
      });

      // visible_count buffer — reuse sort_vals_a as a scratch buffer for visible_count
      // (visible_count is declared in shader but not actually written in this variant)
      const visibleCountBuf = device.createBuffer({
        size: Math.max(n * 4, 4), label: 'visible_count_buf',
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // Create bind group for group(0) — wire splat input buffers + projected output buffers
      this._preprocessBindGroup = device.createBindGroup({
        layout: dataLayout,
        entries: [
          { binding: 0, resource: { buffer: splats.positions } },
          { binding: 1, resource: { buffer: splats.rotations } },
          { binding: 2, resource: { buffer: splats.scales } },
          { binding: 3, resource: { buffer: splats.opacities } },
          { binding: 4, resource: { buffer: splats.shCoeffs } },
          { binding: 5, resource: { buffer: this._projectedCenter! } },
          { binding: 6, resource: { buffer: this._projectedConic! } },
          { binding: 7, resource: { buffer: this._projectedColor! } },
          { binding: 8, resource: { buffer: sr.keysA } },
          { binding: 9, resource: { buffer: sr.valsA } },
          { binding: 10, resource: { buffer: visibleCountBuf } },
        ],
        label: 'preprocess_bind_group',
      });

      // Create bind group for group(1) — push uniform
      this._preprocessPushBindGroup = device.createBindGroup({
        layout: pushLayout,
        entries: [
          { binding: 0, resource: { buffer: this._preprocessPushBuffer! } },
        ],
        label: 'preprocess_push_bind_group',
      });

      console.log('Preprocess pipeline created successfully');
    } catch (e) {
      console.warn('Preprocess shader not available, splat rendering disabled:', e);
    }
  }

  private async _createSortPipelines(basePath: string): Promise<void> {
    const device = this._device;

    // Sort shaders are alongside the splat shaders
    const sortBase = basePath.replace(/[^/]+$/, 'radix_sort');

    try {
      const [histShader, prefixShader, scatterShader] = await Promise.all([
        loadShader(device, `${sortBase}/histogram`),
        loadShader(device, `${sortBase}/prefix_sum`),
        loadShader(device, `${sortBase}/scatter`),
      ]);

      // Histogram: binding 0 = keys_in (read), binding 1 = histograms (write), push uniform
      const histLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const histPushLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
      });
      const histPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [histLayout, histPushLayout] });
      const histogram = device.createComputePipeline({
        layout: histPipeLayout, compute: { module: histShader.module, entryPoint: 'main' },
      });

      // Prefix sum: binding 0 = histograms (rw), binding 1 = partition_sums (rw), push uniform
      const prefixLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      const prefixPushLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
      });
      const prefixPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [prefixLayout, prefixPushLayout] });
      const prefix = device.createComputePipeline({
        layout: prefixPipeLayout, compute: { module: prefixShader.module, entryPoint: 'main' },
      });

      // Scatter: bindings 0-4 = keys_in, keys_out, vals_in, vals_out, histograms; push uniform
      const scatterLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
      });
      const scatterPushLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
      });
      const scatterPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [scatterLayout, scatterPushLayout] });
      const scatter = device.createComputePipeline({
        layout: scatterPipeLayout, compute: { module: scatterShader.module, entryPoint: 'main' },
      });

      this._sort = {
        histogram, histogramLayout: histLayout,
        prefix, prefixLayout,
        scatter, scatterLayout,
      };
    } catch {
      console.warn('Sort shaders not available, rendering without GPU sort (unsorted splats)');
    }
  }

  private async _createRenderPipeline(basePath: string, colorFormat: GPUTextureFormat): Promise<void> {
    const device = this._device;
    const sr = this._sortRes!;

    try {
      const [vertShader, fragShader] = await Promise.all([
        loadShader(device, `${basePath}.vert`),
        loadShader(device, `${basePath}.frag`),
      ]);

      // Layout from reflection:
      // Vertex:   group(0) = projected_center(0), projected_conic(1), projected_color(2), sorted_indices(3)
      //           group(1) = push uniform (screen_size, visible_count, alpha_cutoff)
      // Fragment: group(2) = push uniform (same fields as vertex push)

      const dataLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // projected_center
          { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // projected_conic
          { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // projected_color
          { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // sorted_indices
        ],
        label: 'render_data_layout',
      });

      // Vertex push emulated at set 1
      const vertPushLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
        label: 'render_vert_push_layout',
      });

      // Fragment push emulated at set 2
      const fragPushLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
        label: 'render_frag_push_layout',
      });

      const pipeLayout = device.createPipelineLayout({
        bindGroupLayouts: [dataLayout, vertPushLayout, fragPushLayout],
        label: 'render_pipeline_layout',
      });

      this._renderPipeline = device.createRenderPipeline({
        layout: pipeLayout,
        vertex: { module: vertShader.module, entryPoint: 'main' },
        fragment: {
          module: fragShader.module,
          entryPoint: 'main',
          targets: [{
            format: colorFormat,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
        // Depth read (no write) for hybrid splat+mesh: splats occluded by mesh geometry
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: false,
          depthCompare: 'less',
        },
        label: 'splat_render_pipeline',
      });

      // Create data bind group (set 0) with projected buffers + sorted indices
      // sorted_indices = valsA after sort (or identity if no sort)
      this._renderBindGroup = device.createBindGroup({
        layout: dataLayout,
        entries: [
          { binding: 0, resource: { buffer: this._projectedCenter! } },
          { binding: 1, resource: { buffer: this._projectedConic! } },
          { binding: 2, resource: { buffer: this._projectedColor! } },
          { binding: 3, resource: { buffer: sr.valsA } },  // sorted_indices
        ],
        label: 'render_data_bind_group',
      });

      // Vertex push bind group (set 1)
      this._renderPushBindGroup = device.createBindGroup({
        layout: vertPushLayout,
        entries: [
          { binding: 0, resource: { buffer: this._renderPushBuffer! } },
        ],
        label: 'render_vert_push_bind_group',
      });

      // Fragment push bind group (set 2) — same buffer
      this._renderFragPushBindGroup = device.createBindGroup({
        layout: fragPushLayout,
        entries: [
          { binding: 0, resource: { buffer: this._renderPushBuffer! } },
        ],
        label: 'render_frag_push_bind_group',
      });

      console.log('Splat render pipeline created successfully');
    } catch (e) {
      console.warn('Splat render shaders not available:', e);
    }
  }

  /**
   * Record all compute + render commands for one frame.
   */
  render(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    depthView: GPUTextureView,
    viewMatrix: mat4,
    projMatrix: mat4,
    viewport: [number, number],
  ): void {
    // We need at least the preprocess pipeline OR the render pipeline
    if (!this._preprocessPipeline && !this._renderPipeline) return;

    const device = this._device;
    const n = this._numSplats;
    const sr = this._sortRes;

    // ================================================================
    // Stage 1: Preprocess compute
    // ================================================================
    if (this._preprocessPipeline && this._preprocessBindGroup && this._preprocessPushBindGroup) {
      // Compute focal lengths from projection matrix
      // proj[0][0] = 2*near*focal_x / width (for standard perspective)
      // focal_x = proj[0][0] * width / 2, focal_y = proj[1][1] * height / 2
      const projF32 = projMatrix as unknown as Float32Array;
      const focalX = projF32[0] * viewport[0] * 0.5;
      const focalY = projF32[5] * viewport[1] * 0.5;

      // Compute camera position from inverse view matrix
      const invView = mat4.create();
      mat4.invert(invView, viewMatrix);
      const camPos = vec3.fromValues(
        (invView as Float32Array)[12],
        (invView as Float32Array)[13],
        (invView as Float32Array)[14],
      );

      // Update preprocess push constants matching the reflection layout:
      // offset 0:   view_matrix (mat4, 64 bytes)
      // offset 64:  proj_matrix (mat4, 64 bytes)
      // offset 128: cam_pos (vec3, 12 bytes)
      // offset 144: screen_size (vec2, 8 bytes) — note: vec3 pads to 16 bytes
      // offset 152: total_splats (uint, 4 bytes)
      // offset 156: focal_x (f32, 4 bytes)
      // offset 160: focal_y (f32, 4 bytes)
      // offset 164: sh_degree (int, 4 bytes)
      const pushData = new ArrayBuffer(256);
      const f32 = new Float32Array(pushData);
      const u32 = new Uint32Array(pushData);
      const i32 = new Int32Array(pushData);
      f32.set(viewMatrix as unknown as Float32Array, 0);    // offset 0: view_matrix
      f32.set(projMatrix as unknown as Float32Array, 16);   // offset 64: proj_matrix
      f32[32] = camPos[0];   // offset 128: cam_pos.x
      f32[33] = camPos[1];   // offset 132: cam_pos.y
      f32[34] = camPos[2];   // offset 136: cam_pos.z
      f32[36] = viewport[0]; // offset 144: screen_size.x
      f32[37] = viewport[1]; // offset 148: screen_size.y
      u32[38] = n;           // offset 152: total_splats
      f32[39] = focalX;      // offset 156: focal_x
      f32[40] = focalY;      // offset 160: focal_y
      i32[41] = this._shDegree; // offset 164: sh_degree
      device.queue.writeBuffer(this._preprocessPushBuffer!, 0, pushData);

      const pass = encoder.beginComputePass({ label: 'preprocess' });
      pass.setPipeline(this._preprocessPipeline);
      pass.setBindGroup(0, this._preprocessBindGroup);
      pass.setBindGroup(1, this._preprocessPushBindGroup);
      pass.dispatchWorkgroups(Math.ceil(n / 256));
      pass.end();
    }

    // ================================================================
    // Stage 2: GPU Radix Sort — 4 bit-passes (8 bits each = 32-bit keys)
    // ================================================================
    if (this._sort && sr) {
      const sort = this._sort;

      // Each bit-pass: histogram -> prefix_sum (3 dispatches) -> scatter
      // Ping-pong: pass 0 A->B, pass 1 B->A, pass 2 A->B, pass 3 B->A -> result in A
      for (let bitPass = 0; bitPass < 4; bitPass++) {
        const bitOffset = bitPass * 8;
        const keysIn = bitPass % 2 === 0 ? sr.keysA : sr.keysB;
        const keysOut = bitPass % 2 === 0 ? sr.keysB : sr.keysA;
        const valsIn = bitPass % 2 === 0 ? sr.valsA : sr.valsB;
        const valsOut = bitPass % 2 === 0 ? sr.valsB : sr.valsA;

        // ---- Histogram ----
        {
          device.queue.writeBuffer(this._sortPushBuffer!, 0, new Uint32Array([n, bitOffset]));

          const histBindGroup = device.createBindGroup({
            layout: sort.histogramLayout,
            entries: [
              { binding: 0, resource: { buffer: keysIn } },
              { binding: 1, resource: { buffer: sr.histogram } },
            ],
          });
          const pushBG = device.createBindGroup({
            layout: sort.histogram.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this._sortPushBuffer! } }],
          });

          const pass = encoder.beginComputePass({ label: `sort_histogram_bit${bitOffset}` });
          pass.setPipeline(sort.histogram);
          pass.setBindGroup(0, histBindGroup);
          pass.setBindGroup(1, pushBG);
          pass.dispatchWorkgroups(this._histogramWorkgroups);
          pass.end();
        }

        // ---- Prefix sum (3 sub-dispatches, each in own compute pass for barriers) ----
        const prefixBindGroup = device.createBindGroup({
          layout: sort.prefixLayout,
          entries: [
            { binding: 0, resource: { buffer: sr.histogram } },
            { binding: 1, resource: { buffer: sr.partitionSums } },
          ],
        });

        // Pass 0: local scan
        {
          device.queue.writeBuffer(this._sortPushBuffer!, 0, new Uint32Array([this._totalHistogramEntries, 0]));
          const pushBG = device.createBindGroup({
            layout: sort.prefix.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this._sortPushBuffer! } }],
          });
          const pass = encoder.beginComputePass({ label: `sort_prefix_local_bit${bitOffset}` });
          pass.setPipeline(sort.prefix);
          pass.setBindGroup(0, prefixBindGroup);
          pass.setBindGroup(1, pushBG);
          pass.dispatchWorkgroups(this._prefixPartitions);
          pass.end();
        }

        // Pass 1: spine scan
        {
          device.queue.writeBuffer(this._sortPushBuffer!, 0, new Uint32Array([this._prefixPartitions, 1]));
          const pushBG = device.createBindGroup({
            layout: sort.prefix.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this._sortPushBuffer! } }],
          });
          const pass = encoder.beginComputePass({ label: `sort_prefix_spine_bit${bitOffset}` });
          pass.setPipeline(sort.prefix);
          pass.setBindGroup(0, prefixBindGroup);
          pass.setBindGroup(1, pushBG);
          pass.dispatchWorkgroups(1);
          pass.end();
        }

        // Pass 2: propagate
        {
          device.queue.writeBuffer(this._sortPushBuffer!, 0, new Uint32Array([this._totalHistogramEntries, 2]));
          const pushBG = device.createBindGroup({
            layout: sort.prefix.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this._sortPushBuffer! } }],
          });
          const pass = encoder.beginComputePass({ label: `sort_prefix_propagate_bit${bitOffset}` });
          pass.setPipeline(sort.prefix);
          pass.setBindGroup(0, prefixBindGroup);
          pass.setBindGroup(1, pushBG);
          pass.dispatchWorkgroups(this._prefixPartitions);
          pass.end();
        }

        // ---- Scatter ----
        {
          device.queue.writeBuffer(this._sortPushBuffer!, 0, new Uint32Array([n, bitOffset]));

          const scatterBindGroup = device.createBindGroup({
            layout: sort.scatterLayout,
            entries: [
              { binding: 0, resource: { buffer: keysIn } },
              { binding: 1, resource: { buffer: keysOut } },
              { binding: 2, resource: { buffer: valsIn } },
              { binding: 3, resource: { buffer: valsOut } },
              { binding: 4, resource: { buffer: sr.histogram } },
            ],
          });
          const pushBG = device.createBindGroup({
            layout: sort.scatter.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this._sortPushBuffer! } }],
          });

          const pass = encoder.beginComputePass({ label: `sort_scatter_bit${bitOffset}` });
          pass.setPipeline(sort.scatter);
          pass.setBindGroup(0, scatterBindGroup);
          pass.setBindGroup(1, pushBG);
          pass.dispatchWorkgroups(this._histogramWorkgroups);
          pass.end();
        }
      }
    }

    // After 4 passes (even count), result is back in keysA/valsA

    // ================================================================
    // Stage 3: Instanced quad draw with alpha blending
    // ================================================================
    if (this._renderPipeline && this._renderBindGroup && this._renderPushBindGroup) {
      // Update render push constants: screen_size (vec2), visible_count (u32), alpha_cutoff (f32)
      const pushData = new ArrayBuffer(16);
      const f32 = new Float32Array(pushData);
      const u32 = new Uint32Array(pushData);
      f32[0] = viewport[0]; // screen_size.x
      f32[1] = viewport[1]; // screen_size.y
      u32[2] = n;           // visible_count (all splats, culled ones are transparent)
      f32[3] = 0.004;       // alpha_cutoff
      device.queue.writeBuffer(this._renderPushBuffer!, 0, pushData);

      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: targetView,
          loadOp: 'load',
          storeOp: 'store',
        }],
        // Depth test for hybrid splat+mesh: splats behind mesh geometry are discarded.
        // Splats don't write depth (they rely on sort order for mutual ordering).
        depthStencilAttachment: depthView ? {
          view: depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
          depthReadOnly: true,
        } : undefined,
        label: 'splat_render',
      });

      pass.setPipeline(this._renderPipeline);
      pass.setBindGroup(0, this._renderBindGroup);
      pass.setBindGroup(1, this._renderPushBindGroup);
      if (this._renderFragPushBindGroup) {
        pass.setBindGroup(2, this._renderFragPushBindGroup);
      }
      // Draw 6 vertices (instanced quad) x N splats
      pass.draw(6, n);
      pass.end();
    }
  }

  destroy(): void {
    this._projectedCenter?.destroy();
    this._projectedConic?.destroy();
    this._projectedColor?.destroy();
    this._preprocessPushBuffer?.destroy();
    this._sortPushBuffer?.destroy();
    this._renderPushBuffer?.destroy();

    if (this._sortRes) {
      Object.values(this._sortRes).forEach((buf: GPUBuffer) => buf.destroy());
    }
  }
}
