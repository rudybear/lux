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

import { mat4 } from 'gl-matrix';
import { loadShader, type ReflectionJSON } from './shader-loader';
import { createReflectedComputePipeline } from './reflected-pipeline';

// Sort algorithm constants (must match GLSL shaders)
const HISTOGRAM_WG_SIZE = 256;
const TILE_SIZE = 3840;
const PREFIX_WG_SIZE = 1024;
const PREFIX_BLOCK_SIZE = 2048;
const RADIX = 256;

export interface SplatBuffers {
  positions: GPUBuffer;       // vec3, float32
  scales: GPUBuffer;          // vec3, float32
  rotations: GPUBuffer;       // vec4 quat, float32
  opacities: GPUBuffer;       // float32
  shCoeffs: GPUBuffer;        // float32, degree-dependent
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

  // GPU buffers
  private _sortRes: SortResources | null = null;
  private _projectedCenter: GPUBuffer | null = null;   // vec4: ndc_x, ndc_y, depth, radius
  private _projectedConic: GPUBuffer | null = null;     // vec4: inv_a, inv_b, inv_c, opacity
  private _projectedColor: GPUBuffer | null = null;     // vec4: r, g, b, opacity

  // Sort dispatch sizing
  private _histogramWorkgroups = 0;
  private _prefixPartitions = 0;
  private _totalHistogramEntries = 0;

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
    const histSize = this._totalHistogramEntries * 4;
    const partSize = Math.max(this._prefixPartitions, 2048) * 4;
    this._sortRes = {
      keysA: device.createBuffer({ size: n * 4, label: 'sort_keys_a', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      keysB: device.createBuffer({ size: n * 4, label: 'sort_keys_b', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      valsA: device.createBuffer({ size: n * 4, label: 'sort_vals_a', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      valsB: device.createBuffer({ size: n * 4, label: 'sort_vals_b', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      histogram: device.createBuffer({ size: histSize, label: 'sort_histogram', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      partitionSums: device.createBuffer({ size: partSize, label: 'sort_partition_sums', usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
      indirectDraw: device.createBuffer({
        size: 16, label: 'indirect_draw',  // 4 × uint32: vertexCount, instanceCount, firstVertex, firstInstance
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    };

    // ---- Push constant emulation buffers ----
    this._preprocessPushBuffer = device.createBuffer({
      size: 256, label: 'preprocess_push',
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._sortPushBuffer = device.createBuffer({
      size: 16, label: 'sort_push',  // num_elements (u32) + bit_offset (u32) or num_entries + pass_id
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ---- Load shaders and create pipelines ----
    await this._createPreprocessPipeline(shaderBasePath, splatBuffers);
    await this._createSortPipelines(shaderBasePath);
    await this._createRenderPipeline(shaderBasePath, colorFormat);

    // Write initial indirect draw buffer: 6 vertices (quad), 0 instances
    device.queue.writeBuffer(this._sortRes.indirectDraw, 0, new Uint32Array([6, 0, 0, 0]));
  }

  private async _createPreprocessPipeline(basePath: string, splats: SplatBuffers): Promise<void> {
    const device = this._device;
    try {
      const shader = await loadShader(device, `${basePath}.comp`);
      const { pipeline } = createReflectedComputePipeline(device, shader.module, shader.reflection);
      this._preprocessPipeline = pipeline;
    } catch {
      console.warn('Preprocess shader not available, splat rendering disabled');
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
      console.warn('Sort shaders not available, splat rendering disabled');
    }
  }

  private async _createRenderPipeline(basePath: string, colorFormat: GPUTextureFormat): Promise<void> {
    const device = this._device;
    try {
      const [vertShader, fragShader] = await Promise.all([
        loadShader(device, `${basePath}.vert`),
        loadShader(device, `${basePath}.frag`),
      ]);

      // Simple layout: set 0 = sorted SSBO data, set 1 = push emulated
      const dataLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // projected_center
          { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // projected_conic
          { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // projected_color
          { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // sorted_indices
        ],
      });
      const pushLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
      });
      const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [dataLayout, pushLayout] });

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
        // No depth test for alpha-blended splats (back-to-front sorted)
        label: 'splat_render_pipeline',
      });
    } catch {
      console.warn('Splat render shaders not available');
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
    if (!this._preprocessPipeline || !this._sort || !this._sortRes) return;

    const device = this._device;
    const n = this._numSplats;
    const sort = this._sort;
    const sr = this._sortRes;

    // ================================================================
    // Stage 1: Preprocess compute
    // ================================================================
    if (this._preprocessPipeline && this._preprocessBindGroup) {
      // Update preprocess push constants (view/proj matrices, viewport, etc.)
      const pushData = new ArrayBuffer(256);
      const f32 = new Float32Array(pushData);
      f32.set(viewMatrix as unknown as Float32Array, 0);  // offset 0: view_matrix (64 bytes)
      f32.set(projMatrix as unknown as Float32Array, 16); // offset 64: proj_matrix (64 bytes)
      f32[32] = viewport[0];  // width
      f32[33] = viewport[1];  // height
      f32[34] = n;            // num_splats
      device.queue.writeBuffer(this._preprocessPushBuffer!, 0, pushData);

      const pass = encoder.beginComputePass({ label: 'preprocess' });
      pass.setPipeline(this._preprocessPipeline);
      pass.setBindGroup(0, this._preprocessBindGroup);
      if (this._preprocessPushBindGroup) pass.setBindGroup(1, this._preprocessPushBindGroup);
      pass.dispatchWorkgroups(Math.ceil(n / 256));
      pass.end();
    }

    // ================================================================
    // Stage 2: GPU Radix Sort — 4 bit-passes (8 bits each = 32-bit keys)
    // ================================================================
    // Each bit-pass: histogram → prefix_sum (3 dispatches) → scatter
    // Ping-pong: pass 0 A→B, pass 1 B→A, pass 2 A→B, pass 3 B→A → result in A
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

    // After 4 passes (even count), result is back in keysA/valsA

    // ================================================================
    // Stage 3: Instanced quad draw with alpha blending
    // ================================================================
    if (this._renderPipeline && this._renderBindGroup) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: targetView,
          loadOp: 'load',
          storeOp: 'store',
        }],
        label: 'splat_render',
      });

      pass.setPipeline(this._renderPipeline);
      pass.setBindGroup(0, this._renderBindGroup);
      // Use indirect draw — instanceCount written by preprocess compute
      pass.drawIndirect(sr.indirectDraw, 0);
      pass.end();
    }
  }

  destroy(): void {
    this._projectedCenter?.destroy();
    this._projectedConic?.destroy();
    this._projectedColor?.destroy();
    this._preprocessPushBuffer?.destroy();
    this._sortPushBuffer?.destroy();

    if (this._sortRes) {
      Object.values(this._sortRes).forEach((buf: GPUBuffer) => buf.destroy());
    }
  }
}
