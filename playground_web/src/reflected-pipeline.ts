/**
 * Data-driven pipeline creation from Lux reflection JSON.
 *
 * Port of playground/reflected_pipeline.py — maps reflection metadata
 * to WebGPU bind group layouts, pipeline layouts, and render/compute pipelines.
 */

import type { ReflectionJSON, DescriptorBinding, PushEmulated } from './shader-loader';

// --------------------------------------------------------------------------
// Vertex format mapping (reflection format string → WebGPU)
// --------------------------------------------------------------------------

const FORMAT_MAP: Record<string, GPUVertexFormat> = {
  'R32_SFLOAT': 'float32',
  'R32G32_SFLOAT': 'float32x2',
  'R32G32B32_SFLOAT': 'float32x3',
  'R32G32B32A32_SFLOAT': 'float32x4',
  'R32_SINT': 'sint32',
  'R32G32_SINT': 'sint32x2',
  'R32G32B32_SINT': 'sint32x3',
  'R32G32B32A32_SINT': 'sint32x4',
  'R32_UINT': 'uint32',
  'R32G32_UINT': 'uint32x2',
  'R32G32B32_UINT': 'uint32x3',
  'R32G32B32A32_UINT': 'uint32x4',
};

// --------------------------------------------------------------------------
// Binding type → GPUBindGroupLayoutEntry
// --------------------------------------------------------------------------

function bindingLayoutEntry(
  b: DescriptorBinding,
  binding: number,
  visibility: GPUShaderStageFlags,
): GPUBindGroupLayoutEntry {
  const base: GPUBindGroupLayoutEntry = { binding, visibility };

  switch (b.type) {
    case 'uniform_buffer':
      return { ...base, buffer: { type: 'uniform' } };
    case 'storage_buffer':
      return { ...base, buffer: { type: 'read-only-storage' } };
    case 'sampler':
      return { ...base, sampler: { type: 'filtering' } };
    case 'sampled_image':
      return { ...base, texture: { sampleType: 'float', viewDimension: '2d' } };
    case 'sampled_cube_image':
      return { ...base, texture: { sampleType: 'float', viewDimension: 'cube' } };
    case 'storage_image':
      return { ...base, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' } };
    default:
      console.warn(`Unknown binding type: ${b.type}`);
      return { ...base, buffer: { type: 'uniform' } };
  }
}

function stageToVisibility(stageFlags: string[]): GPUShaderStageFlags {
  let vis: GPUShaderStageFlags = 0;
  for (const s of stageFlags) {
    if (s === 'vertex') vis |= GPUShaderStage.VERTEX;
    else if (s === 'fragment') vis |= GPUShaderStage.FRAGMENT;
    else if (s === 'compute') vis |= GPUShaderStage.COMPUTE;
  }
  return vis;
}

// --------------------------------------------------------------------------
// Merge descriptor sets across vertex + fragment reflections
// --------------------------------------------------------------------------

function mergeDescriptorSets(
  vertReflection: ReflectionJSON | null,
  fragReflection: ReflectionJSON | null,
): Map<number, DescriptorBinding[]> {
  const merged = new Map<number, DescriptorBinding[]>();

  function addBindings(refl: ReflectionJSON | null) {
    if (!refl) return;
    for (const [setStr, bindings] of Object.entries(refl.descriptor_sets)) {
      const setNum = parseInt(setStr, 10);
      if (!merged.has(setNum)) merged.set(setNum, []);
      const existing = merged.get(setNum)!;
      for (const b of bindings) {
        // Check for duplicate binding numbers (shared across stages)
        const dup = existing.find(e => e.binding === b.binding);
        if (dup) {
          // Merge stage_flags
          for (const sf of b.stage_flags) {
            if (!dup.stage_flags.includes(sf)) dup.stage_flags.push(sf);
          }
        } else {
          existing.push({ ...b });
        }
      }
    }
  }

  addBindings(vertReflection);
  addBindings(fragReflection);
  return merged;
}

// --------------------------------------------------------------------------
// ReflectedPipeline
// --------------------------------------------------------------------------

export interface PipelineResources {
  /** Resource name → GPUBuffer | GPUSampler | GPUTextureView */
  [name: string]: GPUBuffer | GPUSampler | GPUTextureView;
}

export class ReflectedPipeline {
  readonly renderPipeline: GPURenderPipeline;
  readonly pipelineLayout: GPUPipelineLayout;
  readonly bindGroupLayouts: GPUBindGroupLayout[];
  readonly vertReflection: ReflectionJSON | null;
  readonly fragReflection: ReflectionJSON | null;
  readonly pushEmulated: PushEmulated | null;
  private readonly _setBindingMap: Map<number, DescriptorBinding[]>;

  private constructor(
    renderPipeline: GPURenderPipeline,
    pipelineLayout: GPUPipelineLayout,
    bindGroupLayouts: GPUBindGroupLayout[],
    vertReflection: ReflectionJSON | null,
    fragReflection: ReflectionJSON | null,
    pushEmulated: PushEmulated | null,
    setBindingMap: Map<number, DescriptorBinding[]>,
  ) {
    this.renderPipeline = renderPipeline;
    this.pipelineLayout = pipelineLayout;
    this.bindGroupLayouts = bindGroupLayouts;
    this.vertReflection = vertReflection;
    this.fragReflection = fragReflection;
    this.pushEmulated = pushEmulated;
    this._setBindingMap = setBindingMap;
  }

  /**
   * Create a render pipeline from vertex + fragment reflection metadata.
   */
  static create(
    device: GPUDevice,
    vertModule: GPUShaderModule,
    fragModule: GPUShaderModule,
    vertReflection: ReflectionJSON,
    fragReflection: ReflectionJSON,
    colorFormat: GPUTextureFormat,
    depthFormat: GPUTextureFormat = 'depth24plus',
    cullMode: GPUCullMode = 'back',
    strideOverride = 0,
  ): ReflectedPipeline {
    const setBindings = mergeDescriptorSets(vertReflection, fragReflection);

    // Merge push_emulated from either reflection
    const pushEmulated = vertReflection.push_emulated ?? fragReflection.push_emulated ?? null;

    // Determine max set index (including push emulated set and any gaps)
    let maxSet = -1;
    for (const s of setBindings.keys()) maxSet = Math.max(maxSet, s);
    if (pushEmulated) maxSet = Math.max(maxSet, pushEmulated.set);

    // Create bind group layouts for each set (including empty ones for gaps)
    const bindGroupLayouts: GPUBindGroupLayout[] = [];
    for (let s = 0; s <= maxSet; s++) {
      const bindings = setBindings.get(s);
      if (bindings && bindings.length > 0) {
        const entries = bindings.map(b =>
          bindingLayoutEntry(b, b.binding, stageToVisibility(b.stage_flags)),
        );
        bindGroupLayouts.push(device.createBindGroupLayout({ entries, label: `set${s}` }));
      } else if (pushEmulated && s === pushEmulated.set) {
        // Push constant emulation uniform
        bindGroupLayouts.push(device.createBindGroupLayout({
          entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          }],
          label: `set${s}_push_emulated`,
        }));
      } else {
        // Empty layout for gap
        bindGroupLayouts.push(device.createBindGroupLayout({ entries: [], label: `set${s}_empty` }));
      }
    }

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts,
      label: 'reflected_pipeline_layout',
    });

    // Vertex buffer layout from reflection
    const vertexBuffers: GPUVertexBufferLayout[] = [];
    if (vertReflection.vertex_attributes.length > 0) {
      const attrs: GPUVertexAttribute[] = vertReflection.vertex_attributes.map(a => ({
        shaderLocation: a.location,
        offset: a.offset,
        format: FORMAT_MAP[a.format] ?? 'float32x4',
      }));
      vertexBuffers.push({
        arrayStride: strideOverride || vertReflection.vertex_stride,
        stepMode: 'vertex',
        attributes: attrs,
      });
    }

    const renderPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: vertModule,
        entryPoint: 'main',
        buffers: vertexBuffers,
      },
      fragment: {
        module: fragModule,
        entryPoint: 'main',
        targets: [{ format: colorFormat }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode,
        frontFace: 'ccw',
      },
      depthStencil: {
        format: depthFormat,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      label: 'reflected_render_pipeline',
    });

    return new ReflectedPipeline(
      renderPipeline, pipelineLayout, bindGroupLayouts,
      vertReflection, fragReflection, pushEmulated, setBindings,
    );
  }

  /**
   * Create a fullscreen pipeline (no vertex buffers, no depth test).
   */
  static createFullscreen(
    device: GPUDevice,
    vertModule: GPUShaderModule,
    fragModule: GPUShaderModule,
    fragReflection: ReflectionJSON,
    colorFormat: GPUTextureFormat,
  ): ReflectedPipeline {
    const setBindings = mergeDescriptorSets(null, fragReflection);
    const pushEmulated = fragReflection.push_emulated ?? null;

    let maxSet = -1;
    for (const s of setBindings.keys()) maxSet = Math.max(maxSet, s);
    if (pushEmulated) maxSet = Math.max(maxSet, pushEmulated.set);

    const bindGroupLayouts: GPUBindGroupLayout[] = [];
    for (let s = 0; s <= maxSet; s++) {
      const bindings = setBindings.get(s);
      if (bindings && bindings.length > 0) {
        const entries = bindings.map(b =>
          bindingLayoutEntry(b, b.binding, stageToVisibility(b.stage_flags)),
        );
        bindGroupLayouts.push(device.createBindGroupLayout({ entries, label: `set${s}` }));
      } else if (pushEmulated && s === pushEmulated.set) {
        bindGroupLayouts.push(device.createBindGroupLayout({
          entries: [{
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          }],
          label: `set${s}_push_emulated`,
        }));
      } else {
        bindGroupLayouts.push(device.createBindGroupLayout({ entries: [], label: `set${s}_empty` }));
      }
    }

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

    const renderPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: vertModule, entryPoint: 'main' },
      fragment: {
        module: fragModule,
        entryPoint: 'main',
        targets: [{ format: colorFormat }],
      },
      primitive: { topology: 'triangle-list' },
      label: 'fullscreen_pipeline',
    });

    return new ReflectedPipeline(
      renderPipeline, pipelineLayout, bindGroupLayouts,
      null, fragReflection, pushEmulated, setBindings,
    );
  }

  /**
   * Create bind groups from a name → resource mapping.
   */
  createBindGroups(
    device: GPUDevice,
    resources: PipelineResources,
  ): GPUBindGroup[] {
    const groups: GPUBindGroup[] = [];

    for (let s = 0; s < this.bindGroupLayouts.length; s++) {
      // Push emulated bind group
      if (this.pushEmulated && s === this.pushEmulated.set) {
        const buf = resources['_push_emulated'] as GPUBuffer;
        if (buf) {
          groups.push(device.createBindGroup({
            layout: this.bindGroupLayouts[s],
            entries: [{ binding: 0, resource: { buffer: buf } }],
          }));
          continue;
        }
      }

      const bindings = this._setBindingMap.get(s);
      if (!bindings || bindings.length === 0) {
        groups.push(device.createBindGroup({
          layout: this.bindGroupLayouts[s],
          entries: [],
        }));
        continue;
      }

      const entries: GPUBindGroupEntry[] = [];
      for (const b of bindings) {
        const res = resources[b.name];
        if (!res) {
          console.warn(`Missing resource for binding '${b.name}' (set=${s}, binding=${b.binding})`);
          continue;
        }

        if (b.type === 'uniform_buffer' || b.type === 'storage_buffer') {
          entries.push({ binding: b.binding, resource: { buffer: res as GPUBuffer } });
        } else if (b.type === 'sampler') {
          entries.push({ binding: b.binding, resource: res as GPUSampler });
        } else if (b.type === 'sampled_image' || b.type === 'sampled_cube_image') {
          entries.push({ binding: b.binding, resource: res as GPUTextureView });
        } else if (b.type === 'storage_image') {
          entries.push({ binding: b.binding, resource: res as GPUTextureView });
        }
      }

      groups.push(device.createBindGroup({
        layout: this.bindGroupLayouts[s],
        entries,
        label: `bind_group_set${s}`,
      }));
    }

    return groups;
  }
}

// --------------------------------------------------------------------------
// Compute pipeline helper
// --------------------------------------------------------------------------

export function createReflectedComputePipeline(
  device: GPUDevice,
  module: GPUShaderModule,
  reflection: ReflectionJSON,
): { pipeline: GPUComputePipeline; layout: GPUPipelineLayout; bindGroupLayouts: GPUBindGroupLayout[] } {
  const setBindings = mergeDescriptorSets(null, reflection);
  const pushEmulated = reflection.push_emulated ?? null;

  let maxSet = -1;
  for (const s of setBindings.keys()) maxSet = Math.max(maxSet, s);
  if (pushEmulated) maxSet = Math.max(maxSet, pushEmulated.set);

  const bindGroupLayouts: GPUBindGroupLayout[] = [];
  for (let s = 0; s <= maxSet; s++) {
    const bindings = setBindings.get(s);
    if (bindings && bindings.length > 0) {
      const entries = bindings.map(b =>
        bindingLayoutEntry(b, b.binding, GPUShaderStage.COMPUTE),
      );
      bindGroupLayouts.push(device.createBindGroupLayout({ entries }));
    } else if (pushEmulated && s === pushEmulated.set) {
      bindGroupLayouts.push(device.createBindGroupLayout({
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        }],
      }));
    } else {
      bindGroupLayouts.push(device.createBindGroupLayout({ entries: [] }));
    }
  }

  const layout = device.createPipelineLayout({ bindGroupLayouts });
  const pipeline = device.createComputePipeline({
    layout,
    compute: { module, entryPoint: 'main' },
  });

  return { pipeline, layout, bindGroupLayouts };
}
