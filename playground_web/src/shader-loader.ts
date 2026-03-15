/**
 * Load pre-compiled WGSL shaders + reflection JSON from the server.
 */

export interface ReflectionJSON {
  version: number;
  source: string;
  stage: string;
  execution_model: string;
  inputs: Array<{ name: string; type: string; location: number }>;
  outputs: Array<{ name: string; type: string; location: number }>;
  descriptor_sets: Record<string, DescriptorBinding[]>;
  push_constants: PushConstantBlock[];
  push_emulated?: PushEmulated;
  wgsl_file?: string;
  vertex_attributes: VertexAttribute[];
  vertex_stride: number;
  compute?: { workgroup_size: number[] };
  gaussian_splatting?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DescriptorBinding {
  binding: number;
  type: string;
  name: string;
  fields?: Array<{ name: string; type: string; offset: number; size: number; default?: unknown }>;
  size?: number;
  element_type?: string;
  max_count?: number;
  stage_flags: string[];
}

export interface PushConstantBlock {
  name: string;
  fields: Array<{ name: string; type: string; offset: number; size: number }>;
  size: number;
  stage_flags: string[];
}

export interface PushEmulated {
  set: number;
  binding: number;
  fields: Array<{ name: string; type: string; offset: number; size: number }>;
  size: number;
}

export interface VertexAttribute {
  location: number;
  type: string;
  name: string;
  format: string;
  offset: number;
}

export interface LoadedShader {
  module: GPUShaderModule;
  reflection: ReflectionJSON;
}

export async function loadShader(
  device: GPUDevice,
  basePath: string,
): Promise<LoadedShader> {
  const [wgslResponse, jsonResponse] = await Promise.all([
    fetch(`${basePath}.wgsl`),
    fetch(`${basePath}.json`),
  ]);

  if (!wgslResponse.ok) throw new Error(`Failed to load ${basePath}.wgsl`);
  if (!jsonResponse.ok) throw new Error(`Failed to load ${basePath}.json`);

  const wgslCode = await wgslResponse.text();
  const reflection: ReflectionJSON = await jsonResponse.json();

  const module = device.createShaderModule({
    code: wgslCode,
    label: basePath,
  });

  return { module, reflection };
}
