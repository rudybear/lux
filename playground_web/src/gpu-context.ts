/**
 * WebGPU device initialization and feature detection.
 */

export interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<GPUContext> {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported');
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) {
    throw new Error('No WebGPU adapter found');
  }

  // Request higher limits for Gaussian splatting (11 storage buffers in preprocess)
  const device = await adapter.requestDevice({
    requiredFeatures: [],
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(
        adapter.limits.maxStorageBuffersPerShaderStage, 12,
      ),
    },
  });

  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.message}`);
  });

  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error('Cannot get WebGPU canvas context');
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  return { adapter, device, context, format, canvas };
}

/** Resize depth texture and reconfigure swap chain on canvas resize. */
export function createDepthTexture(device: GPUDevice, width: number, height: number): GPUTexture {
  return device.createTexture({
    size: [width, height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}
