/**
 * Lux Playground -- WebGPU entry point.
 *
 * Loads DamagedHelmet as default scene (fetched from KhronosGroup glTF samples),
 * with fallback to a PBR sphere if the model can't be loaded.
 */

import { initWebGPU } from './gpu-context';
import { loadShader, type ReflectionJSON } from './shader-loader';
import { ReflectedPipeline } from './reflected-pipeline';
import { RenderEngine, type RenderState, type DrawCall, type UniformBufferBinding } from './render-engine';
import { UI, observeCanvasResize } from './ui';
import { loadGLB, loadGLBFromBuffer } from './gltf-loader';
import type { Scene } from './scene';
import {
  FALLBACK_VERT_WGSL, FALLBACK_FRAG_WGSL,
  FALLBACK_VERT_REFLECTION, FALLBACK_FRAG_REFLECTION,
} from './fallback-shaders';
import { createGltfPbrPipeline, buildGltfSceneDrawCalls } from './gltf-pbr';
import { loadIBL } from './ibl-loader';
import { generateProceduralIBL, type ProceduralIBL } from './procedural-ibl';

const DAMAGED_HELMET_URL =
  'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb';

// --------------------------------------------------------------------------
// Geometry helpers
// --------------------------------------------------------------------------

function createSphereBuffers(device: GPUDevice, segments = 32, rings = 16): DrawCall {
  const verts: number[] = [];
  const indices: number[] = [];

  for (let r = 0; r <= rings; r++) {
    const phi = (r / rings) * Math.PI;
    for (let s = 0; s <= segments; s++) {
      const theta = (s / segments) * Math.PI * 2;
      const x = Math.sin(phi) * Math.cos(theta);
      const y = Math.cos(phi);
      const z = Math.sin(phi) * Math.sin(theta);
      const u = s / segments;
      const v = r / rings;
      verts.push(x, y, z, x, y, z, u, v);
    }
  }

  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < segments; s++) {
      const a = r * (segments + 1) + s;
      const b = a + segments + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }

  const vertData = new Float32Array(verts);
  const idxData = new Uint32Array(indices);

  const vertexBuffer = device.createBuffer({
    size: vertData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertData);

  const indexBuffer = device.createBuffer({
    size: idxData.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, idxData);

  return { vertexBuffer, indexBuffer, indexCount: idxData.length, vertexCount: 0 };
}

// --------------------------------------------------------------------------
// Fallback sphere pipeline
// --------------------------------------------------------------------------

function createFallbackPipeline(device: GPUDevice, format: GPUTextureFormat): ReflectedPipeline {
  const vertModule = device.createShaderModule({ code: FALLBACK_VERT_WGSL, label: 'fallback_vert' });
  const fragModule = device.createShaderModule({ code: FALLBACK_FRAG_WGSL, label: 'fallback_frag' });
  return ReflectedPipeline.create(
    device, vertModule, fragModule,
    FALLBACK_VERT_REFLECTION as unknown as ReflectionJSON,
    FALLBACK_FRAG_REFLECTION as unknown as ReflectionJSON,
    format,
  );
}

function buildSphereRenderState(
  device: GPUDevice,
  pipeline: ReflectedPipeline,
): RenderState {
  let pushBuffer: GPUBuffer | null = null;
  if (pipeline.pushEmulated) {
    pushBuffer = device.createBuffer({
      size: Math.max(pipeline.pushEmulated.size, 16),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'push_emulated',
    });
  }

  // Collect all uniform buffers from reflection
  const uniformBuffers: UniformBufferBinding[] = [];
  const resources: Record<string, GPUBuffer | GPUSampler | GPUTextureView> = {};
  for (const r of [pipeline.vertReflection, pipeline.fragReflection]) {
    if (!r) continue;
    for (const bindings of Object.values(r.descriptor_sets)) {
      for (const b of bindings) {
        if (b.type === 'uniform_buffer' && !resources[b.name]) {
          const size = b.size ?? 160;
          const buffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: b.name,
          });
          resources[b.name] = buffer;
          uniformBuffers.push({
            buffer,
            fields: (b.fields ?? []) as UniformBufferBinding['fields'],
            size,
          });
        }
      }
    }
  }

  if (pushBuffer) {
    resources['_push_emulated'] = pushBuffer;
  }

  const bindGroups = pipeline.createBindGroups(device, resources);
  const sphere = createSphereBuffers(device);

  return { pipeline, bindGroups, draws: [sphere], pushBuffer, uniformBuffers };
}

// --------------------------------------------------------------------------
// glTF scene render state
// --------------------------------------------------------------------------

async function buildGltfRenderState(
  device: GPUDevice,
  format: GPUTextureFormat,
  scene: Scene,
  ibl: ProceduralIBL,
): Promise<RenderState> {
  const pipeline = await createGltfPbrPipeline(device, format);
  const { draws, uniformBuffers, mvpBindGroup, storageBuffers } = buildGltfSceneDrawCalls(device, pipeline, scene, ibl);

  return {
    pipeline,
    bindGroups: [mvpBindGroup],
    draws,
    pushBuffer: null,
    uniformBuffers,
    storageBuffers,
  };
}

// --------------------------------------------------------------------------
// Pre-compiled shader pipeline (optional)
// --------------------------------------------------------------------------

async function loadShaderPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
  sceneName: string,
): Promise<ReflectedPipeline | null> {
  try {
    const vertShader = await loadShader(device, `shaders/${sceneName}.vert`);
    const fragShader = await loadShader(device, `shaders/${sceneName}.frag`);
    return ReflectedPipeline.create(
      device, vertShader.module, fragShader.module,
      vertShader.reflection, fragShader.reflection, format,
    );
  } catch {
    return null;
  }
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

async function main() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const noWebGPU = document.getElementById('no-webgpu')!;

  if (!navigator.gpu) {
    noWebGPU.style.display = 'block';
    return;
  }

  let gpu;
  try {
    gpu = await initWebGPU(canvas);
  } catch (e) {
    noWebGPU.style.display = 'block';
    console.error(e);
    return;
  }

  const { device, format } = gpu;
  const engine = new RenderEngine(gpu);
  const ui = new UI();

  observeCanvasResize(canvas, (w, h) => {
    gpu.context.configure({ device, format, alphaMode: 'opaque' });
    engine.resize(w, h);
  });

  engine.camera.attach(canvas);

  // Load pre-computed IBL cubemaps (same assets as C++ viewer), fall back to procedural
  let ibl: ProceduralIBL;
  try {
    ibl = await loadIBL(device, 'assets/ibl');
    console.log('Loaded pre-computed IBL (Pisa)');
  } catch (e) {
    console.warn('Pre-computed IBL not available, generating procedural IBL:', e);
    ibl = generateProceduralIBL(device);
  }

  // Load DamagedHelmet as default scene
  let currentScene: string = 'damaged_helmet';
  try {
    console.log('Loading DamagedHelmet...');
    const scene = await loadGLB(device, DAMAGED_HELMET_URL);
    console.log(`Loaded ${scene.meshes.length} meshes, ${scene.materials.length} materials, ${scene.lights.length} lights, ${scene.drawRanges.length} draw ranges`);
    const state = await buildGltfRenderState(device, format, scene, ibl);
    engine.setRenderState(state);
    engine.camera.frameScene(scene.boundsMin, scene.boundsMax);
    engine.lightCount = Math.max(1, scene.lights.length);
    console.log('DamagedHelmet loaded successfully');
  } catch (e) {
    console.warn('Failed to load DamagedHelmet, falling back to PBR sphere:', e);
    currentScene = 'pbr_sphere';
    const pipeline = createFallbackPipeline(device, format);
    const state = buildSphereRenderState(device, pipeline);
    engine.setRenderState(state);
  }

  // UI callbacks
  ui.onSceneChange = async (sceneName) => {
    if (sceneName === currentScene) return;
    currentScene = sceneName;

    if (sceneName === 'damaged_helmet') {
      try {
        const scene = await loadGLB(device, DAMAGED_HELMET_URL);
        const state = await buildGltfRenderState(device, format, scene, ibl);
        engine.setRenderState(state);
        engine.camera.frameScene(scene.boundsMin, scene.boundsMax);
        engine.lightCount = Math.max(1, scene.lights.length);
      } catch (e) {
        console.error('Failed to load DamagedHelmet:', e);
      }
    } else if (sceneName === 'pbr_sphere') {
      const pipeline = createFallbackPipeline(device, format);
      const state = buildSphereRenderState(device, pipeline);
      engine.setRenderState(state);
      engine.camera.distance = 3.0;
    } else {
      // Try pre-compiled shaders, fall back to sphere
      const compiled = await loadShaderPipeline(device, format, sceneName);
      const pipeline = compiled ?? createFallbackPipeline(device, format);
      const state = buildSphereRenderState(device, pipeline);
      engine.setRenderState(state);
    }
  };

  ui.onScreenshot = () => {
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'lux_screenshot.png';
      a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  };

  ui.onFileDrop = async (buffer, name) => {
    console.log(`Loading dropped file: ${name}`);
    try {
      const glScene = await loadGLBFromBuffer(device, buffer);
      console.log(`Loaded ${glScene.meshes.length} meshes, ${glScene.materials.length} materials, ${glScene.lights.length} lights`);
      const state = await buildGltfRenderState(device, format, glScene, ibl);
      engine.setRenderState(state);
      engine.camera.frameScene(glScene.boundsMin, glScene.boundsMax);
      engine.lightCount = Math.max(1, glScene.lights.length);
      currentScene = 'custom';
    } catch (e) {
      console.error('Failed to load glTF:', e);
    }
  };

  // Render loop
  let lastTime = performance.now();
  function frame() {
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;

    engine.metallic = ui.state.metallic;
    engine.roughness = ui.state.roughness;
    engine.exposure = ui.state.exposure;
    engine.lightDirY = ui.state.lightDirY;

    engine.renderFrame(dt);
    ui.updateFPS(dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch(console.error);
