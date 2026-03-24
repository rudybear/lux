/**
 * Lux Playground -- WebGPU entry point.
 *
 * Loads DamagedHelmet as default scene (fetched from KhronosGroup glTF samples),
 * with fallback to a PBR sphere if the model can't be loaded.
 */

import { mat4 } from 'gl-matrix';
import { initWebGPU, type GPUContext } from './gpu-context';
import { loadShader, type ReflectionJSON } from './shader-loader';
import { ReflectedPipeline } from './reflected-pipeline';
import { RenderEngine, type RenderState, type DrawCall, type UniformBufferBinding } from './render-engine';
import { UI, observeCanvasResize } from './ui';
import { loadGLB, loadGLBFromBuffer } from './gltf-loader';
import type { Scene, SplatData } from './scene';
import {
  FALLBACK_VERT_WGSL, FALLBACK_FRAG_WGSL,
  FALLBACK_VERT_REFLECTION, FALLBACK_FRAG_REFLECTION,
} from './fallback-shaders';
import { createGltfPbrPipeline, buildGltfSceneDrawCalls } from './gltf-pbr';
import { fillWebMaterialUBO } from './gltf-pbr';
import { loadIBL } from './ibl-loader';
import { generateProceduralIBL, type ProceduralIBL } from './procedural-ibl';
import { SplatRenderer, type SplatBuffers } from './splat-renderer';

const DAMAGED_HELMET_URL =
  'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb';

const LUIGI_URL = 'assets/luigi.glb';

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
// Splat buffer creation
// --------------------------------------------------------------------------

function createSplatBuffers(device: GPUDevice, data: SplatData): SplatBuffers {
  const makeBuffer = (arr: Float32Array, label: string) => {
    const buf = device.createBuffer({
      size: arr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label,
    });
    device.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return buf;
  };

  // Pack positions as vec4 (x,y,z,1) for the shader
  const pos4 = new Float32Array(data.numSplats * 4);
  for (let i = 0; i < data.numSplats; i++) {
    pos4[i * 4 + 0] = data.positions[i * 3 + 0];
    pos4[i * 4 + 1] = data.positions[i * 3 + 1];
    pos4[i * 4 + 2] = data.positions[i * 3 + 2];
    pos4[i * 4 + 3] = 1.0;
  }

  // Pack scales as vec4 (x,y,z,0) — already in log-space from loader
  const scale4 = new Float32Array(data.numSplats * 4);
  for (let i = 0; i < data.numSplats; i++) {
    scale4[i * 4 + 0] = data.scales[i * 3 + 0];
    scale4[i * 4 + 1] = data.scales[i * 3 + 1];
    scale4[i * 4 + 2] = data.scales[i * 3 + 2];
    scale4[i * 4 + 3] = 0.0;
  }

  // Rotations are already vec4 quaternions, pass through
  const rot4 = data.rotations;

  // Opacities are scalar (already in logit-space from loader), pass through
  const opacities = data.opacities;

  // Pack SH coefficients as vec4 (x,y,z,0) for the shader
  let shData: Float32Array;
  if (data.shCoeffs.length > 0) {
    // Use degree 0 (DC color) — each coefficient is vec3 per splat
    const dc = data.shCoeffs[0];
    shData = new Float32Array(data.numSplats * 4);
    for (let i = 0; i < data.numSplats; i++) {
      shData[i * 4 + 0] = dc[i * 3 + 0];
      shData[i * 4 + 1] = dc[i * 3 + 1];
      shData[i * 4 + 2] = dc[i * 3 + 2];
      shData[i * 4 + 3] = 0.0;
    }
  } else {
    // No SH data — fill with default gray
    shData = new Float32Array(data.numSplats * 4);
    for (let i = 0; i < data.numSplats; i++) {
      shData[i * 4 + 0] = 0.5;
      shData[i * 4 + 1] = 0.5;
      shData[i * 4 + 2] = 0.5;
      shData[i * 4 + 3] = 0.0;
    }
  }

  return {
    positions: makeBuffer(pos4, 'splat_pos'),
    scales: makeBuffer(scale4, 'splat_scale'),
    rotations: makeBuffer(rot4, 'splat_rot'),
    opacities: makeBuffer(opacities, 'splat_opacity'),
    shCoeffs: makeBuffer(shData, 'splat_sh'),
    count: data.numSplats,
    shDegree: data.shDegree,
  };
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
): Promise<{ state: RenderState; materialUboBuffers: Map<number, GPUBuffer> }> {
  const pipeline = await createGltfPbrPipeline(device, format);
  const { draws, uniformBuffers, mvpBindGroup, materialUboBuffers } = buildGltfSceneDrawCalls(device, pipeline, scene, ibl);

  return {
    state: {
      pipeline,
      bindGroups: [mvpBindGroup],
      draws,
      pushBuffer: null,
      uniformBuffers,
    },
    materialUboBuffers,
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

  let gpu: GPUContext;
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
  let currentLoadedScene: Scene | null = null;
  let currentMaterialBuffers: Map<number, GPUBuffer> | null = null;
  let currentSplatRenderer: SplatRenderer | null = null;

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

  function updateSceneUI(scene: Scene): void {
    ui.updateSceneInfo({
      meshes: scene.meshes.length,
      materials: scene.materials.length,
      lights: scene.lights.length,
      drawRanges: scene.drawRanges.length,
      vertices: scene.totalVertices,
      triangles: scene.totalTriangles,
      bounds: { min: scene.boundsMin, max: scene.boundsMax },
      materialNames: scene.materials.map((m, i) =>
        `Mat${i} (${m.alphaMode}${m.hasClearcoat ? ' +coat' : ''}${m.hasSheen ? ' +sheen' : ''}${m.hasTransmission ? ' +trans' : ''})`,
      ),
      lightDescriptions: scene.lights.map(l =>
        `${l.type} [${l.color.map(c => c.toFixed(1)).join(',')}] i=${l.intensity.toFixed(1)}`,
      ),
    });
    ui.setMaterials(scene.materials);
  }

  // Load DamagedHelmet as default scene
  let currentScene: string = 'damaged_helmet';
  try {
    console.log('Loading DamagedHelmet...');
    const scene = await loadGLB(device, DAMAGED_HELMET_URL);
    console.log(`Loaded ${scene.meshes.length} meshes, ${scene.materials.length} materials, ${scene.lights.length} lights, ${scene.drawRanges.length} draw ranges`);
    const { state, materialUboBuffers } = await buildGltfRenderState(device, format, scene, ibl);
    engine.setRenderState(state);
    engine.camera.frameScene(scene.boundsMin, scene.boundsMax);
    engine.lightCount = Math.max(1, scene.lights.length);
    currentLoadedScene = scene;
    currentMaterialBuffers = materialUboBuffers;
    updateSceneUI(scene);
    console.log('DamagedHelmet loaded successfully');
  } catch (e) {
    console.warn('Failed to load DamagedHelmet, falling back to PBR sphere:', e);
    currentScene = 'pbr_sphere';
    const pipeline = createFallbackPipeline(device, format);
    const state = buildSphereRenderState(device, pipeline);
    engine.setRenderState(state);
    currentLoadedScene = null;
    currentMaterialBuffers = null;
  }

  // UI callbacks
  ui.onSceneChange = async (sceneName) => {
    if (sceneName === currentScene) return;
    currentScene = sceneName;

    if (sceneName === 'damaged_helmet') {
      currentSplatRenderer?.destroy();
      currentSplatRenderer = null;
      try {
        const scene = await loadGLB(device, DAMAGED_HELMET_URL);
        const { state, materialUboBuffers } = await buildGltfRenderState(device, format, scene, ibl);
        engine.setRenderState(state);
        engine.camera.frameScene(scene.boundsMin, scene.boundsMax);
        engine.lightCount = Math.max(1, scene.lights.length);
        currentLoadedScene = scene;
        currentMaterialBuffers = materialUboBuffers;
        updateSceneUI(scene);
      } catch (e) {
        console.error('Failed to load DamagedHelmet:', e);
      }
    } else if (sceneName === 'luigi_splat') {
      try {
        console.log('Loading Luigi (Gaussian Splat)...');
        const scene = await loadGLB(device, LUIGI_URL);
        console.log(`Loaded ${scene.meshes.length} meshes, ${scene.materials.length} materials`);

        if (scene.splatData) {
          console.log(`Splat data: ${scene.splatData.numSplats} splats, SH degree ${scene.splatData.shDegree}`);
          const splatBufs = createSplatBuffers(device, scene.splatData);
          const splat = new SplatRenderer(device, scene.splatData.numSplats);
          await splat.init('shaders/gaussian_splat', splatBufs, format);

          // Destroy previous splat renderer if any
          currentSplatRenderer?.destroy();
          currentSplatRenderer = splat;

          // Set a null pipeline render state so the engine just clears
          engine.setRenderState({ pipeline: null, bindGroups: [], draws: [], pushBuffer: null, uniformBuffers: [] });
          engine.camera.frameScene(scene.boundsMin, scene.boundsMax);
          currentLoadedScene = scene;
          currentMaterialBuffers = null;
          updateSceneUI(scene);
          console.log('Splat renderer initialized');
        } else {
          // No splat data found, fall back to PBR rendering
          console.warn('No KHR_gaussian_splatting data found in Luigi GLB, using PBR fallback');
          currentSplatRenderer?.destroy();
          currentSplatRenderer = null;
          const { state, materialUboBuffers } = await buildGltfRenderState(device, format, scene, ibl);
          engine.setRenderState(state);
          engine.camera.frameScene(scene.boundsMin, scene.boundsMax);
          engine.lightCount = Math.max(1, scene.lights.length);
          currentLoadedScene = scene;
          currentMaterialBuffers = materialUboBuffers;
          updateSceneUI(scene);
        }
      } catch (e) {
        console.error('Failed to load Luigi splat scene:', e);
      }
    } else if (sceneName === 'pbr_sphere') {
      currentSplatRenderer?.destroy();
      currentSplatRenderer = null;
      const pipeline = createFallbackPipeline(device, format);
      const state = buildSphereRenderState(device, pipeline);
      engine.setRenderState(state);
      engine.camera.distance = 3.0;
      currentLoadedScene = null;
      currentMaterialBuffers = null;
      ui.setMaterials([]);
    } else {
      currentSplatRenderer?.destroy();
      currentSplatRenderer = null;
      // Try pre-compiled shaders, fall back to sphere
      const compiled = await loadShaderPipeline(device, format, sceneName);
      const pipeline = compiled ?? createFallbackPipeline(device, format);
      const state = buildSphereRenderState(device, pipeline);
      engine.setRenderState(state);
      currentLoadedScene = null;
      currentMaterialBuffers = null;
      ui.setMaterials([]);
    }
  };

  ui.onMaterialChange = (materialIndex, field, value) => {
    // Update the in-memory material
    const mat = currentLoadedScene?.materials[materialIndex];
    if (!mat) return;
    (mat as any)[field] = value;

    // Re-pack and upload to GPU via Material UBO
    const buf = currentMaterialBuffers?.get(materialIndex);
    if (buf) {
      const data = fillWebMaterialUBO(mat);
      device.queue.writeBuffer(buf, 0, data);
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
    currentSplatRenderer?.destroy();
    currentSplatRenderer = null;
    try {
      const glScene = await loadGLBFromBuffer(device, buffer);
      console.log(`Loaded ${glScene.meshes.length} meshes, ${glScene.materials.length} materials, ${glScene.lights.length} lights`);
      const { state, materialUboBuffers } = await buildGltfRenderState(device, format, glScene, ibl);
      engine.setRenderState(state);
      engine.camera.frameScene(glScene.boundsMin, glScene.boundsMax);
      engine.lightCount = Math.max(1, glScene.lights.length);
      currentLoadedScene = glScene;
      currentMaterialBuffers = materialUboBuffers;
      updateSceneUI(glScene);
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

    engine.exposure = ui.state.exposure;
    engine.lightDirY = ui.state.lightDirY;

    if (currentSplatRenderer) {
      // Splat rendering: clear + compute preprocess + sort + render
      const textureView = gpu.context.getCurrentTexture().createView();
      const encoder = gpu.device.createCommandEncoder();
      const depthView = engine.getDepthView();

      // Clear pass (background)
      const clearPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: textureView,
          clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      clearPass.end();

      // Get camera matrices
      const view = mat4.create();
      const proj = mat4.create();
      engine.camera.getViewMatrix(view);
      engine.camera.getProjectionMatrix(proj, engine.width / engine.height);

      // Run splat renderer (preprocess + sort + render)
      currentSplatRenderer.render(
        encoder, textureView, depthView,
        view, proj, [engine.width, engine.height],
      );

      gpu.device.queue.submit([encoder.finish()]);
    } else {
      engine.renderFrame(dt);
    }

    ui.updateFPS(dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch(console.error);
