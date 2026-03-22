/**
 * Multi-light system matching the C++ scene_light.h GPU layout.
 *
 * Per-light packing (std430, 64 bytes):
 *   vec4: (light_type, intensity, range, inner_cone)
 *   vec4: (position.xyz, outer_cone)
 *   vec4: (direction.xyz, shadow_index)
 *   vec4: (color.xyz, pad)
 *
 * Light types: 0 = directional, 1 = point, 2 = spot.
 */

import { mat4, vec3 } from 'gl-matrix';
import type { Light, Scene, SceneNode } from './scene';

/** Bytes per light in the GPU SSBO. */
export const LIGHT_BUFFER_STRIDE = 64;

/** Maximum number of lights supported by the engine. */
export const MAX_LIGHTS = 16;

/** Numeric encoding of light type for the GPU. */
const LIGHT_TYPE_MAP: Record<Light['type'], number> = {
  directional: 0,
  point: 1,
  spot: 2,
};

/**
 * Pack an array of scene lights into a flat Float32Array for GPU SSBO upload.
 *
 * Each light occupies 64 bytes (16 floats). The array is clamped to MAX_LIGHTS.
 * Returns a Float32Array sized for exactly the packed lights (or 1 light minimum
 * so the buffer is never zero-length).
 */
export function packLightsBuffer(lights: Light[]): Float32Array {
  const count = Math.min(lights.length, MAX_LIGHTS);
  const floatsPerLight = LIGHT_BUFFER_STRIDE / 4; // 16
  const data = new Float32Array(Math.max(count, 1) * floatsPerLight);

  for (let i = 0; i < count; i++) {
    const l = lights[i];
    const base = i * floatsPerLight;

    // vec4: (light_type, intensity, range, inner_cone)
    data[base + 0] = LIGHT_TYPE_MAP[l.type];
    data[base + 1] = l.intensity;
    data[base + 2] = l.range;
    data[base + 3] = l.innerCone;

    // vec4: (position.xyz, outer_cone)
    data[base + 4] = l.position[0];
    data[base + 5] = l.position[1];
    data[base + 6] = l.position[2];
    data[base + 7] = l.outerCone;

    // vec4: (direction.xyz, shadow_index)
    data[base + 8] = l.direction[0];
    data[base + 9] = l.direction[1];
    data[base + 10] = l.direction[2];
    data[base + 11] = -1.0; // shadow_index: -1 = no shadow

    // vec4: (color.xyz, pad)
    data[base + 12] = l.color[0];
    data[base + 13] = l.color[1];
    data[base + 14] = l.color[2];
    data[base + 15] = 0.0;
  }

  return data;
}

/**
 * Create the SceneLightUBO data: 16 bytes containing vec3 view_pos + int light_count.
 *
 * Layout (std140):
 *   offset  0: vec3 view_pos (12 bytes)
 *   offset 12: int  light_count (4 bytes, stored as float for uniform compatibility)
 */
export function packSceneLightUBO(
  viewPos: [number, number, number],
  lightCount: number,
): Float32Array {
  const data = new Float32Array(4);
  data[0] = viewPos[0];
  data[1] = viewPos[1];
  data[2] = viewPos[2];
  // Store light count — reinterpret as float for the uniform buffer.
  // The shader will use floatBitsToInt() or we store as float and cast in shader.
  // Using DataView to write an actual int32 at byte offset 12.
  const buf = data.buffer;
  const view = new DataView(buf);
  view.setInt32(12, Math.min(lightCount, MAX_LIGHTS), true);
  return data;
}

/**
 * Extract lights from a loaded glTF scene, applying world transforms to
 * positions and directions.
 *
 * If the scene contains no lights, a default directional light is created
 * with direction [0.5, 0.7, 0.3] (normalized), white color, intensity 1.
 */
export function populateLightsFromScene(scene: Scene): Light[] {
  // If the scene already has lights with position/direction baked in from the
  // loader, try to enrich them with node transforms.
  const transformedLights = extractNodeLights(scene);

  if (transformedLights.length > 0) {
    return transformedLights;
  }

  // If we got lights from the scene object directly (parsed from
  // KHR_lights_punctual at the extension level but not attached to nodes),
  // return those.
  if (scene.lights.length > 0) {
    return scene.lights.slice(0, MAX_LIGHTS);
  }

  // No lights at all — create a sensible default directional light.
  const dir = vec3.normalize(vec3.create(), vec3.fromValues(0.5, 0.7, 0.3));
  return [
    {
      type: 'directional',
      color: [1, 1, 1],
      intensity: 1,
      position: [0, 0, 0],
      direction: [dir[0], dir[1], dir[2]],
      range: 0,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
  ];
}

// ---------------------------------------------------------------------------
// Internal: walk scene graph to find node-attached lights
// ---------------------------------------------------------------------------

/**
 * Walk the scene graph and collect lights with world-space transforms applied.
 *
 * In glTF, lights are defined in the top-level KHR_lights_punctual extension
 * and *referenced* by node index. The node's world transform determines the
 * light's position and direction. The light definitions (type, color,
 * intensity, range, cone angles) come from the extension array, and the
 * scene.lights array holds those parsed values (indexed in the same order).
 *
 * We walk all nodes, and for each node that could carry a light, we apply
 * the accumulated world transform to the light's default direction (0, 0, -1)
 * and use the node's translation as the world position.
 */
function extractNodeLights(scene: Scene): Light[] {
  if (scene.lights.length === 0) return [];

  const result: Light[] = [];
  const identity = mat4.create();

  function walk(nodes: SceneNode[], parentWorld: mat4): void {
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const world = mat4.create();
      mat4.multiply(world, parentWorld, node.transform as mat4);

      // A node with a light reference will have a matching index in scene.lights.
      // Since the current gltf-loader attaches lights to the scene-level array
      // in order and nodes reference them by extension index, we check if this
      // node index maps to a light. However, the SceneNode interface doesn't
      // carry a lightIndex field, so we scan all nodes and match by traversal
      // order against the scene.lights array.
      //
      // For now we rely on the caller (populateLightsFromScene) to fall back to
      // scene.lights directly when node-level attachment isn't available.

      walk(node.children, world);
    }
  }

  walk(scene.nodes, identity);

  // If no node-attached lights were found through explicit references,
  // apply transforms from root nodes to the scene-level lights as a heuristic.
  // Many exporters place light nodes at root level in the same order as the
  // lights array.
  if (result.length === 0 && scene.lights.length > 0) {
    // Collect all leaf/root node transforms that don't have meshes
    // (likely light nodes).
    const lightNodeTransforms: mat4[] = [];
    function collectLightNodes(nodes: SceneNode[], parentWorld: mat4): void {
      for (const node of nodes) {
        const world = mat4.create();
        mat4.multiply(world, parentWorld, node.transform as mat4);
        if (!node.mesh) {
          lightNodeTransforms.push(world);
        }
        collectLightNodes(node.children, world);
      }
    }
    collectLightNodes(scene.nodes, identity);

    for (let i = 0; i < scene.lights.length && i < MAX_LIGHTS; i++) {
      const base = scene.lights[i];
      if (i < lightNodeTransforms.length) {
        result.push(applyTransformToLight(base, lightNodeTransforms[i]));
      } else {
        result.push({ ...base });
      }
    }
  }

  return result;
}

/**
 * Apply a world transform matrix to a light, producing world-space position
 * and direction.
 */
function applyTransformToLight(light: Light, world: mat4): Light {
  // Extract world-space position from the matrix translation column.
  const pos = vec3.fromValues(
    (world as Float32Array)[12],
    (world as Float32Array)[13],
    (world as Float32Array)[14],
  );

  // Default light direction in glTF is (0, 0, -1) in local space.
  // Transform it by the upper-3x3 of the world matrix (rotation only).
  const localDir = vec3.fromValues(0, 0, -1);
  const worldDir = vec3.create();
  vec3.transformMat4(worldDir, localDir, world);
  // Subtract the translation to get just the rotated direction.
  vec3.subtract(worldDir, worldDir, pos);
  vec3.normalize(worldDir, worldDir);

  return {
    type: light.type,
    color: [...light.color],
    intensity: light.intensity,
    position: [pos[0], pos[1], pos[2]],
    direction: [worldDir[0], worldDir[1], worldDir[2]],
    range: light.range,
    innerCone: light.innerCone,
    outerCone: light.outerCone,
  };
}
