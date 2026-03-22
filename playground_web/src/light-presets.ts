/**
 * Named light presets matching C++ engine configurations.
 *
 * Each preset returns an array of Light objects for common scene setups.
 */

import type { Light } from './scene';

function normalize(v: [number, number, number]): [number, number, number] {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len < 1e-8) return [0, -1, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

/** Default single directional light (used when scene has no lights). */
export function defaultLight(): Light[] {
  return [{
    type: 'directional',
    color: [1, 0.98, 0.95],
    intensity: 1.0,
    position: [0, 0, 0],
    direction: normalize([0.5, 0.7, 0.3]),
    range: 0,
    innerCone: 0,
    outerCone: Math.PI / 4,
  }];
}

/** Demo lights: directional sun + red point + green spot (matches C++ --demo-lights). */
export function demoLights(): Light[] {
  return [
    {
      type: 'directional',
      color: [1, 0.98, 0.95],
      intensity: 2.5,
      position: [0, 0, 0],
      direction: normalize([0.5, -0.7, 0.3]),
      range: 0,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
    {
      type: 'point',
      color: [1, 0.2, 0.1],
      intensity: 5.0,
      position: [2, 1, 0],
      direction: [0, -1, 0],
      range: 10,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
    {
      type: 'spot',
      color: [0.1, 1, 0.2],
      intensity: 8.0,
      position: [-1, 3, 1],
      direction: normalize([0.2, -1, -0.1]),
      range: 15,
      innerCone: Math.PI / 8,
      outerCone: Math.PI / 4,
    },
  ];
}

/** Sponza courtyard lights: sun + orbiting torch + accent (matches C++ --sponza-lights). */
export function sponzaLights(): Light[] {
  return [
    {
      type: 'directional',
      color: [1, 0.95, 0.85],
      intensity: 3.0,
      position: [0, 0, 0],
      direction: normalize([0.3, -0.8, 0.5]),
      range: 0,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
    {
      type: 'point',
      color: [1, 0.7, 0.3],
      intensity: 10.0,
      position: [3, 2, 0],
      direction: [0, -1, 0],
      range: 15,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
    {
      type: 'point',
      color: [0.5, 0.6, 1.0],
      intensity: 4.0,
      position: [-2, 3, 2],
      direction: [0, -1, 0],
      range: 12,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
  ];
}

/** Lighttest scene: directional sun + red point + green point for shadow validation. */
export function lighttestLights(): Light[] {
  return [
    {
      type: 'directional',
      color: [1, 1, 1],
      intensity: 2.0,
      position: [0, 0, 0],
      direction: normalize([0.3, -0.8, 0.3]),
      range: 0,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
    {
      type: 'point',
      color: [1, 0.1, 0.1],
      intensity: 5.0,
      position: [2, 2, 0],
      direction: [0, -1, 0],
      range: 8,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
    {
      type: 'point',
      color: [0.1, 1, 0.1],
      intensity: 5.0,
      position: [-2, 2, 0],
      direction: [0, -1, 0],
      range: 8,
      innerCone: 0,
      outerCone: Math.PI / 4,
    },
  ];
}
