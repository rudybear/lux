/**
 * Shader permutation manifest: per-material feature detection and best-match selection.
 *
 * Mirrors the C++ SceneManager permutation logic:
 *   detectMaterialFeatures() -> featuresToSuffix() -> groupMaterialsByFeatures()
 */

import type { Material } from './scene';

// --------------------------------------------------------------------------
// Manifest types
// --------------------------------------------------------------------------

export interface ManifestPermutation {
  suffix: string;
  features: Record<string, boolean>;
}

export interface ShaderManifest {
  pipeline: string;
  features: string[];
  permutations: ManifestPermutation[];
}

// --------------------------------------------------------------------------
// Manifest loading
// --------------------------------------------------------------------------

/** Load a shader manifest JSON from the server. */
export async function loadManifest(basePath: string): Promise<ShaderManifest | null> {
  try {
    const resp = await fetch(`${basePath}.manifest.json`);
    if (!resp.ok) return null;
    return await resp.json() as ShaderManifest;
  } catch {
    return null;
  }
}

// --------------------------------------------------------------------------
// Per-material feature detection (mirrors C++ SceneManager::detectMaterialFeatures)
// --------------------------------------------------------------------------

/** Detect rendering features needed by a single material. */
export function detectMaterialFeatures(mat: Material): Set<string> {
  const features = new Set<string>();

  if (mat.textures.has('normal')) {
    features.add('has_normal_map');
  }

  if (mat.textures.has('emissive') ||
      (mat.emissive[0] > 0 || mat.emissive[1] > 0 || mat.emissive[2] > 0)) {
    features.add('has_emission');
  }

  if (mat.hasClearcoat) {
    features.add('has_clearcoat');
  }

  if (mat.hasSheen) {
    features.add('has_sheen');
  }

  if (mat.hasTransmission) {
    features.add('has_transmission');
  }

  return features;
}

// --------------------------------------------------------------------------
// Suffix construction (mirrors C++ SceneManager::featuresToSuffix)
// --------------------------------------------------------------------------

/** Build a permutation suffix string from a feature set, e.g. "+clearcoat+normal_map". */
export function featuresToSuffix(features: Set<string>): string {
  if (features.size === 0) return '';
  // Sort for deterministic ordering, strip "has_" prefix
  const sorted = Array.from(features).sort();
  return sorted.map(f => '+' + f.replace(/^has_/, '')).join('');
}

// --------------------------------------------------------------------------
// Material grouping (mirrors C++ SceneManager::groupMaterialsByFeatures)
// --------------------------------------------------------------------------

/** Group materials by their feature suffix. Returns {suffix: [materialIndex, ...]}. */
export function groupMaterialsByFeatures(
  materials: Material[],
): Map<string, number[]> {
  const groups = new Map<string, number[]>();

  for (let i = 0; i < materials.length; i++) {
    const feats = detectMaterialFeatures(materials[i]);
    const suffix = featuresToSuffix(feats);
    const list = groups.get(suffix);
    if (list) {
      list.push(i);
    } else {
      groups.set(suffix, [i]);
    }
  }

  return groups;
}

// --------------------------------------------------------------------------
// Permutation matching
// --------------------------------------------------------------------------

/** Find the best matching permutation suffix from a manifest for a given feature set. */
export function findBestPermutation(
  manifest: ShaderManifest,
  features: Set<string>,
): string {
  const target = featuresToSuffix(features);

  // Exact match
  for (const perm of manifest.permutations) {
    if (perm.suffix === target) return perm.suffix;
  }

  // Find permutation with most matching features (subset match)
  let bestSuffix = '';
  let bestScore = -1;

  for (const perm of manifest.permutations) {
    const permFeatures = new Set(
      Object.entries(perm.features)
        .filter(([, v]) => v)
        .map(([k]) => k),
    );

    // Check if permutation features are a subset of target features
    let isSubset = true;
    for (const f of permFeatures) {
      if (!features.has(f)) {
        isSubset = false;
        break;
      }
    }

    if (isSubset && permFeatures.size > bestScore) {
      bestScore = permFeatures.size;
      bestSuffix = perm.suffix;
    }
  }

  return bestSuffix;
}

/** Check if a specific permutation variant's shaders exist on the server. */
export async function permutationExists(basePath: string, suffix: string): Promise<boolean> {
  const path = suffix ? `${basePath}${suffix}` : basePath;
  try {
    const resp = await fetch(`shaders/${path}.vert.wgsl`, { method: 'HEAD' });
    return resp.ok;
  } catch {
    return false;
  }
}
