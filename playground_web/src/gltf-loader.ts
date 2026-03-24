/**
 * Minimal glTF/GLB loader for WebGPU.
 *
 * Supports: GLB binary container, meshes (positions, normals, UVs, tangents, indices),
 * PBR metallic-roughness materials, textures, node transforms, KHR_lights_punctual.
 */

import { mat4, quat } from 'gl-matrix';
import type { Scene, Mesh, Material, Light, SceneNode, DrawRange } from './scene';

// --------------------------------------------------------------------------
// GLB binary container parsing
// --------------------------------------------------------------------------

interface GLBData {
  json: Record<string, unknown>;
  bin: ArrayBuffer;
}

function parseGLB(buffer: ArrayBuffer): GLBData {
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== 0x46546C67) throw new Error('Not a GLB file');

  const version = view.getUint32(4, true);
  if (version !== 2) throw new Error(`Unsupported glTF version: ${version}`);

  // Chunk 0: JSON
  const jsonLen = view.getUint32(12, true);
  const jsonType = view.getUint32(16, true);
  if (jsonType !== 0x4E4F534A) throw new Error('Expected JSON chunk');
  const jsonBytes = new Uint8Array(buffer, 20, jsonLen);
  const json = JSON.parse(new TextDecoder().decode(jsonBytes));

  // Chunk 1: BIN (optional)
  let bin = new ArrayBuffer(0);
  const chunk1Offset = 20 + jsonLen;
  if (chunk1Offset < buffer.byteLength - 8) {
    const binLen = view.getUint32(chunk1Offset, true);
    const binType = view.getUint32(chunk1Offset + 4, true);
    if (binType === 0x004E4942) {
      bin = buffer.slice(chunk1Offset + 8, chunk1Offset + 8 + binLen);
    }
  }

  return { json, bin };
}

// --------------------------------------------------------------------------
// Accessor helpers
// --------------------------------------------------------------------------

interface Accessor {
  bufferView: number;
  byteOffset: number;
  componentType: number;
  count: number;
  type: string;
  min?: number[];
  max?: number[];
}

interface BufferView {
  buffer: number;
  byteOffset: number;
  byteLength: number;
  byteStride?: number;
}

const COMPONENT_SIZES: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

const TYPE_COUNTS: Record<string, number> = {
  SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16,
};

function getAccessorData(
  accessor: Accessor,
  bufferViews: BufferView[],
  bin: ArrayBuffer,
): ArrayBuffer {
  const bv = bufferViews[accessor.bufferView];
  const compSize = COMPONENT_SIZES[accessor.componentType] ?? 4;
  const count = TYPE_COUNTS[accessor.type] ?? 1;
  const byteOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const byteLength = accessor.count * count * compSize;
  return bin.slice(byteOffset, byteOffset + byteLength);
}

// --------------------------------------------------------------------------
// Texture loading
// --------------------------------------------------------------------------

async function loadTexture(
  device: GPUDevice,
  imageIndex: number,
  gltf: Record<string, unknown>,
  bufferViews: BufferView[],
  bin: ArrayBuffer,
): Promise<GPUTexture> {
  const images = gltf.images as Array<{ bufferView?: number; mimeType?: string; uri?: string }>;
  const img = images[imageIndex];

  let blob: Blob;
  if (img.bufferView !== undefined) {
    const bv = bufferViews[img.bufferView];
    const data = bin.slice(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength);
    blob = new Blob([data], { type: img.mimeType ?? 'image/png' });
  } else {
    throw new Error('External URI textures not yet supported');
  }

  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });

  const texture = device.createTexture({
    size: [bitmap.width, bitmap.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture },
    [bitmap.width, bitmap.height],
  );

  return texture;
}

// --------------------------------------------------------------------------
// Node transform
// --------------------------------------------------------------------------

function nodeTransform(node: Record<string, unknown>): Float32Array {
  if (node.matrix) {
    return new Float32Array(node.matrix as number[]);
  }

  const out = mat4.create();
  const t = (node.translation as number[]) ?? [0, 0, 0];
  const r = (node.rotation as number[]) ?? [0, 0, 0, 1];
  const s = (node.scale as number[]) ?? [1, 1, 1];

  mat4.fromRotationTranslationScale(
    out,
    quat.fromValues(r[0], r[1], r[2], r[3]),
    [t[0], t[1], t[2]],
    [s[0], s[1], s[2]],
  );

  return out as Float32Array;
}

// --------------------------------------------------------------------------
// Main loader
// --------------------------------------------------------------------------

export async function loadGLB(
  device: GPUDevice,
  url: string,
): Promise<Scene> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}`);
  const buffer = await response.arrayBuffer();
  return parseGLBScene(device, buffer);
}

export async function loadGLBFromBuffer(
  device: GPUDevice,
  buffer: ArrayBuffer,
): Promise<Scene> {
  return parseGLBScene(device, buffer);
}

async function parseGLBScene(
  device: GPUDevice,
  buffer: ArrayBuffer,
): Promise<Scene> {
  const { json: gltf, bin } = parseGLB(buffer);

  const accessors = (gltf.accessors ?? []) as Accessor[];
  const bufferViews = (gltf.bufferViews ?? []) as BufferView[];
  const gltfMeshes = (gltf.meshes ?? []) as Array<{ primitives: Array<Record<string, unknown>> }>;
  const gltfMaterials = (gltf.materials ?? []) as Array<Record<string, unknown>>;
  const gltfNodes = (gltf.nodes ?? []) as Array<Record<string, unknown>>;
  const gltfTextures = (gltf.textures ?? []) as Array<{ source?: number }>;

  // Default sampler
  const defaultSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  });

  // Load textures
  const textureCache = new Map<number, GPUTexture>();
  async function getTexture(index: number): Promise<GPUTexture> {
    if (textureCache.has(index)) return textureCache.get(index)!;
    const texDef = gltfTextures[index];
    if (texDef?.source !== undefined) {
      const tex = await loadTexture(device, texDef.source, gltf, bufferViews, bin);
      textureCache.set(index, tex);
      return tex;
    }
    throw new Error(`Invalid texture index: ${index}`);
  }

  // Parse materials
  const materials: Material[] = [];
  for (let matIdx = 0; matIdx < gltfMaterials.length; matIdx++) {
    const mat = gltfMaterials[matIdx];
    const pbr = (mat.pbrMetallicRoughness ?? {}) as Record<string, unknown>;
    const bc = (pbr.baseColorFactor as number[]) ?? [1, 1, 1, 1];
    const texMap = new Map<string, GPUTextureView>();

    // Load base color texture
    const bct = pbr.baseColorTexture as { index?: number } | undefined;
    if (bct?.index !== undefined) {
      const tex = await getTexture(bct.index);
      texMap.set('baseColor', tex.createView());
    }

    // Load normal texture
    const nt = mat.normalTexture as { index?: number } | undefined;
    if (nt?.index !== undefined) {
      const tex = await getTexture(nt.index);
      texMap.set('normal', tex.createView());
    }

    // Load metallic-roughness texture
    const mrt = pbr.metallicRoughnessTexture as { index?: number } | undefined;
    if (mrt?.index !== undefined) {
      const tex = await getTexture(mrt.index);
      texMap.set('metallicRoughness', tex.createView());
    }

    // Load emissive texture
    const et = mat.emissiveTexture as { index?: number } | undefined;
    if (et?.index !== undefined) {
      const tex = await getTexture(et.index);
      texMap.set('emissive', tex.createView());
    }

    const emf = (mat.emissiveFactor as number[]) ?? [0, 0, 0];

    // Parse material extensions
    const exts = (mat.extensions ?? {}) as Record<string, Record<string, unknown>>;

    // KHR_materials_clearcoat
    const clearcoatExt = exts.KHR_materials_clearcoat;
    const hasClearcoat = clearcoatExt !== undefined;
    const clearcoatFactor = hasClearcoat ? (clearcoatExt.clearcoatFactor as number) ?? 0.0 : 0.0;
    const clearcoatRoughnessFactor = hasClearcoat ? (clearcoatExt.clearcoatRoughnessFactor as number) ?? 0.0 : 0.0;

    // KHR_materials_sheen
    const sheenExt = exts.KHR_materials_sheen;
    const hasSheen = sheenExt !== undefined;
    const sheenColor = hasSheen ? (sheenExt.sheenColorFactor as number[]) ?? [0, 0, 0] : [0, 0, 0];
    const sheenRoughnessFactor = hasSheen ? (sheenExt.sheenRoughnessFactor as number) ?? 0.0 : 0.0;

    // KHR_materials_transmission
    const transmissionExt = exts.KHR_materials_transmission;
    const hasTransmission = transmissionExt !== undefined;
    const transmissionFactor = hasTransmission ? (transmissionExt.transmissionFactor as number) ?? 0.0 : 0.0;

    // KHR_materials_ior
    const iorExt = exts.KHR_materials_ior;
    const ior = iorExt ? (iorExt.ior as number) ?? 1.5 : 1.5;

    // KHR_materials_emissive_strength
    const emStrExt = exts.KHR_materials_emissive_strength;
    const emissiveStrength = emStrExt ? (emStrExt.emissiveStrength as number) ?? 1.0 : 1.0;

    // KHR_materials_unlit
    const isUnlit = exts.KHR_materials_unlit !== undefined;

    materials.push({
      baseColor: [bc[0], bc[1], bc[2], bc[3]],
      metallic: (pbr.metallicFactor as number) ?? 1.0,
      roughness: (pbr.roughnessFactor as number) ?? 1.0,
      emissive: [emf[0], emf[1], emf[2]],
      textures: texMap,
      sampler: defaultSampler,
      alphaMode: (mat.alphaMode as Material['alphaMode']) ?? 'OPAQUE',
      alphaCutoff: (mat.alphaCutoff as number) ?? 0.5,
      doubleSided: (mat.doubleSided as boolean) ?? false,
      ior,
      emissiveStrength,
      isUnlit,
      hasClearcoat,
      clearcoatFactor,
      clearcoatRoughnessFactor,
      hasSheen,
      sheenColorFactor: [sheenColor[0], sheenColor[1], sheenColor[2]],
      sheenRoughnessFactor,
      hasTransmission,
      transmissionFactor,
      materialIndex: matIdx,
    });
  }

  // Parse meshes
  // Track per-primitive material indices and position data for bounds/draw ranges
  const meshes: Mesh[] = [];
  const primitiveMaterialIndices: number[] = [];  // parallel to meshes[]
  const boundsMin: [number, number, number] = [Infinity, Infinity, Infinity];
  const boundsMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  for (const gm of gltfMeshes) {
    for (const prim of gm.primitives) {
      const attrs = prim.attributes as Record<string, number>;

      // Track which material this primitive uses
      const primMatIdx = (prim.material as number) ?? -1;
      primitiveMaterialIndices.push(primMatIdx);

      // Interleave: position(3) + normal(3) + uv(2) + tangent(4) = 12 floats = 48 bytes
      const posAccess = accessors[attrs.POSITION];
      const posData = new Float32Array(getAccessorData(posAccess, bufferViews, bin));
      const vertCount = posAccess.count;

      // Update scene bounds from position data
      for (let i = 0; i < vertCount; i++) {
        const x = posData[i * 3 + 0];
        const y = posData[i * 3 + 1];
        const z = posData[i * 3 + 2];
        if (x < boundsMin[0]) boundsMin[0] = x;
        if (y < boundsMin[1]) boundsMin[1] = y;
        if (z < boundsMin[2]) boundsMin[2] = z;
        if (x > boundsMax[0]) boundsMax[0] = x;
        if (y > boundsMax[1]) boundsMax[1] = y;
        if (z > boundsMax[2]) boundsMax[2] = z;
      }

      let norData: Float32Array | null = null;
      if (attrs.NORMAL !== undefined) {
        norData = new Float32Array(getAccessorData(accessors[attrs.NORMAL], bufferViews, bin));
      }

      let uvData: Float32Array | null = null;
      if (attrs.TEXCOORD_0 !== undefined) {
        uvData = new Float32Array(getAccessorData(accessors[attrs.TEXCOORD_0], bufferViews, bin));
      }

      let tanData: Float32Array | null = null;
      if (attrs.TANGENT !== undefined) {
        tanData = new Float32Array(getAccessorData(accessors[attrs.TANGENT], bufferViews, bin));
      }

      // Interleave
      const stride = 12; // 3+3+2+4 floats
      const interleavedData = new Float32Array(vertCount * stride);
      for (let i = 0; i < vertCount; i++) {
        interleavedData[i * stride + 0] = posData[i * 3 + 0];
        interleavedData[i * stride + 1] = posData[i * 3 + 1];
        interleavedData[i * stride + 2] = posData[i * 3 + 2];
        interleavedData[i * stride + 3] = norData ? norData[i * 3 + 0] : 0;
        interleavedData[i * stride + 4] = norData ? norData[i * 3 + 1] : 1;
        interleavedData[i * stride + 5] = norData ? norData[i * 3 + 2] : 0;
        interleavedData[i * stride + 6] = uvData ? uvData[i * 2 + 0] : 0;
        interleavedData[i * stride + 7] = uvData ? uvData[i * 2 + 1] : 0;
        interleavedData[i * stride + 8] = tanData ? tanData[i * 4 + 0] : 1;
        interleavedData[i * stride + 9] = tanData ? tanData[i * 4 + 1] : 0;
        interleavedData[i * stride + 10] = tanData ? tanData[i * 4 + 2] : 0;
        interleavedData[i * stride + 11] = tanData ? tanData[i * 4 + 3] : 1;
      }

      const vertexBuffer = device.createBuffer({
        size: interleavedData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(vertexBuffer, 0, interleavedData);

      // Index buffer
      let indexBuffer: GPUBuffer | null = null;
      let indexCount = 0;
      if (prim.indices !== undefined) {
        const idxAccess = accessors[prim.indices as number];
        const idxData = getAccessorData(idxAccess, bufferViews, bin);

        // Convert to uint32
        let uint32Data: Uint32Array;
        if (idxAccess.componentType === 5123) {
          // UNSIGNED_SHORT → UNSIGNED_INT
          const u16 = new Uint16Array(idxData);
          uint32Data = new Uint32Array(u16.length);
          for (let i = 0; i < u16.length; i++) uint32Data[i] = u16[i];
        } else if (idxAccess.componentType === 5125) {
          uint32Data = new Uint32Array(idxData);
        } else {
          const u8 = new Uint8Array(idxData);
          uint32Data = new Uint32Array(u8.length);
          for (let i = 0; i < u8.length; i++) uint32Data[i] = u8[i];
        }

        indexCount = uint32Data.length;
        indexBuffer = device.createBuffer({
          size: uint32Data.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(indexBuffer, 0, uint32Data.buffer, uint32Data.byteOffset, uint32Data.byteLength);
      }

      meshes.push({
        vertexBuffer,
        indexBuffer,
        indexCount,
        vertexCount: vertCount,
        vertexStride: stride * 4,
      });
    }
  }

  // Clamp bounds if no geometry was loaded
  if (boundsMin[0] === Infinity) {
    boundsMin[0] = boundsMin[1] = boundsMin[2] = 0;
    boundsMax[0] = boundsMax[1] = boundsMax[2] = 0;
  }

  // Parse lights (KHR_lights_punctual)
  const lights: Light[] = [];
  const extensions = gltf.extensions as Record<string, unknown> | undefined;
  if (extensions?.KHR_lights_punctual) {
    const khrLights = (extensions.KHR_lights_punctual as Record<string, unknown>).lights as Array<Record<string, unknown>>;
    for (const l of khrLights ?? []) {
      const color = (l.color as number[]) ?? [1, 1, 1];
      lights.push({
        type: (l.type as Light['type']) ?? 'directional',
        color: [color[0], color[1], color[2]],
        intensity: (l.intensity as number) ?? 1.0,
        position: [0, 0, 0],
        direction: [0, -1, 0],
        range: (l.range as number) ?? 0,
        innerCone: 0,
        outerCone: Math.PI / 4,
      });
    }
  }

  // Build a mapping from glTF mesh index to the range of entries in meshes[]
  // (each glTF mesh may have multiple primitives, each becoming one Mesh entry)
  const meshPrimitiveRanges: Array<{ start: number; count: number }> = [];
  let flatIdx = 0;
  for (const gm of gltfMeshes) {
    const count = gm.primitives.length;
    meshPrimitiveRanges.push({ start: flatIdx, count });
    flatIdx += count;
  }

  // Build scene graph
  const nodes: SceneNode[] = [];
  for (const gn of gltfNodes) {
    const node: SceneNode = {
      transform: nodeTransform(gn),
      children: [],
    };
    if (gn.mesh !== undefined) {
      const meshIdx = gn.mesh as number;
      // Point to first primitive mesh (backward compat)
      const range = meshPrimitiveRanges[meshIdx];
      node.mesh = meshes[range.start];
      // Material from first primitive
      const gm = gltfMeshes[meshIdx];
      const matIdx = gm.primitives[0]?.material as number | undefined;
      if (matIdx !== undefined && matIdx < materials.length) {
        node.material = materials[matIdx];
      }
    }
    nodes.push(node);
  }

  // Wire up children
  for (let i = 0; i < gltfNodes.length; i++) {
    const children = gltfNodes[i].children as number[] | undefined;
    if (children) {
      for (const ci of children) {
        if (ci < nodes.length) nodes[i].children.push(nodes[ci]);
      }
    }
  }

  // Root nodes from scene
  const sceneIdx = (gltf.scene as number) ?? 0;
  const scenes = gltf.scenes as Array<{ nodes?: number[] }>;
  const rootNodeIndices = scenes?.[sceneIdx]?.nodes ?? [];
  const rootNodes = rootNodeIndices.map((i: number) => nodes[i]).filter(Boolean);

  // Compute world transforms and build draw ranges by traversing the scene graph.
  // Each node with a mesh generates one DrawRange per primitive of that glTF mesh.
  const drawRanges: DrawRange[] = [];

  function traverseForDrawRanges(
    nodeIdx: number,
    parentWorldTransform: Float32Array,
  ): void {
    const gn = gltfNodes[nodeIdx];
    const localTransform = nodeTransform(gn);
    const worldTransform = mat4.create() as Float32Array;
    mat4.multiply(worldTransform, parentWorldTransform, localTransform);

    if (gn.mesh !== undefined) {
      const gltfMeshIdx = gn.mesh as number;
      const range = meshPrimitiveRanges[gltfMeshIdx];
      const gm = gltfMeshes[gltfMeshIdx];

      for (let p = 0; p < gm.primitives.length; p++) {
        const meshEntry = meshes[range.start + p];
        const primMatIdx = primitiveMaterialIndices[range.start + p];

        drawRanges.push({
          indexOffset: 0,
          indexCount: meshEntry.indexCount,
          vertexOffset: 0,
          vertexCount: meshEntry.vertexCount,
          materialIndex: primMatIdx >= 0 ? primMatIdx : 0,
          worldTransform: new Float32Array(worldTransform),
          meshIndex: range.start + p,
        });
      }
    }

    // Recurse into children
    const children = gn.children as number[] | undefined;
    if (children) {
      for (const ci of children) {
        traverseForDrawRanges(ci, worldTransform);
      }
    }
  }

  const identityMat = mat4.create() as Float32Array;
  for (const ri of rootNodeIndices) {
    traverseForDrawRanges(ri, identityMat);
  }

  // Compute totals
  let totalVertices = 0;
  let totalTriangles = 0;
  for (const m of meshes) {
    totalVertices += m.vertexCount;
    totalTriangles += m.indexCount > 0 ? Math.floor(m.indexCount / 3) : Math.floor(m.vertexCount / 3);
  }

  return {
    nodes: rootNodes,
    meshes,
    materials,
    lights,
    boundsMin,
    boundsMax,
    drawRanges,
    totalVertices,
    totalTriangles,
  };
}
