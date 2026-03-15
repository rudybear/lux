/**
 * Procedural IBL texture generation.
 *
 * Generates specular cubemap (with roughness mip chain), irradiance cubemap,
 * all on the CPU at startup. Uses importance-sampled GGX for specular
 * prefiltering and cosine-weighted hemisphere sampling for irradiance.
 */

export interface ProceduralIBL {
  specular: GPUTexture;
  irradiance: GPUTexture;
  brdfLut: GPUTexture;
  specularView: GPUTextureView;
  irradianceView: GPUTextureView;
  brdfLutView: GPUTextureView;
  sampler: GPUSampler;
}

// --------------------------------------------------------------------------
// Math helpers
// --------------------------------------------------------------------------

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function normalize3(x: number, y: number, z: number): [number, number, number] {
  const len = Math.sqrt(x * x + y * y + z * z) || 1;
  return [x / len, y / len, z / len];
}

function dot3(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: number[], b: number[]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

// --------------------------------------------------------------------------
// Float16 conversion
// --------------------------------------------------------------------------

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

function floatToHalf(value: number): number {
  _f32[0] = value;
  const x = _u32[0];
  const sign = (x >> 16) & 0x8000;
  const exponent = ((x >> 23) & 0xff) - 127 + 15;
  const mantissa = (x >> 13) & 0x03ff;
  if (exponent <= 0) return sign;
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | mantissa;
}

// --------------------------------------------------------------------------
// Cubemap direction from face + UV
// --------------------------------------------------------------------------

function cubemapDir(face: number, u: number, v: number): [number, number, number] {
  switch (face) {
    case 0: return normalize3( 1, -v, -u); // +X
    case 1: return normalize3(-1, -v,  u); // -X
    case 2: return normalize3( u,  1,  v); // +Y
    case 3: return normalize3( u, -1, -v); // -Y
    case 4: return normalize3( u, -v,  1); // +Z
    case 5: return normalize3(-u, -v, -1); // -Z
    default: return [0, 1, 0];
  }
}

// --------------------------------------------------------------------------
// Procedural HDR environment
// --------------------------------------------------------------------------

function envColor(dx: number, dy: number, dz: number): [number, number, number] {
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
  dx /= len; dy /= len; dz /= len;

  const t = Math.max(0, Math.min(1, dy * 0.5 + 0.5));

  // Sky gradient
  let r: number, g: number, b: number;
  if (t > 0.5) {
    const s = (t - 0.5) * 2;
    r = lerp(0.9, 0.35, s);
    g = lerp(0.85, 0.55, s);
    b = lerp(0.8, 1.6, s);
  } else {
    const s = t * 2;
    r = lerp(0.06, 0.9, s);
    g = lerp(0.04, 0.85, s);
    b = lerp(0.03, 0.8, s);
  }

  // Sun disk + glow
  const sx = 0.5, sy = 0.7, sz = 0.3;
  const slen = Math.sqrt(sx * sx + sy * sy + sz * sz);
  const sunDot = Math.max(0, (dx * sx + dy * sy + dz * sz) / slen);
  const sun = Math.pow(sunDot, 512) * 30;
  const glow = Math.pow(sunDot, 8) * 0.8;

  r += sun + glow;
  g += sun * 0.9 + glow * 0.85;
  b += sun * 0.7 + glow * 0.6;

  return [r, g, b];
}

// --------------------------------------------------------------------------
// Quasi-random sampling (Hammersley sequence)
// --------------------------------------------------------------------------

function vanDerCorput(n: number): number {
  let bits = n;
  bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >>> 1);
  bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >>> 2);
  bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >>> 4);
  bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >>> 8);
  bits = (bits << 16) | (bits >>> 16);
  return (bits >>> 0) / 0x100000000;
}

// --------------------------------------------------------------------------
// Importance-sampled GGX for specular prefiltering
// --------------------------------------------------------------------------

function importanceSampleGGX(
  xi0: number, xi1: number,
  N: number[],
  roughness: number,
): [number, number, number] {
  const a = roughness * roughness;
  const a2 = a * a;
  const phi = 2 * Math.PI * xi0;
  const cosTheta = Math.sqrt((1 - xi1) / (1 + (a2 - 1) * xi1));
  const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);

  const hx = sinTheta * Math.cos(phi);
  const hy = sinTheta * Math.sin(phi);
  const hz = cosTheta;

  // Build tangent frame from N
  const up: [number, number, number] = Math.abs(N[1]) < 0.999 ? [0, 1, 0] : [1, 0, 0];
  const T = normalize3(...cross3(up, N));
  const B = cross3(N, T);

  return normalize3(
    T[0] * hx + B[0] * hy + N[0] * hz,
    T[1] * hx + B[1] * hy + N[1] * hz,
    T[2] * hx + B[2] * hy + N[2] * hz,
  );
}

function prefilterEnvMap(
  dir: [number, number, number],
  roughness: number,
  numSamples: number,
): [number, number, number] {
  if (roughness < 0.01) {
    return envColor(dir[0], dir[1], dir[2]);
  }

  const N = dir;
  const V = dir; // V = N approximation (isotropic)

  let totalWeight = 0;
  let r = 0, g = 0, b = 0;

  for (let i = 0; i < numSamples; i++) {
    const xi0 = (i + 0.5) / numSamples;
    const xi1 = vanDerCorput(i);
    const H = importanceSampleGGX(xi0, xi1, N, roughness);

    const vdotH = dot3(V, H);
    const L: [number, number, number] = [
      2 * vdotH * H[0] - V[0],
      2 * vdotH * H[1] - V[1],
      2 * vdotH * H[2] - V[2],
    ];

    const ndotL = dot3(N, L);
    if (ndotL > 0) {
      const [sr, sg, sb] = envColor(L[0], L[1], L[2]);
      r += sr * ndotL;
      g += sg * ndotL;
      b += sb * ndotL;
      totalWeight += ndotL;
    }
  }

  if (totalWeight > 0) {
    return [r / totalWeight, g / totalWeight, b / totalWeight];
  }
  return envColor(dir[0], dir[1], dir[2]);
}

// --------------------------------------------------------------------------
// Cosine-weighted hemisphere for irradiance
// --------------------------------------------------------------------------

function computeIrradiance(
  N: [number, number, number],
  numSamples: number,
): [number, number, number] {
  const up: [number, number, number] = Math.abs(N[1]) < 0.999 ? [0, 1, 0] : [1, 0, 0];
  const T = normalize3(...cross3(up, N));
  const B = cross3(N, T);

  let r = 0, g = 0, b = 0;

  for (let i = 0; i < numSamples; i++) {
    const xi0 = (i + 0.5) / numSamples;
    const xi1 = vanDerCorput(i);

    // Cosine-weighted hemisphere: pdf = cos(theta) / PI
    const phi = 2 * Math.PI * xi0;
    const cosTheta = Math.sqrt(1 - xi1);
    const sinTheta = Math.sqrt(xi1);

    const dx = sinTheta * Math.cos(phi);
    const dy = sinTheta * Math.sin(phi);
    const dz = cosTheta;

    const wx = T[0] * dx + B[0] * dy + N[0] * dz;
    const wy = T[1] * dx + B[1] * dy + N[1] * dz;
    const wz = T[2] * dx + B[2] * dy + N[2] * dz;

    const [sr, sg, sb] = envColor(wx, wy, wz);
    r += sr;
    g += sg;
    b += sb;
  }

  // For cosine-weighted sampling: E = PI * (1/N) * sum(L(wi))
  const scale = Math.PI / numSamples;
  return [r * scale, g * scale, b * scale];
}

// --------------------------------------------------------------------------
// BRDF Integration LUT (split-sum approximation)
// --------------------------------------------------------------------------

function integrateBRDF(NdotV: number, roughness: number, numSamples: number): [number, number] {
  const V: [number, number, number] = [Math.sqrt(1 - NdotV * NdotV), 0, NdotV];
  const N: number[] = [0, 0, 1];
  let A = 0, B = 0;

  for (let i = 0; i < numSamples; i++) {
    const xi0 = (i + 0.5) / numSamples;
    const xi1 = vanDerCorput(i);
    const H = importanceSampleGGX(xi0, xi1, N, roughness);
    const VdotH = V[0] * H[0] + V[1] * H[1] + V[2] * H[2];
    const L: [number, number, number] = [
      2 * VdotH * H[0] - V[0],
      2 * VdotH * H[1] - V[1],
      2 * VdotH * H[2] - V[2],
    ];
    const NdotL = Math.max(L[2], 0);
    const NdotH = Math.max(H[2], 0);
    const VdotH2 = Math.max(VdotH, 0);

    if (NdotL > 0) {
      const a = roughness * roughness;
      const k = a / 2; // IBL variant: k = alpha/2 where alpha = roughness^2
      const G_V = NdotV / (NdotV * (1 - k) + k);
      const G_L = NdotL / (NdotL * (1 - k) + k);
      const G = G_V * G_L;
      const G_Vis = (G * VdotH2) / (NdotH * NdotV + 0.0001);
      const Fc = Math.pow(1 - VdotH2, 5);
      A += (1 - Fc) * G_Vis;
      B += Fc * G_Vis;
    }
  }
  return [A / numSamples, B / numSamples];
}

// --------------------------------------------------------------------------
// Main generation function
// --------------------------------------------------------------------------

export function generateProceduralIBL(device: GPUDevice): ProceduralIBL {
  const SPEC_SIZE = 128;
  const SPEC_MIPS = 8; // log2(128) + 1
  const IRR_SIZE = 32;
  const BRDF_SIZE = 128;
  const SPEC_SAMPLES = 64;
  const IRR_SAMPLES = 128;
  const BRDF_SAMPLES = 64;

  console.time('Procedural IBL generation');

  // === Specular cubemap with roughness mip chain ===
  const specular = device.createTexture({
    size: [SPEC_SIZE, SPEC_SIZE, 6],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    mipLevelCount: SPEC_MIPS,
    dimension: '2d',
    label: 'procedural_specular',
  });

  for (let mip = 0; mip < SPEC_MIPS; mip++) {
    const mipSize = Math.max(1, SPEC_SIZE >> mip);
    const roughness = mip / (SPEC_MIPS - 1);
    const samples = mip === 0 ? 1 : SPEC_SAMPLES;

    for (let face = 0; face < 6; face++) {
      const data = new Uint16Array(mipSize * mipSize * 4);
      let idx = 0;

      for (let y = 0; y < mipSize; y++) {
        const v = -((y + 0.5) / mipSize * 2 - 1);
        for (let x = 0; x < mipSize; x++) {
          const u = (x + 0.5) / mipSize * 2 - 1;
          const dir = cubemapDir(face, u, v);
          const [r, g, b] = prefilterEnvMap(dir, roughness, samples);
          data[idx++] = floatToHalf(r);
          data[idx++] = floatToHalf(g);
          data[idx++] = floatToHalf(b);
          data[idx++] = floatToHalf(1.0);
        }
      }

      device.queue.writeTexture(
        { texture: specular, mipLevel: mip, origin: [0, 0, face] },
        data,
        { bytesPerRow: mipSize * 8 },
        [mipSize, mipSize, 1],
      );
    }
  }

  // === Irradiance cubemap ===
  const irradiance = device.createTexture({
    size: [IRR_SIZE, IRR_SIZE, 6],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    dimension: '2d',
    label: 'procedural_irradiance',
  });

  for (let face = 0; face < 6; face++) {
    const data = new Uint16Array(IRR_SIZE * IRR_SIZE * 4);
    let idx = 0;

    for (let y = 0; y < IRR_SIZE; y++) {
      const v = -((y + 0.5) / IRR_SIZE * 2 - 1);
      for (let x = 0; x < IRR_SIZE; x++) {
        const u = (x + 0.5) / IRR_SIZE * 2 - 1;
        const dir = cubemapDir(face, u, v);
        const [r, g, b] = computeIrradiance(dir, IRR_SAMPLES);
        data[idx++] = floatToHalf(r);
        data[idx++] = floatToHalf(g);
        data[idx++] = floatToHalf(b);
        data[idx++] = floatToHalf(1.0);
      }
    }

    device.queue.writeTexture(
      { texture: irradiance, origin: [0, 0, face] },
      data,
      { bytesPerRow: IRR_SIZE * 8 },
      [IRR_SIZE, IRR_SIZE, 1],
    );
  }

  // === BRDF Integration LUT ===
  const brdfLut = device.createTexture({
    size: [BRDF_SIZE, BRDF_SIZE],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    label: 'brdf_lut',
  });

  {
    const data = new Uint16Array(BRDF_SIZE * BRDF_SIZE * 4);
    let idx = 0;
    for (let y = 0; y < BRDF_SIZE; y++) {
      const roughness = (y + 0.5) / BRDF_SIZE;
      for (let x = 0; x < BRDF_SIZE; x++) {
        const NdotV = Math.max((x + 0.5) / BRDF_SIZE, 0.001);
        const [a, b] = integrateBRDF(NdotV, roughness, BRDF_SAMPLES);
        data[idx++] = floatToHalf(a);
        data[idx++] = floatToHalf(b);
        data[idx++] = floatToHalf(0);
        data[idx++] = floatToHalf(1);
      }
    }
    device.queue.writeTexture(
      { texture: brdfLut },
      data,
      { bytesPerRow: BRDF_SIZE * 8 },
      [BRDF_SIZE, BRDF_SIZE],
    );
  }

  console.timeEnd('Procedural IBL generation');

  const specularView = specular.createView({ dimension: 'cube' });
  const irradianceView = irradiance.createView({ dimension: 'cube' });
  const brdfLutView = brdfLut.createView();

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });

  return { specular, irradiance, brdfLut, specularView, irradianceView, brdfLutView, sampler };
}
