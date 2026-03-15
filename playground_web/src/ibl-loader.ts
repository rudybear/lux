/**
 * IBL (Image-Based Lighting) cubemap loader for WebGPU.
 *
 * Loads preprocessed IBL binaries from public/assets/ibl/:
 *   specular.bin   — float16 RGBA, 6 faces x mip levels
 *   irradiance.bin — float16 RGBA, 6 faces
 *   brdf_lut.bin   — float16 RG, 512x512
 *   manifest.json  — dimensions, mip counts, face order
 */

export interface IBLTextures {
  specular: GPUTexture;
  irradiance: GPUTexture;
  brdfLut: GPUTexture;
  specularView: GPUTextureView;
  irradianceView: GPUTextureView;
  brdfLutView: GPUTextureView;
  sampler: GPUSampler;
}

interface IBLManifest {
  specular: {
    size: number;
    mipLevels: number;
    format: string;
  };
  irradiance: {
    size: number;
    format: string;
  };
  brdfLut: {
    width: number;
    height: number;
    format: string;
  };
}

export async function loadIBL(
  device: GPUDevice,
  basePath: string,
): Promise<IBLTextures> {
  // Load manifest
  const manifestRes = await fetch(`${basePath}/manifest.json`);
  if (!manifestRes.ok) throw new Error(`IBL manifest not found at ${basePath}/manifest.json`);
  const manifest: IBLManifest = await manifestRes.json();

  // Load binary files in parallel
  const [specData, irrData, brdfData] = await Promise.all([
    fetch(`${basePath}/specular.bin`).then(r => r.arrayBuffer()),
    fetch(`${basePath}/irradiance.bin`).then(r => r.arrayBuffer()),
    fetch(`${basePath}/brdf_lut.bin`).then(r => r.arrayBuffer()),
  ]);

  // Create specular cubemap with mips
  const specSize = manifest.specular.size;
  const specMips = manifest.specular.mipLevels;
  const specular = device.createTexture({
    size: [specSize, specSize, 6],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    mipLevelCount: specMips,
    dimension: '2d',
  });

  // Upload specular faces + mips
  let specOffset = 0;
  for (let mip = 0; mip < specMips; mip++) {
    const mipSize = Math.max(1, specSize >> mip);
    const faceBytes = mipSize * mipSize * 8; // rgba16float = 8 bytes/pixel
    for (let face = 0; face < 6; face++) {
      device.queue.writeTexture(
        { texture: specular, mipLevel: mip, origin: [0, 0, face] },
        new Uint8Array(specData, specOffset, faceBytes),
        { bytesPerRow: mipSize * 8, rowsPerImage: mipSize },
        [mipSize, mipSize, 1],
      );
      specOffset += faceBytes;
    }
  }

  // Create irradiance cubemap
  const irrSize = manifest.irradiance.size;
  const irradiance = device.createTexture({
    size: [irrSize, irrSize, 6],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    dimension: '2d',
  });

  let irrOffset = 0;
  const irrFaceBytes = irrSize * irrSize * 8;
  for (let face = 0; face < 6; face++) {
    device.queue.writeTexture(
      { texture: irradiance, origin: [0, 0, face] },
      new Uint8Array(irrData, irrOffset, irrFaceBytes),
      { bytesPerRow: irrSize * 8, rowsPerImage: irrSize },
      [irrSize, irrSize, 1],
    );
    irrOffset += irrFaceBytes;
  }

  // Create BRDF LUT
  const brdfW = manifest.brdfLut.width;
  const brdfH = manifest.brdfLut.height;
  const brdfLut = device.createTexture({
    size: [brdfW, brdfH],
    format: 'rg16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  device.queue.writeTexture(
    { texture: brdfLut },
    brdfData,
    { bytesPerRow: brdfW * 4 }, // rg16float = 4 bytes/pixel
    [brdfW, brdfH],
  );

  // Create views
  const specularView = specular.createView({ dimension: 'cube' });
  const irradianceView = irradiance.createView({ dimension: 'cube' });
  const brdfLutView = brdfLut.createView();

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });

  return {
    specular, irradiance, brdfLut,
    specularView, irradianceView, brdfLutView,
    sampler,
  };
}
