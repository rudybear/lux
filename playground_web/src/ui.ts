/**
 * UI controls: sidebar sliders, FPS counter, drag-and-drop, scene selector.
 */

import type { Material } from './scene';

export interface UIState {
  exposure: number;
  lightDirY: number;
  selectedScene: string;
}

export type SceneChangeCallback = (scene: string) => void;
export type FileDropCallback = (buffer: ArrayBuffer, name: string) => void;

export class UI {
  private _fpsEl: HTMLElement;
  private _frameTimes: number[] = [];
  private _state: UIState = {
    exposure: 1.0,
    lightDirY: 0.7,
    selectedScene: 'pbr_basic',
  };

  /** Currently loaded materials for the explorer. */
  private _materials: Material[] = [];
  private _selectedMaterialIndex = 0;

  onSceneChange: SceneChangeCallback | null = null;
  onFileDrop: FileDropCallback | null = null;
  onScreenshot: (() => void) | null = null;
  onMaterialChange: ((index: number, field: string, value: number | number[]) => void) | null = null;

  constructor() {
    this._fpsEl = document.getElementById('fps')!;

    // Wire up sliders
    this._bindSlider('exposure', v => { this._state.exposure = v; });
    this._bindSlider('light-y', v => { this._state.lightDirY = v; });

    // Scene selector
    const select = document.getElementById('scene-select') as HTMLSelectElement;
    select?.addEventListener('change', () => {
      this._state.selectedScene = select.value;
      this.onSceneChange?.(select.value);
    });

    // Screenshot button
    const screenshotBtn = document.getElementById('screenshot-btn');
    screenshotBtn?.addEventListener('click', () => this.onScreenshot?.());

    // Drag and drop
    this._setupDragDrop();
  }

  get state(): Readonly<UIState> {
    return this._state;
  }

  /** Populate the material explorer with scene materials. */
  setMaterials(materials: Material[]): void {
    this._materials = materials;
    this._selectedMaterialIndex = 0;

    const select = document.getElementById('material-select') as HTMLSelectElement;
    if (!select) return;
    select.innerHTML = '';

    materials.forEach((mat, i) => {
      const opt = document.createElement('option');
      opt.value = String(i);
      const flags = [
        mat.hasClearcoat ? 'coat' : '',
        mat.hasSheen ? 'sheen' : '',
        mat.hasTransmission ? 'trans' : '',
        mat.isUnlit ? 'unlit' : '',
      ].filter(Boolean).join('+');
      opt.textContent = `Material ${i} (${mat.alphaMode}${flags ? ' ' + flags : ''})`;
      select.appendChild(opt);
    });

    select.addEventListener('change', () => {
      this._selectedMaterialIndex = parseInt(select.value);
      this._renderMaterialProps();
    });

    this._renderMaterialProps();
  }

  /** Update the scene info panel with loaded scene metadata. */
  updateSceneInfo(info: {
    meshes: number;
    materials: number;
    lights: number;
    drawRanges: number;
    vertices: number;
    triangles: number;
    bounds: { min: [number, number, number]; max: [number, number, number] };
    materialNames?: string[];
    lightDescriptions?: string[];
  }): void {
    const infoEl = document.getElementById('scene-info');
    if (infoEl) {
      const bMin = info.bounds.min.map(v => v.toFixed(2)).join(', ');
      const bMax = info.bounds.max.map(v => v.toFixed(2)).join(', ');
      const fmtNum = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : `${n}`;
      infoEl.innerHTML = [
        `Vertices: ${fmtNum(info.vertices)} | Triangles: ${fmtNum(info.triangles)}`,
        `Meshes: ${info.meshes} | Materials: ${info.materials}`,
        `Lights: ${info.lights} | Draw calls: ${info.drawRanges}`,
        `Bounds: [${bMin}]`,
        `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[${bMax}]`,
      ].join('<br>');
    }

    const lightEl = document.getElementById('light-list');
    if (lightEl && info.lightDescriptions) {
      lightEl.innerHTML = info.lightDescriptions.map((desc, i) =>
        `<div style="padding:2px 0;color:#aab">${i}: ${desc}</div>`,
      ).join('');
    }
  }

  /** Call each frame with delta time in seconds. */
  updateFPS(dt: number): void {
    this._frameTimes.push(dt);
    if (this._frameTimes.length > 60) this._frameTimes.shift();
    const avg = this._frameTimes.reduce((a, b) => a + b, 0) / this._frameTimes.length;
    const fps = avg > 0 ? 1 / avg : 0;
    this._fpsEl.textContent = `${fps.toFixed(0)} fps`;
  }

  private _renderMaterialProps(): void {
    const container = document.getElementById('material-props');
    if (!container || !this._materials.length) return;

    const mat = this._materials[this._selectedMaterialIndex];
    if (!mat) return;

    container.innerHTML = '';
    const idx = this._selectedMaterialIndex;

    // Helper to create a slider row
    const addSlider = (label: string, field: string, value: number, min: number, max: number, step: number) => {
      const row = document.createElement('div');
      row.innerHTML = `
        <label style="display:block;font-size:11px;margin-top:6px;color:#a0a0c0">${label}</label>
        <div class="slider-row">
          <input type="range" min="${min}" max="${max}" step="${step}" value="${value}" style="flex:1" />
          <span style="font-size:11px;min-width:36px;text-align:right;font-family:monospace">${value.toFixed(2)}</span>
        </div>`;
      const input = row.querySelector('input')!;
      const span = row.querySelector('span')!;
      input.addEventListener('input', () => {
        const v = parseFloat(input.value);
        span.textContent = v.toFixed(2);
        this.onMaterialChange?.(idx, field, v);
      });
      container.appendChild(row);
    };

    // Helper to create a color display with RGB values
    const addColor = (label: string, _field: string, rgb: number[]) => {
      const r = Math.round(rgb[0] * 255);
      const g = Math.round(rgb[1] * 255);
      const b = Math.round(rgb[2] * 255);
      const row = document.createElement('div');
      row.innerHTML = `
        <label style="display:block;font-size:11px;margin-top:6px;color:#a0a0c0">${label}</label>
        <div style="display:flex;align-items:center;gap:6px;margin-top:2px">
          <div style="width:20px;height:20px;border-radius:3px;border:1px solid #555;background:rgb(${r},${g},${b})"></div>
          <span style="font-size:10px;font-family:monospace;color:#888">(${rgb[0].toFixed(2)}, ${rgb[1].toFixed(2)}, ${rgb[2].toFixed(2)})</span>
        </div>`;
      container.appendChild(row);
    };

    // Core PBR properties
    addColor('Base Color', 'baseColor', [mat.baseColor[0], mat.baseColor[1], mat.baseColor[2]]);
    addSlider('Metallic', 'metallic', mat.metallic, 0, 1, 0.01);
    addSlider('Roughness', 'roughness', mat.roughness, 0, 1, 0.01);
    addSlider('IOR', 'ior', mat.ior, 1.0, 3.0, 0.01);

    // Emissive
    if (mat.emissive[0] > 0 || mat.emissive[1] > 0 || mat.emissive[2] > 0 || mat.emissiveStrength > 0) {
      addColor('Emissive', 'emissive', mat.emissive);
      addSlider('Emissive Strength', 'emissiveStrength', mat.emissiveStrength, 0, 10, 0.1);
    }

    // Clearcoat
    if (mat.hasClearcoat) {
      addSlider('Clearcoat', 'clearcoatFactor', mat.clearcoatFactor, 0, 1, 0.01);
      addSlider('Clearcoat Roughness', 'clearcoatRoughnessFactor', mat.clearcoatRoughnessFactor, 0, 1, 0.01);
    }

    // Sheen
    if (mat.hasSheen) {
      addColor('Sheen Color', 'sheenColorFactor', mat.sheenColorFactor);
      addSlider('Sheen Roughness', 'sheenRoughnessFactor', mat.sheenRoughnessFactor, 0, 1, 0.01);
    }

    // Transmission
    if (mat.hasTransmission) {
      addSlider('Transmission', 'transmissionFactor', mat.transmissionFactor, 0, 1, 0.01);
    }

    // Alpha info
    const alphaInfo = document.createElement('div');
    alphaInfo.innerHTML = `<div style="font-size:10px;margin-top:8px;padding:4px;background:#0a0a2a;border-radius:3px;color:#666">
      Alpha: ${mat.alphaMode}${mat.alphaMode === 'MASK' ? ` (cutoff ${mat.alphaCutoff})` : ''} |
      ${mat.doubleSided ? 'Double-sided' : 'Single-sided'} |
      Textures: ${mat.textures.size}
    </div>`;
    container.appendChild(alphaInfo);
  }

  private _bindSlider(id: string, cb: (v: number) => void): void {
    const slider = document.getElementById(id) as HTMLInputElement;
    const valEl = document.getElementById(`${id}-val`);
    if (!slider) return;

    const update = () => {
      const v = parseFloat(slider.value);
      cb(v);
      if (valEl) valEl.textContent = v.toFixed(2);
    };

    slider.addEventListener('input', update);
    update(); // initial
  }

  private _setupDragDrop(): void {
    const container = document.getElementById('canvas-container')!;
    const overlay = document.getElementById('drop-overlay')!;

    container.addEventListener('dragenter', (e) => {
      e.preventDefault();
      overlay.style.display = 'flex';
    });

    container.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    container.addEventListener('dragleave', (e) => {
      if (e.relatedTarget && container.contains(e.relatedTarget as Node)) return;
      overlay.style.display = 'none';
    });

    container.addEventListener('drop', async (e) => {
      e.preventDefault();
      overlay.style.display = 'none';

      const file = e.dataTransfer?.files[0];
      if (!file) return;

      const name = file.name.toLowerCase();
      if (name.endsWith('.glb') || name.endsWith('.gltf')) {
        const buffer = await file.arrayBuffer();
        this.onFileDrop?.(buffer, file.name);
      }
    });
  }
}

/** Set up ResizeObserver for canvas auto-resize. */
export function observeCanvasResize(
  canvas: HTMLCanvasElement,
  onResize: (width: number, height: number) => void,
): void {
  const observer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const { width, height } = entry.contentRect;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(width * dpr));
      const h = Math.max(1, Math.floor(height * dpr));
      canvas.width = w;
      canvas.height = h;
      onResize(w, h);
    }
  });
  observer.observe(canvas);
}
