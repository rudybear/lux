/**
 * UI controls: sidebar sliders, FPS counter, drag-and-drop, scene selector.
 */

export interface UIState {
  metallic: number;
  roughness: number;
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
    metallic: 0,
    roughness: 0.5,
    exposure: 1.0,
    lightDirY: 0.7,
    selectedScene: 'pbr_basic',
  };

  onSceneChange: SceneChangeCallback | null = null;
  onFileDrop: FileDropCallback | null = null;
  onScreenshot: (() => void) | null = null;

  constructor() {
    this._fpsEl = document.getElementById('fps')!;

    // Wire up sliders
    this._bindSlider('metallic', v => { this._state.metallic = v; });
    this._bindSlider('roughness', v => { this._state.roughness = v; });
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

  /** Update the scene info panel with loaded scene metadata. */
  updateSceneInfo(info: {
    meshes: number;
    materials: number;
    lights: number;
    drawRanges: number;
    bounds: { min: [number, number, number]; max: [number, number, number] };
    materialNames?: string[];
    lightDescriptions?: string[];
  }): void {
    const infoEl = document.getElementById('scene-info');
    if (infoEl) {
      const bMin = info.bounds.min.map(v => v.toFixed(2)).join(', ');
      const bMax = info.bounds.max.map(v => v.toFixed(2)).join(', ');
      infoEl.innerHTML = [
        `Meshes: ${info.meshes}`,
        `Materials: ${info.materials}`,
        `Lights: ${info.lights}`,
        `Draw calls: ${info.drawRanges}`,
        `Bounds: [${bMin}]`,
        `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[${bMax}]`,
      ].join('<br>');
    }

    const matEl = document.getElementById('material-list');
    if (matEl && info.materialNames) {
      matEl.innerHTML = info.materialNames.map((name, i) =>
        `<div style="padding:2px 0;color:#aab">${i}: ${name}</div>`,
      ).join('');
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
