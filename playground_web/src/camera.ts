/**
 * Orbit camera with mouse/touch controls.
 */
import { mat4, vec3 } from 'gl-matrix';

export class OrbitCamera {
  /** Spherical coordinates: azimuth (radians) */
  azimuth = 0;
  /** Spherical coordinates: elevation (radians) */
  elevation = 0.4;
  /** Distance from target */
  distance = 3.0;
  /** Look-at target */
  target = vec3.fromValues(0, 0, 0);

  /** Near/far clip */
  near = 0.01;
  far = 100.0;
  /** Field of view (radians) */
  fov = Math.PI / 4;

  private _dragging = false;
  private _panning = false;
  private _lastX = 0;
  private _lastY = 0;

  /** Configure camera to frame scene bounds (matches C++ computeAutoCamera). */
  frameScene(
    boundsMin: [number, number, number],
    boundsMax: [number, number, number],
  ): void {
    const cx = (boundsMin[0] + boundsMax[0]) * 0.5;
    const cy = (boundsMin[1] + boundsMax[1]) * 0.5;
    const cz = (boundsMin[2] + boundsMax[2]) * 0.5;
    this.target = vec3.fromValues(cx, cy, cz);

    const dx = boundsMax[0] - boundsMin[0];
    const dy = boundsMax[1] - boundsMin[1];
    const dz = boundsMax[2] - boundsMin[2];
    const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5;

    this.distance = radius * 2.5;
    this.far = radius * 10;
    this.near = Math.max(0.01, radius * 0.01);
    this.elevation = 0.2;
    this.azimuth = Math.PI * 0.25;
  }

  /** Attach mouse/touch listeners to a canvas element. */
  attach(canvas: HTMLCanvasElement): void {
    canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
    canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
    canvas.addEventListener('mouseup', () => this._onMouseUp());
    canvas.addEventListener('mouseleave', () => this._onMouseUp());
    canvas.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    // Touch support
    canvas.addEventListener('touchstart', (e) => this._onTouchStart(e), { passive: false });
    canvas.addEventListener('touchmove', (e) => this._onTouchMove(e), { passive: false });
    canvas.addEventListener('touchend', () => this._onMouseUp());
  }

  /** Compute view matrix. */
  getViewMatrix(out: mat4): mat4 {
    const eye = this._getEyePosition();
    mat4.lookAt(out, eye, this.target, [0, 1, 0]);
    return out;
  }

  /** Compute projection matrix (WebGPU depth range [0,1]). */
  getProjectionMatrix(out: mat4, aspect: number): mat4 {
    mat4.perspective(out, this.fov, aspect, this.near, this.far);
    // gl-matrix produces OpenGL [-1,1] depth; remap to WebGPU [0,1]
    out[10] = out[10] * 0.5 + out[11] * 0.5;
    out[14] = out[14] * 0.5 + out[15] * 0.5;
    return out;
  }

  /** Get eye position in world space. */
  getEyePosition(): vec3 {
    return this._getEyePosition();
  }

  private _getEyePosition(): vec3 {
    const x = this.distance * Math.cos(this.elevation) * Math.sin(this.azimuth);
    const y = this.distance * Math.sin(this.elevation);
    const z = this.distance * Math.cos(this.elevation) * Math.cos(this.azimuth);
    return vec3.fromValues(
      this.target[0] + x,
      this.target[1] + y,
      this.target[2] + z,
    );
  }

  private _onMouseDown(e: MouseEvent): void {
    if (e.button === 0) {
      this._dragging = true;
    } else if (e.button === 1 || e.button === 2) {
      this._panning = true;
    }
    this._lastX = e.clientX;
    this._lastY = e.clientY;
  }

  private _onMouseMove(e: MouseEvent): void {
    const dx = e.clientX - this._lastX;
    const dy = e.clientY - this._lastY;
    this._lastX = e.clientX;
    this._lastY = e.clientY;

    if (this._dragging) {
      this.azimuth -= dx * 0.005;
      this.elevation += dy * 0.005;
      this.elevation = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.elevation));
    } else if (this._panning) {
      const panSpeed = this.distance * 0.002;
      const right = vec3.fromValues(
        Math.cos(this.azimuth), 0, -Math.sin(this.azimuth),
      );
      const up = vec3.fromValues(0, 1, 0);
      vec3.scaleAndAdd(this.target, this.target, right, -dx * panSpeed);
      vec3.scaleAndAdd(this.target, this.target, up, dy * panSpeed);
    }
  }

  private _onMouseUp(): void {
    this._dragging = false;
    this._panning = false;
  }

  private _onWheel(e: WheelEvent): void {
    e.preventDefault();
    this.distance *= 1 + e.deltaY * 0.001;
    this.distance = Math.max(0.1, Math.min(100, this.distance));
  }

  private _prevTouchDist = 0;
  private _onTouchStart(e: TouchEvent): void {
    e.preventDefault();
    if (e.touches.length === 1) {
      this._dragging = true;
      this._lastX = e.touches[0].clientX;
      this._lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      this._dragging = false;
      const dx = e.touches[1].clientX - e.touches[0].clientX;
      const dy = e.touches[1].clientY - e.touches[0].clientY;
      this._prevTouchDist = Math.hypot(dx, dy);
    }
  }

  private _onTouchMove(e: TouchEvent): void {
    e.preventDefault();
    if (e.touches.length === 1 && this._dragging) {
      const dx = e.touches[0].clientX - this._lastX;
      const dy = e.touches[0].clientY - this._lastY;
      this._lastX = e.touches[0].clientX;
      this._lastY = e.touches[0].clientY;
      this.azimuth -= dx * 0.005;
      this.elevation += dy * 0.005;
      this.elevation = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.elevation));
    } else if (e.touches.length === 2) {
      const dx = e.touches[1].clientX - e.touches[0].clientX;
      const dy = e.touches[1].clientY - e.touches[0].clientY;
      const dist = Math.hypot(dx, dy);
      if (this._prevTouchDist > 0) {
        this.distance *= this._prevTouchDist / dist;
        this.distance = Math.max(0.1, Math.min(100, this.distance));
      }
      this._prevTouchDist = dist;
    }
  }
}
