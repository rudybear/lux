#pragma once

// Shared orbit-camera math for the juggle_p0.8_stride4/juggle_full_stride4
// live-reconstruction demo scene -- used by BOTH playground_ios/Source/
// SplatView.mm (the live iOS demo) and metal_main.cpp's `--live-bench` CLI
// mode, so the Mac benchmark drives MetalLiveReconstruct with the exact same
// per-frame camera/jitter sequence the iPad does. Moved out of SplatView.mm
// (which used to have its own private copy of this) so there's one source of
// truth for both hosts.
//
// Matches mobiledlss/datagen/camera.py::orbit_path exactly (OpenCV viewmat_cv
// + pinhole K, converted via DlssIO::cvViewToGl / DlssIO::
// buildIntrinsicsProjection(metalYConvention=true for the CLI camera-json
// path, false for MetalSplatLuxcRenderer's own transpiled-from-Vulkan
// convention -- see metal_splat_luxc_renderer.h's kMetalYConvention comment)
// -- verified against the macOS CLI binary (lux-playground-metal
// --camera-json ...) before this was first written into SplatView.mm: frame
// 0 at radius=2.0/elevation=15deg/center=fg_center reproduces the expected
// framing of the juggling actor.

#include <glm/glm.hpp>
#include <cstdint>

namespace LiveOrbitCamera {

// --- Scene-specific constants (demo/data/scene_meta.json for
// juggle_p0.8_stride4.glb / juggle_p0.8_stride2.glb -- see
// demo/ios_assets/README in mobiledlss) ---
constexpr glm::vec3 kFgCenter(-0.2183254063129425f, -1.0417373180389404f, 0.08556576073169708f);

// --- Orbit camera (per task spec): radius 2.0, elevation 15deg,
// 0.6deg/frame, fovY 50deg (mobiledlss/datagen/camera.py::orbit_path
// defaults), y-down world (PanopticSports convention). ---
constexpr float kOrbitRadius = 2.0f;
constexpr float kElevationDeg = 15.0f;
constexpr float kDegPerFrame = 0.6f;
constexpr float kFovYDeg = 50.0f;

struct Result {
    glm::vec3 eye, r, u, f;
    glm::mat4 viewGl{1.0f};
    glm::mat4 proj{1.0f};
    float fx = 0.0f, fy = 0.0f;
};

// Builds the OpenCV-convention world->camera viewmat + GL view/proj matrices
// for orbit frame `frame` at resolution `width x height`. `metalYConvention`:
// false for MetalSplatLuxcRenderer (its transpiled-from-Vulkan vertex shader
// has its own Y handling -- see kMetalYConvention), true for anything using
// the hand-written Metal convention.
Result compute(int frame, uint32_t width, uint32_t height, bool metalYConvention);

// mobiledlss/datagen/camera.py::halton (1-indexed Halton low-discrepancy sequence).
float halton(int index, int base);

// mobiledlss/datagen/camera.py::taa_jitter -- Halton(2,3) TAA jitter in
// TARGET-pixel units, cycling every `period` frames. Callers wanting the
// proxy-pixel jitter (what ProxyRenderer::setJitter expects when applied to
// a proxy-res renderer) must scale by (proxyW/targetW) themselves
// (mobiledlss/train/data.py's "jitter_proxy = jitter / S" convention).
void taaJitterTargetPx(int frame, int period, float& jxOut, float& jyOut);

}  // namespace LiveOrbitCamera
