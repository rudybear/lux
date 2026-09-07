// Stage A: live orbit-camera splat rendering on device.
//
// Reuses playground_cpp/src's Metal pipeline classes exactly as the macOS
// interactive loop (metal_main.cpp) does -- MetalContext (initHeadless(),
// then pointed at our own CAMetalLayer instead of a GLFW window),
// MetalSceneManager::loadScene()/hasSplatData()/getSplatData(), and
// MetalSplatLuxcRenderer::{init,updateCameraExplicit,setPreviousCameraExplicit,
// setMorphTime,seedPreviousMorphTime,render}. No modifications to any of
// those files -- see playground_ios/README.md.
//
// Target mode originally rendered via the hand-written MetalSplatRenderer
// (CPU sort, no `sort:` source of truth -- visibly wrong splat ordering).
// It now uses a second MetalSplatLuxcRenderer instance (same shaderBase as
// the proxy pass, `sort: view_depth` baked into the compiled pipeline) at
// full 960x540 resolution, rendered offscreen and copied into the drawable
// via the same un-premultiply compute kernel Proxy/Bicubic use (scale=1).
//
// Camera convention matches mobiledlss/datagen/camera.py::orbit_path exactly
// (OpenCV viewmat_cv + pinhole K, converted via DlssIO::cvViewToGl /
// DlssIO::buildIntrinsicsProjection(metalYConvention=true), the same path
// metal_main.cpp's --camera-json takes) -- verified against the macOS CLI
// binary (lux-playground-metal --camera-json ...) before writing this file:
// frame 0 at radius=2.0/elevation=15deg/center=fg_center reproduces the
// expected framing of the juggling actor.
#import "SplatView.h"
#import <QuartzCore/QuartzCore.h>
#import <Metal/Metal.h>

#include "metal_context.h"
#include "metal_scene_manager.h"
#include "metal_splat_luxc_renderer.h"
#include "dlss_io.h"
#include "metal_screenshot.h"
#include "NetInputAssembly.h"
#include "MPSGraphUNet.h"
#include "ReconstructPass.h"

#include <glm/glm.hpp>
#include <array>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <unistd.h>

// B1 update (per coordinator): the luxc-transpiled splat backend
// (metal_splat_luxc_renderer.*) is now at parity with Vulkan/gsplat (commit
// 08cfe8e: 51.3dB vs gsplat, 83.7dB vs Vulkan) and is the network's actual
// training-time renderer, so both the proxy pass and Target mode use it
// (the hand-written MetalSplatRenderer path was removed entirely -- see
// 80e33f4). getAuxTexture()/getFgTexture() (lux 6ed0334's packed DLSS
// attachment layout) are MetalSplatLuxcRenderer-only, so this alias is no
// longer a drop-in fallback point the way it once was.
using ProxyRenderer = MetalSplatLuxcRenderer;

// --- Scene-specific constants (demo/data/scene_meta.json for
// juggle_p0.8_stride4.glb / juggle_p0.8_stride2.glb -- see
// demo/ios_assets/README in mobiledlss) ---
static const glm::vec3 kFgCenter(-0.2183254063129425f, -1.0417373180389404f, 0.08556576073169708f);

// --- Orbit camera (per task spec): radius 2.0, elevation 15deg,
// 0.6deg/frame, fovY 50deg (mobiledlss/datagen/camera.py::orbit_path
// defaults), y-down world (PanopticSports convention). ---
static const float kOrbitRadius = 2.0f;
static const float kElevationDeg = 15.0f;
static const float kDegPerFrame = 0.6f;
static const float kFovYDeg = 50.0f;
static const float kSimFps = 30.0f;  // frame-index <-> morph-time mapping cadence

// Target display resolution (task spec).
static const uint32_t kTargetW = 960;
static const uint32_t kTargetH = 540;

// Proxy resolution (B1: s=2 downscale of target, per task spec's per-frame
// pipeline). S = kTargetW/kProxyW = kTargetH/kProxyH = 2.
static const uint32_t kProxyW = 480;
static const uint32_t kProxyH = 270;
static const float kProxyScale = static_cast<float>(kProxyW) / static_cast<float>(kTargetW);  // 0.5

// One-shot proxy aux dump for B1 validation against the macOS CLI, at this
// frame index (has both real camera motion and real actor motion, unlike
// frame 0). See -dumpProxyFrame.
static const int kProxyDumpFrame = 10;

// Toggle to "juggle_p0.8_stride2" once app size / load time budget allows
// (see task notes: stride4 first for size). Proxy/Reconstruction render from
// this 80%-background-pruned scene -- matches what the network was trained
// on (mobiledlss/datagen/make_clips.py::render_clip's `proxy` split).
static NSString *const kSceneAssetName = @"juggle_p0.8_stride4";

// Target mode's OWN scene, unpruned (336,568 gaussians, stride 4) --
// mobiledlss/reports/ipad_recon_gap.md's root-cause finding: Target used to
// render the SAME pruned scene the network reconstructs *toward*, so the
// on-device PSNR reference was itself missing 80% of the background (a
// ~22.5dB ceiling on its own, independent of any reconstruction defect).
// mobiledlss/datagen/make_clips.py::render_clip always renders `target_*`
// from the unpruned scene and `proxy_*` from the pruned one -- this matches
// that convention. Target-mode display/PSNR-reference only; the proxy/
// reconstruction pipeline (_splatRProxy) keeps using kSceneAssetName above.
static NSString *const kSceneAssetNameFullTarget = @"juggle_full_stride4";

// Builds the OpenCV-convention world->camera viewmat + GL view/proj matrices
// for orbit frame `frame` at resolution `width x height`, exactly matching
// mobiledlss/datagen/camera.py::look_at + orbit_path + intrinsics, and the
// DlssIO conversion metal_main.cpp's --camera-json path uses. Used for both
// the target-res live view and the proxy-res DLSS pass (B1) -- orbit_path's
// own `intrinsics(w,h,fovYdeg)` scales linearly with height, so calling this
// directly at proxy resolution is equivalent to `Camera.scaled()`.
static void computeOrbitCamera(int frame, uint32_t width, uint32_t height, bool metalYConvention,
                                glm::vec3 &eyeOut, glm::vec3 &rOut, glm::vec3 &uOut, glm::vec3 &fOut,
                                glm::mat4 &viewGlOut, glm::mat4 &projOut, float &fxOut, float &fyOut) {
    const float elRad = glm::radians(kElevationDeg);
    const float az = glm::radians(kDegPerFrame * static_cast<float>(frame));
    const glm::vec3 eye = kFgCenter + kOrbitRadius * glm::vec3(cosf(elRad) * cosf(az),
                                                                -sinf(elRad),
                                                                cosf(elRad) * sinf(az));
    // PanopticSports is a y-down world -- world "up" for the look_at basis is (0,-1,0).
    const glm::vec3 up(0.0f, -1.0f, 0.0f);
    const glm::vec3 f = glm::normalize(kFgCenter - eye);
    const glm::vec3 r = glm::normalize(glm::cross(-up, f));
    const glm::vec3 u = glm::cross(f, r);

    std::array<float, 16> viewCv = {
        r.x, r.y, r.z, -glm::dot(r, eye),
        u.x, u.y, u.z, -glm::dot(u, eye),
        f.x, f.y, f.z, -glm::dot(f, eye),
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    const glm::mat4 viewGl = DlssIO::cvViewToGl(viewCv);

    const float fy = 0.5f * static_cast<float>(height) / tanf(glm::radians(kFovYDeg) * 0.5f);
    const float fx = fy;
    const float cx = 0.5f * static_cast<float>(width);
    const float cy = 0.5f * static_cast<float>(height);
    const glm::mat4 proj = DlssIO::buildIntrinsicsProjection(
        fx, fy, cx, cy, static_cast<float>(width), static_cast<float>(height),
        0.01f, 1000.0f, metalYConvention);

    eyeOut = eye;
    rOut = r; uOut = u; fOut = f;
    viewGlOut = viewGl;
    projOut = proj;
    fxOut = fx;
    fyOut = fy;
}

// mobiledlss/datagen/camera.py::halton (1-indexed Halton low-discrepancy sequence).
static float haltonSeq(int index, int base) {
    float f = 1.0f, r = 0.0f;
    int i = index;
    while (i > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(i % base);
        i /= base;
    }
    return r;
}

// Display modes (B5, pulled forward per coordinator priority change): tap
// cycles through these. Target is the only one that pays for the full-res
// (960x540) splat pass -- Proxy/Bicubic/Reconstruction all display a cheap
// upscale of the *proxy* pass's own colour output instead, per the
// coordinator's explicit "Reconstruction mode must not render the full-res
// pass" instruction. The proxy+net(+reconstruct, once B4 lands) pipeline
// itself always runs every frame regardless of display mode, so the
// recurrent hidden state stays continuous across mode switches.
typedef NS_ENUM(NSInteger, DisplayMode) {
    DisplayModeProxy = 0,
    DisplayModeBicubic,
    DisplayModeReconstruction,
    DisplayModeTarget,
    DisplayModeCount,
};
static const char *kDisplayModeNames[DisplayModeCount] = {"Proxy", "Bilinear", "Reconstruction", "Target"};

// Reads LUX_START_MODE (env var, or -LUX_START_MODE launch argument, which
// NSUserDefaults also surfaces as a value under the same key) so a mode can
// be selected at launch via `devicectl device process launch
// --environment-variables LUX_START_MODE=Target` -- there's no way to tap
// the view remotely over devicectl. One of Target|Proxy|Bicubic|Reconstruction
// (case-insensitive); defaults to Reconstruction if unset/unrecognized.
static DisplayMode startModeFromEnvironment() {
    NSString *val = [[NSProcessInfo processInfo].environment objectForKey:@"LUX_START_MODE"];
    if (!val.length) {
        val = [[NSUserDefaults standardUserDefaults] stringForKey:@"LUX_START_MODE"];
    }
    if (!val.length) return DisplayModeReconstruction;
    NSString *lower = val.lowercaseString;
    if ([lower isEqualToString:@"target"]) return DisplayModeTarget;
    if ([lower isEqualToString:@"proxy"]) return DisplayModeProxy;
    if ([lower isEqualToString:@"bicubic"] || [lower isEqualToString:@"bilinear"]) return DisplayModeBicubic;
    if ([lower isEqualToString:@"reconstruction"]) return DisplayModeReconstruction;
    NSLog(@"[SplatView] LUX_START_MODE='%@' not recognized, defaulting to Reconstruction", val);
    return DisplayModeReconstruction;
}

// Reads an int env var (or -Key launch argument via NSUserDefaults, same
// convention as LUX_START_MODE above), falling back to `def` if unset/empty.
static int intFromEnvironment(NSString *key, int def) {
    NSString *val = [[NSProcessInfo processInfo].environment objectForKey:key];
    if (!val.length) val = [[NSUserDefaults standardUserDefaults] stringForKey:key];
    if (!val.length) return def;
    return [val intValue];
}

// LUX_PSNR_FRAMES (task spec, mobiledlss/reports/rollout_drift.md follow-up):
// length of a continuous, un-windowed Reconstruction-vs-Target/Bicubic-vs-
// Target PSNR capture, in frames -- 0 (default) disables it entirely (no
// extra Target-mode render, no extra readbacks, same behavior as before this
// feature existed). Unlike the existing 16-frame seq dump (dumpSeqFrame,
// which needs 2-3 separate app launches -- one per display mode -- and an
// offline Python PSNR pass), this computes PSNR ON-DEVICE every captured
// frame by *also* running the Target full-res pass and a Bicubic reference
// upscale alongside the normal Reconstruction chain, without ever touching
// _netInput/_reconstruct's history/reset state -- LUX_START_MODE stays
// Reconstruction the whole time, so the recurrence really is continuous.
static int psnrFramesFromEnvironment() {
    int n = intFromEnvironment(@"LUX_PSNR_FRAMES", 0);
    return n > 0 ? n : 0;
}

// LUX_PSNR_START_FRAME: orbit frame index the capture window begins at ("a
// few frames after launch" per the task spec -- default 4). History/hidden
// state is real and continuous from frame 0 regardless (Reconstruction mode
// runs the full chain every frame from launch) -- this only delays when
// logging starts, e.g. to skip the very first jitter-cycle frames.
static int psnrStartFrameFromEnvironment() {
    int n = intFromEnvironment(@"LUX_PSNR_START_FRAME", 4);
    return n >= 0 ? n : 4;
}

// Nearest/bilinear 2x upscale of the proxy colour texture straight into the
// drawable -- un-premultiplies alpha (the proxy render is premultiplied),
// forces alpha=1 (opaque display). "Bicubic" mode currently uses this same
// kernel's bilinear path as a placeholder (TODO: real bicubic) -- labelled
// "Bilinear" in the HUD/log rather than falsely claiming bicubic.
// `scale` = dst-px-per-src-px (2.0 for the proxy->target 2x upscale;
// 1.0 for the Target-mode identity un-premultiply "copy" -- same kernel,
// no separate code path needed since scale=1 nearest reduces to sx=gid.x).
static const char *kUpscaleMSL = R"(
#include <metal_stdlib>
using namespace metal;
kernel void upscale_proxy(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           constant uint& bilinear [[buffer(0)]],
                           constant float2& scale [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint srcW = src.get_width(), srcH = src.get_height();
    float4 rgba;
    if (bilinear == 0) {
        uint sx = min(uint(float(gid.x) / scale.x), srcW - 1);
        uint sy = min(uint(float(gid.y) / scale.y), srcH - 1);
        rgba = src.read(uint2(sx, sy));
    } else {
        float fx = (float(gid.x) + 0.5) / scale.x - 0.5;
        float fy = (float(gid.y) + 0.5) / scale.y - 0.5;
        int x0 = int(floor(fx)); int y0 = int(floor(fy));
        float tx = fx - float(x0); float ty = fy - float(y0);
        int x0c = clamp(x0, 0, int(srcW) - 1); int x1c = clamp(x0 + 1, 0, int(srcW) - 1);
        int y0c = clamp(y0, 0, int(srcH) - 1); int y1c = clamp(y0 + 1, 0, int(srcH) - 1);
        float4 c00 = src.read(uint2(x0c, y0c));
        float4 c10 = src.read(uint2(x1c, y0c));
        float4 c01 = src.read(uint2(x0c, y1c));
        float4 c11 = src.read(uint2(x1c, y1c));
        rgba = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
    }
    float a = rgba.a;
    float3 rgb = (a > 1e-6) ? rgba.rgb / a : float3(0.0);
    dst.write(float4(rgb, 1.0), gid);
}
)";

// mobiledlss/datagen/camera.py::taa_jitter -- Halton(2,3) TAA jitter in
// TARGET-pixel units, cycling every `period` frames. Callers wanting the
// proxy-pixel jitter (what ProxyRenderer::setJitter expects when
// applied to a proxy-res renderer) must scale by kProxyScale themselves
// (mobiledlss/train/data.py's "jitter_proxy = jitter / S" convention).
static void taaJitterTargetPx(int frame, int period, float &jxOut, float &jyOut) {
    const int i = (frame % period) + 1;
    jxOut = haltonSeq(i, 2) - 0.5f;
    jyOut = haltonSeq(i, 3) - 0.5f;
}

// Reads a texture back and returns it as plain floats, regardless of the
// texture's actual storage precision. lux is about to add an
// `aux_precision: half` compiled variant of getAuxTexture() (RGBA16Float,
// currently RGBA32Float) -- CPU-side dump code that hardcodes 4
// bytes/component (float32) would silently misread that as garbage once
// that lands. The on-GPU consumers (NetInputAssembly.mm's
// `texture2d<float, access::read>` kernel args) already don't have this
// problem -- Metal converts any pixel format to float on read regardless of
// storage -- so only this CPU readback path needs to stay format-agnostic.
// Queries `tex->pixelFormat()` rather than assuming a fixed format.
static std::vector<float> readTextureAsFloats(MetalContext &ctx, MTL::Texture *tex, uint32_t w, uint32_t h,
                                               uint32_t channels) {
    bool isHalf = (tex->pixelFormat() == MTL::PixelFormatRGBA16Float);
    uint32_t bytesPerComponent = isHalf ? 2 : 4;
    auto raw = MetalScreenshot::readTextureRaw(ctx, tex, w, h, channels * bytesPerComponent);
    std::vector<float> out(static_cast<size_t>(w) * h * channels);
    if (isHalf) {
        const uint16_t *half = reinterpret_cast<const uint16_t *>(raw.data());
        for (size_t i = 0; i < out.size(); i++) out[i] = DlssIO::halfToFloat(half[i]);
    } else {
        std::memcpy(out.data(), raw.data(), raw.size());
    }
    return out;
}

// RGB-only PSNR (alpha, index 3 of each pixel's 4 floats, is ignored --
// matches mobiledlss/scripts/eval_rollout.py::_psnr_frame's convention of
// comparing 3-channel RGB tensors: mse = mean((pred-target)**2) over RGB,
// psnr = 10*log10(1/mse)). `pred`/`target` are both [h,w,4] float32,
// un-premultiplied (DlssIO::convertRgba16fColorAttachment's output).
static float psnrRgb(const std::vector<float> &pred, const std::vector<float> &target, uint32_t w, uint32_t h) {
    double se = 0.0;
    size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) {
            double d = static_cast<double>(pred[i * 4 + c]) - static_cast<double>(target[i * 4 + c]);
            se += d * d;
        }
    }
    double mse = se / static_cast<double>(n * 3);
    if (mse <= 0.0) return INFINITY;
    return static_cast<float>(10.0 * std::log10(1.0 / mse));
}

@interface SplatView () {
    MetalContext _ctx;
    MetalSceneManager _scene;        // pruned (kSceneAssetName) -- proxy/reconstruction
    MetalSceneManager _sceneTarget;  // unpruned (kSceneAssetNameFullTarget) -- Target mode only
    // Target mode used to be MetalSplatRenderer (hand-written MSL, CPU sort)
    // rendering straight to the drawable -- this produced visibly wrong
    // splat ordering (no `sort:` source of truth, can silently drift from
    // the Vulkan/gsplat-verified convention). Replaced with a second
    // ProxyRenderer (luxc, GPU radix sort, `sort: view_depth` baked into
    // the compiled examples/gaussian_splat_dlss pipeline -- same shaderBase
    // as _splatRProxy, at 960x540) rendered offscreen and un-premultiply-
    // copied into the drawable, exactly like the Proxy/Bicubic display path.
    std::unique_ptr<ProxyRenderer> _splatRTarget;   // target-res (960x540), offscreen (luxc path)
    std::unique_ptr<ProxyRenderer> _splatRProxy;    // proxy-res, offscreen DLSS attachments (B1, luxc path)
    NetInputAssembly _netInput;                     // B2: 26-ch net input assembly compute pass
    MPSGraphUNet _unet;                             // B3: MPSGraph ParamPredUNet port
    MTL::Buffer *_unetOutputBuffer;                 // fp16 NHWC, netW*netH*312
    MTL::ComputePipelineState *_upscalePipeline;    // proxy -> drawable upscale (Proxy/Bicubic modes)
    ReconstructPass _reconstruct;                   // B4: reconstruct-with-memory
    MTL::Texture *_reconOutputTex;                  // RGBA16Float, target res -- Reconstruction mode's own output
    DisplayMode _displayMode;
    BOOL _reconDumped;

    // Per-stage wall-clock ms (each stage's Metal call already commits+waits
    // internally, so CPU-side deltas here are GPU-inclusive) -- logged every
    // ~0.5s alongside FPS. Requested by the coordinator ahead of the full
    // GPUStartTime/GPUEndTime instrumentation.
    double _msTarget, _msProxy, _msInput, _msNet, _msRecon, _msDisplay;
    double _msMorph;       // setMorphTime/seedPreviousMorphTime GPU round-trip(s), whichever renderer runs this frame
    double _msFrameWall;   // tick-to-tick wall clock (CADisplayLink callback interval)
    CFTimeInterval _lastTickTime;
    DisplayMode _lastDisplayMode;  // for detecting a mode-switch INTO Reconstruction (history reset)
    int _frame;
    int _loopFrames;
    BOOL _ready;
    BOOL _wantScreenshot;
    BOOL _proxyDumped;
    BOOL _proxyDumped2;  // frame kProxyDumpFrame+1 -- flicker/order-stability check (f7fde2c single-command-buffer change)
    BOOL _seqDumped[16];  // 16-frame orbit PSNR test: one-shot per-frame-index dump (Target run vs Reconstruction run, separate app launches)

    // LUX_PSNR_FRAMES: continuous, un-windowed on-device PSNR rollout
    // capture (see psnrFramesFromEnvironment()/-capturePsnrFrame:...).
    int _psnrFrames;        // 0 = disabled
    int _psnrStartFrame;
    MTL::Texture *_psnrBicubicTex;  // scratch target-res RGBA16Float, Bicubic-vs-Target reference
    std::vector<int> _psnrLogFrame;
    std::vector<float> _psnrLogRecon, _psnrLogBicubic, _psnrLogAlpha, _psnrLogWm, _psnrLogHiddenRms;
    BOOL _psnrDone;
}
@property(nonatomic, strong) CADisplayLink *displayLink;
@property(nonatomic, strong) UILabel *hudLabel;
@property(nonatomic, assign) NSInteger fpsFrameCount;
@property(nonatomic, assign) CFTimeInterval fpsWindowStart;
@property(nonatomic, assign) double fps;
@end

@implementation SplatView

+ (Class)layerClass {
    return [CAMetalLayer class];
}

- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        self.backgroundColor = [UIColor blackColor];
        _frame = 0;
        _loopFrames = 300;  // replaced once the animation duration is known
        _ready = NO;
        _displayMode = startModeFromEnvironment();
        // Sentinel so a start mode of Reconstruction still triggers the
        // enter-Reconstruction history reset on frame 0 (harmless -- there's
        // no history yet anyway -- but keeps the logic uniform).
        _lastDisplayMode = static_cast<DisplayMode>(-1);
        _lastTickTime = 0.0;
        NSLog(@"[SplatView] start mode: %s", kDisplayModeNames[_displayMode]);

        _psnrFrames = psnrFramesFromEnvironment();
        _psnrStartFrame = psnrStartFrameFromEnvironment();
        _psnrDone = NO;
        if (_psnrFrames > 0) {
            _psnrLogFrame.reserve(_psnrFrames);
            _psnrLogRecon.reserve(_psnrFrames);
            _psnrLogBicubic.reserve(_psnrFrames);
            _psnrLogAlpha.reserve(_psnrFrames);
            _psnrLogWm.reserve(_psnrFrames);
            _psnrLogHiddenRms.reserve(_psnrFrames);
            NSLog(@"[SplatView] LUX_PSNR_FRAMES=%d starting at frame %d (continuous, no history reset)",
                  _psnrFrames, _psnrStartFrame);
        }

        CAMetalLayer *metalLayer = (CAMetalLayer *)self.layer;
        // All display paths write RGBA16Float into the drawable: the
        // upscale_proxy compute kernel (Proxy/Bicubic/Target) and the
        // straight blit from _reconOutputTex (Reconstruction, itself
        // RGBA16Float -- blitCommandEncoder requires matching formats).
        // BGRA8Unorm (the CAMetalLayer default) would silently reinterpret
        // half-float bytes as 8-bit UNORM and come out as garbage --
        // discovered via an on-device screenshot before this fix (back when
        // Target rendered straight to the drawable via the now-removed
        // hand-written MetalSplatRenderer::renderToDrawable path).
        metalLayer.pixelFormat = MTLPixelFormatRGBA16Float;
        // NO (not YES): devicectl has no `screenshot` subcommand for this
        // CoreDevice/iOS 17+ tunnel, so this app saves its own PNG to
        // Documents on tap (see -handleTap:) via a blit readback of the
        // drawable's own texture -- that requires framebufferOnly = NO.
        metalLayer.framebufferOnly = NO;
        metalLayer.drawableSize = CGSizeMake(kTargetW, kTargetH);

        UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self
                                                                                action:@selector(handleTap:)];
        [self addGestureRecognizer:tap];

        self.hudLabel = [[UILabel alloc] initWithFrame:CGRectMake(12, 12, 700, 80)];
        self.hudLabel.textColor = [UIColor colorWithRed:0.3 green:1.0 blue:0.4 alpha:1.0];
        self.hudLabel.backgroundColor = [UIColor colorWithWhite:0 alpha:0.35];
        self.hudLabel.font = [UIFont monospacedSystemFontOfSize:14 weight:UIFontWeightRegular];
        self.hudLabel.numberOfLines = 0;
        self.hudLabel.text = @"loading scene...";
        [self addSubview:self.hudLabel];

        [self setupMetalAsync];
    }
    return self;
}

- (void)setupMetalAsync {
    __weak SplatView *weakSelf = self;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        SplatView *strongSelf = weakSelf;
        if (!strongSelf) return;
        std::string error;
        BOOL ok = [strongSelf loadSceneAndInitRenderer:&error];
        dispatch_async(dispatch_get_main_queue(), ^{
            SplatView *self2 = weakSelf;
            if (!self2) return;
            if (ok) {
                self2->_ready = YES;
                [self2 startDisplayLink];
            } else {
                self2.hudLabel.text = [NSString stringWithFormat:@"Metal init failed:\n%s", error.c_str()];
            }
        });
    });
}

- (BOOL)loadSceneAndInitRenderer:(std::string *)errorOut {
    try {
        // Metal device/queue creation only (no UIKit/CALayer touches here --
        // this runs on a background queue; the CAMetalLayer wiring happens
        // on the main thread in startDisplayLink instead).
        _ctx.initHeadless();

        // MetalSplatLuxcRenderer::init() (and its ShaderTranspiler) reads its
        // compiled pipeline via HARDCODED paths relative to the process's
        // current working directory ("examples/gaussian_splat_dlss.*.spv",
        // "shaders/radix_sort/*.comp.spv" -- see metal_splat_luxc_renderer.cpp),
        // exactly like the macOS CLI run from the repo root. iOS has no
        // meaningful default CWD, so chdir into the bundle's resource dir
        // (where the `examples`/`shaders` folder references below land,
        // preserving that same relative layout) before touching it.
        NSString *resourcePath = [[NSBundle mainBundle] resourcePath];
        chdir(resourcePath.UTF8String);

        NSString *path = [[NSBundle mainBundle] pathForResource:kSceneAssetName ofType:@"glb"];
        if (!path) {
            *errorOut = "bundle resource not found: " + std::string(kSceneAssetName.UTF8String) + ".glb";
            return NO;
        }
        NSLog(@"[SplatView] loading scene (pruned, proxy/reconstruction): %@", path);
        _scene.loadScene(std::string(path.UTF8String));

        if (!_scene.hasSplatData()) {
            *errorOut = "scene has no KHR_gaussian_splatting data";
            return NO;
        }

        // Target mode's own unpruned scene (see kSceneAssetNameFullTarget's
        // comment / mobiledlss/reports/ipad_recon_gap.md) -- a SEPARATE
        // MetalSceneManager since it's a different glb/splat count entirely,
        // not just a different resolution of the same scene.
        NSString *targetPath = [[NSBundle mainBundle] pathForResource:kSceneAssetNameFullTarget ofType:@"glb"];
        if (!targetPath) {
            *errorOut = "bundle resource not found: " + std::string(kSceneAssetNameFullTarget.UTF8String) + ".glb";
            return NO;
        }
        NSLog(@"[SplatView] loading scene (unpruned, Target mode only): %@", targetPath);
        _sceneTarget.loadScene(std::string(targetPath.UTF8String));

        if (!_sceneTarget.hasSplatData()) {
            *errorOut = "target scene has no KHR_gaussian_splatting data";
            return NO;
        }

        // Target mode: same luxc pipeline/shaderBase as the proxy pass, just
        // pointed at the unpruned scene, at full target resolution, no
        // jitter -- gets the same GPU radix sort (`sort: view_depth`, baked
        // into examples/gaussian_splat_dlss's compiled reflection) instead
        // of the old hand-written CPU-sorted path's wrong ordering.
        _splatRTarget = std::make_unique<ProxyRenderer>();
        _splatRTarget->init(_ctx, _sceneTarget.getSplatData(), "examples/gaussian_splat_dlss", kTargetW, kTargetH);
        // Perf (bench/lux_perf_ablation.md in mobiledlss, Mac M4 Max Metal
        // parity session): this continuous per-frame rendering loop was
        // paying the full GPU radix sort every single frame -- measured as
        // the entire remaining gap vs. MetalSplatter on the desktop Metal
        // CLI (~1.87ms/frame, flat, regardless of resolution/scene).
        // setSortSchedule(4, 2.0f) (re-sort every 4th frame OR >=2deg
        // camera-view rotation since the last real sort, whichever first)
        // amortizes this to near-zero on typical orbit/viewing motion with
        // ZERO measured pixel difference at realistic rotation rates (see
        // the ablation doc's bit-exact PNG-diff verification) -- default
        // behavior (unconditional every-frame sort) is unchanged for every
        // other caller (headless CLI, tests).
        _splatRTarget->setSortSchedule(4, 2.0f);

        // B1: a second renderer instance dedicated to the proxy-res (480x270)
        // DLSS-attachment pass -- ProxyRenderer::init() fixes its offscreen
        // colorTarget_/motionTarget_/expectedDepthTarget_ at the given
        // width/height, so this can't share _splatRTarget's instance. luxc
        // needs the compiled pipeline base (examples/gaussian_splat_dlss.*.spv,
        // bundled as a folder reference) -- the same pipeline
        // gaussian_splat_dlss.lux compiles to for the Vulkan/hand-Metal
        // parity comparison in commit 08cfe8e.
        _splatRProxy = std::make_unique<ProxyRenderer>();
        _splatRProxy->init(_ctx, _scene.getSplatData(), "examples/gaussian_splat_dlss", kProxyW, kProxyH);
        // Perf: same sort-scheduling amortization as _splatRTarget above --
        // see that call site's comment.
        _splatRProxy->setSortSchedule(4, 2.0f);

        // B2: net input assembly (bg-sphere UV + texture sample + disocclusion
        // + box-pool). texture.npy/bg_sphere.npy exported by
        // demo/ios_assets/export_ios_weights.py (mobiledlss repo).
        NSString *texPath = [[NSBundle mainBundle] pathForResource:@"texture" ofType:@"npy"];
        NSString *spherePath = [[NSBundle mainBundle] pathForResource:@"bg_sphere" ofType:@"npy"];
        if (!texPath || !spherePath) {
            *errorOut = "bundle resource not found: texture.npy / bg_sphere.npy";
            return NO;
        }
        // lux 6ed0334 merged the DLSS aux attachments (out_motion/out_depth ->
        // one packed out_aux, foreground_coverage -> its own out_fg) -- no more
        // depthAlphaOffset param, see NetInputAssembly.h/.mm.
        _netInput.init(_ctx, std::string(texPath.UTF8String), std::string(spherePath.UTF8String),
                        kProxyW, kProxyH, /*paramStride=*/2, /*hiddenChannels=*/8);
        // lux's foreground_coverage output has landed (examples/gaussian_splat_dlss.lux's
        // `foreground_coverage: true`, compiled fresh from a clean HEAD worktree into
        // playground_ios/CompiledShaders/ -- see the Xcode project's examples/shaders
        // folder references) -- switch off the const-0 interim now that a real per-pixel
        // actor mask is available (NetInputAssembly.mm un-premultiplies it by the fg
        // attachment's own alpha, matching metal_main.cpp's --output-aux _fg.npy
        // convention -- lux 6ed0334 moved fg to its own out_fg attachment).
        if (_splatRProxy->hasForegroundCoverage()) {
            _netInput.setFgSource(NetInputAssembly::kFgSourceFgTexture);
            NSLog(@"[SplatView] fg source: separate fg attachment (real foreground_coverage)");
        } else {
            NSLog(@"[SplatView] WARNING: compiled pipeline has no foreground_coverage -- fg stays const 0");
        }

        // B3: MPSGraph ParamPredUNet port, running at NetInputAssembly's own
        // (already-padded, multiple-of-8) net resolution.
        NSString *unetBinPath = [[NSBundle mainBundle] pathForResource:@"unet_weights.fp16" ofType:@"bin"];
        NSString *unetLayersPath = [[NSBundle mainBundle] pathForResource:@"unet_weights.layers" ofType:@"txt"];
        if (!unetBinPath || !unetLayersPath) {
            *errorOut = "bundle resource not found: unet_weights.fp16.bin / unet_weights.layers.txt";
            return NO;
        }
        _unet.init(_ctx, std::string(unetBinPath.UTF8String), std::string(unetLayersPath.UTF8String),
                   _netInput.getNetW(), _netInput.getNetH());
        _unetOutputBuffer = _ctx.newBuffer(
            static_cast<size_t>(_netInput.getNetW()) * _netInput.getNetH() * _unet.getOutChannels() * sizeof(uint16_t),
            MTL::ResourceStorageModeShared);
        NSLog(@"[SplatView] MPSGraphUNet ready: in=%u out=%u net=%ux%u K=%u hidden=%u",
              _unet.getInChannels(), _unet.getOutChannels(), _unet.getNetW(), _unet.getNetH(),
              _unet.getK(), _unet.getHiddenChannels());

        {
            NS::Error *error = nullptr;
            auto *src = NS::String::string(kUpscaleMSL, NS::UTF8StringEncoding);
            auto *opts = MTL::CompileOptions::alloc()->init();
            auto *lib = _ctx.device->newLibrary(src, opts, &error);
            opts->release();
            if (!lib) {
                *errorOut = std::string("upscale shader compile failed: ") +
                            (error ? error->localizedDescription()->utf8String() : "?");
                return NO;
            }
            auto *fn = lib->newFunction(NS::String::string("upscale_proxy", NS::UTF8StringEncoding));
            _upscalePipeline = _ctx.device->newComputePipelineState(fn, &error);
            fn->release();
            lib->release();
            if (!_upscalePipeline) {
                *errorOut = "failed to create upscale_proxy pipeline";
                return NO;
            }
        }

        // B4: reconstruct-with-memory. nBlend isn't stored directly in the
        // manifest -- derive it from out_channels = sp*sp*K*K + nBlend*sp*sp + hidden.
        {
            uint32_t sp = _unet.getUpscale() * _unet.getParamStride();
            uint32_t kk = _unet.getK() * _unet.getK();
            uint32_t nBlend = (_unet.getOutChannels() - sp * sp * kk - _unet.getHiddenChannels()) / (sp * sp);
            NSString *memHeadPath = [[NSBundle mainBundle] pathForResource:@"memory_head" ofType:@"npz"];
            if (!memHeadPath) {
                *errorOut = "bundle resource not found: memory_head.npz";
                return NO;
            }
            _reconstruct.init(_ctx, std::string(texPath.UTF8String), std::string(spherePath.UTF8String),
                               std::string(memHeadPath.UTF8String), kTargetW, kTargetH, kProxyW, kProxyH,
                               _netInput.getNetW(), _netInput.getNetH(), _unet.getK(), _unet.getHiddenChannels(),
                               nBlend, kTargetW / kProxyW, _unet.getParamStride());
            NSLog(@"[SplatView] ReconstructPass ready: nBlend=%u sp=%u", nBlend, sp);

            auto *reconDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, kTargetW, kTargetH, false);
            reconDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
            reconDesc->setStorageMode(MTL::StorageModePrivate);
            _reconOutputTex = _ctx.newTexture(reconDesc);

            if (_psnrFrames > 0) {
                // LUX_PSNR_FRAMES: scratch target-res texture the Bicubic
                // reference (upscale_proxy, bilinear=1) writes into every
                // captured frame -- same kernel/convention as the Bicubic
                // display mode, just offscreen so it doesn't fight
                // Reconstruction for the drawable.
                auto *bicubicDesc =
                    MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, kTargetW, kTargetH, false);
                bicubicDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
                bicubicDesc->setStorageMode(MTL::StorageModePrivate);
                _psnrBicubicTex = _ctx.newTexture(bicubicDesc);
            }
        }

        if (_splatRTarget->hasMotion()) {
            float dur = _splatRTarget->animationDuration();
            _loopFrames = std::max(1, static_cast<int>(std::round(dur * kSimFps)));
        }
        NSLog(@"[SplatView] ready: %u splats (pruned, proxy/reconstruction), %u splats (unpruned, Target), loopFrames=%d",
              _scene.getSplatData().num_splats, _sceneTarget.getSplatData().num_splats, _loopFrames);
        return YES;
    } catch (const std::exception &e) {
        *errorOut = e.what();
        return NO;
    }
}

- (void)startDisplayLink {
    // Main-thread CAMetalLayer wiring (device + the metal-cpp bridge pointer
    // MetalContext::metalLayer's nextDrawable() call needs), now that
    // _ctx.device exists (created off-thread in loadSceneAndInitRenderer).
    CAMetalLayer *metalLayer = (CAMetalLayer *)self.layer;
    metalLayer.device = (__bridge id<MTLDevice>)_ctx.device;
    _ctx.metalLayer = (__bridge CA::MetalLayer *)metalLayer;

    self.displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(tick:)];
    // Request the panel's full ProMotion range so the compositor doesn't
    // silently clamp us to the legacy 60Hz default -- preferredFramesPerSecond
    // alone isn't honored reliably on ProMotion displays. iPad has no
    // "minimum frame duration" opt-in (that's iPhone-only, via the
    // CADisableMinimumFrameDurationOnPhone Info.plist key); the range API
    // is sufficient here.
    if (@available(iOS 15.0, *)) {
        self.displayLink.preferredFrameRateRange = CAFrameRateRangeMake(30, 120, 120);
    } else {
        self.displayLink.preferredFramesPerSecond = 30;
    }
    [self.displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
    self.fpsWindowStart = CACurrentMediaTime();
    self.fpsFrameCount = 0;
}

- (void)tick:(CADisplayLink *)link {
    if (!_ready || !_splatRTarget) return;

    CFTimeInterval tickT0 = CACurrentMediaTime();
    _msFrameWall = (_lastTickTime > 0.0) ? (tickT0 - _lastTickTime) * 1000.0 : 0.0;
    _lastTickTime = tickT0;

    int prevFrame = (_frame > 0) ? (_frame - 1) : 0;
    float tCur = _splatRTarget->hasMotion() ? _splatRTarget->frameToTime(_frame) : 0.0f;
    float tPrev = _splatRTarget->hasMotion() ? _splatRTarget->frameToTime(prevFrame) : 0.0f;

    // Per coordinator: each mode should only pay for the stages it actually
    // displays -- Target = full-res luxc pass only; Proxy/Bicubic = proxy
    // pass only (no net input/net/reconstruct); Reconstruction = the full
    // proxy+input+net+recon chain. Re-entering Reconstruction after frames
    // were skipped resets the recurrent history (depth/hidden/prevColor) --
    // see NetInputAssembly::reset()/ReconstructPass::reset() -- since it
    // would otherwise be stale by however many frames/orbit-degrees were
    // skipped, not just one frame old.
    BOOL enteringRecon = (_displayMode == DisplayModeReconstruction && _lastDisplayMode != DisplayModeReconstruction);
    if (enteringRecon) {
        _netInput.reset();
        _reconstruct.reset();
        NSLog(@"[SplatView] entering Reconstruction mode -- history reset");
    }
    _lastDisplayMode = _displayMode;

    _msTarget = 0.0;
    _msProxy = 0.0;
    _msInput = 0.0;
    _msNet = 0.0;
    _msRecon = 0.0;
    _msMorph = 0.0;

    // mobiledlss/train/data.py convention: jitter_proxy = jitter_target / S.
    // Computed unconditionally (cheap, CPU-only) since Proxy/Bicubic/
    // Reconstruction all use it for their (still-jittered, TAA-style) proxy
    // render.
    float jxTarget, jyTarget;
    taaJitterTargetPx(_frame, /*period=*/16, jxTarget, jyTarget);
    float jxProxy = jxTarget * kProxyScale;
    float jyProxy = jyTarget * kProxyScale;

    if (_displayMode == DisplayModeTarget) {
        // --- Target mode ONLY: full-res (960x540) luxc splat render,
        // offscreen (no renderToDrawable on this backend -- headless-only).
        // No previous-camera/seedPreviousMorphTime call: this mode's own
        // motion-vector output is never consumed by anything (display is a
        // straight un-premultiply copy in the section below), and
        // seedPreviousMorphTime() is expensive (2 extra blocking
        // setMorphTime GPU round-trips inside MetalSplatLuxcRenderer) --
        // skipping it here removes 2 of what would otherwise be 3
        // synchronous morph evaluations per frame.
        glm::vec3 eyeCur, rCur, uCur, fCur;
        glm::mat4 viewCur, projCur;
        float fxCur, fyCur;
        computeOrbitCamera(_frame, kTargetW, kTargetH, ProxyRenderer::kMetalYConvention,
                            eyeCur, rCur, uCur, fCur, viewCur, projCur, fxCur, fyCur);
        _splatRTarget->updateCameraExplicit(eyeCur, viewCur, projCur, fxCur, fyCur);
        _splatRTarget->setJitter(0.0f, 0.0f);  // ground-truth full-res reference: no TAA jitter.

        CFTimeInterval m0 = CACurrentMediaTime();
        if (_splatRTarget->hasMotion()) {
            _splatRTarget->setMorphTime(tCur);
        }
        CFTimeInterval m1 = CACurrentMediaTime();
        _msMorph = (m1 - m0) * 1000.0;

        @autoreleasepool {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            _splatRTarget->render(_ctx);
            pool->release();
        }
        _msTarget = (CACurrentMediaTime() - m1) * 1000.0;

        if (_frame < 16 && !_seqDumped[_frame]) {
            _seqDumped[_frame] = YES;
            [self dumpSeqFrame:_splatRTarget->getOutputTexture() tag:@"target" frame:_frame];
        }

    } else if (_displayMode == DisplayModeProxy || _displayMode == DisplayModeBicubic) {
        // --- Proxy/Bicubic: proxy-res render + upscale display only. No
        // net input/net/reconstruct, and (like Target above) no
        // seedPreviousMorphTime -- MV isn't consumed by a plain colour
        // upscale.
        glm::vec3 peyeCur, prCur, puCur, pfCur;
        glm::mat4 pviewCur, pprojCur;
        float pfxCur, pfyCur;
        computeOrbitCamera(_frame, kProxyW, kProxyH, ProxyRenderer::kMetalYConvention,
                            peyeCur, prCur, puCur, pfCur, pviewCur, pprojCur, pfxCur, pfyCur);
        _splatRProxy->updateCameraExplicit(peyeCur, pviewCur, pprojCur, pfxCur, pfyCur);
        _splatRProxy->setJitter(jxProxy, jyProxy);

        CFTimeInterval m0 = CACurrentMediaTime();
        if (_splatRProxy->hasMotion()) {
            _splatRProxy->setMorphTime(tCur);
        }
        CFTimeInterval m1 = CACurrentMediaTime();
        _msMorph = (m1 - m0) * 1000.0;

        @autoreleasepool {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            _splatRProxy->render(_ctx);
            pool->release();
        }
        _msProxy = (CACurrentMediaTime() - m1) * 1000.0;

    } else {
        // --- Reconstruction: proxy render (with real previous-camera/morph
        // for correct MV) + net input assembly (B2) + net (B3) + reconstruct
        // (B4). This is the one mode that still pays for the full 3x
        // blocking morph evaluation (setMorphTime(tCur) +
        // seedPreviousMorphTime(tPrev)'s own internal 2 calls) inside
        // MetalSplatLuxcRenderer, since real per-frame MV is load-bearing
        // here (disocclusion + warp) -- reducing that further needs a change
        // inside metal_splat_luxc_renderer.cpp itself (shared playground_cpp
        // file, out of this pass's scope).
        glm::vec3 peyeCur, peyePrev, prCur, puCur, pfCur, prPrev, puPrev, pfPrev;
        glm::mat4 pviewCur, pviewPrev, pprojCur, pprojPrev;
        float pfxCur, pfyCur, pfxPrev, pfyPrev;
        computeOrbitCamera(_frame, kProxyW, kProxyH, ProxyRenderer::kMetalYConvention,
                            peyeCur, prCur, puCur, pfCur, pviewCur, pprojCur, pfxCur, pfyCur);
        computeOrbitCamera(prevFrame, kProxyW, kProxyH, ProxyRenderer::kMetalYConvention,
                            peyePrev, prPrev, puPrev, pfPrev, pviewPrev, pprojPrev, pfxPrev, pfyPrev);

        _splatRProxy->updateCameraExplicit(peyeCur, pviewCur, pprojCur, pfxCur, pfyCur);
        _splatRProxy->setPreviousCameraExplicit(pviewPrev, pprojPrev);
        _splatRProxy->setJitter(jxProxy, jyProxy);

        CFTimeInterval m0 = CACurrentMediaTime();
        if (_splatRProxy->hasMotion()) {
            _splatRProxy->setMorphTime(tCur);
            _splatRProxy->seedPreviousMorphTime(tPrev);
        }
        CFTimeInterval m1 = CACurrentMediaTime();
        _msMorph = (m1 - m0) * 1000.0;

        @autoreleasepool {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            _splatRProxy->render(_ctx);
            pool->release();
        }
        CFTimeInterval tp1 = CACurrentMediaTime();
        _msProxy = (tp1 - m1) * 1000.0;

        // B4 step 0 (must run BEFORE B2/B3 this frame): warp+downsample last
        // frame's raw hidden state into this frame's net-res hidden_in.
        // lux 6ed0334: getMotionTexture()/getExpectedDepthTexture() are gone,
        // replaced by one packed getAuxTexture() (mv.x*a, mv.y*a, depth*a, a)
        // -- both prepareHiddenInput (mv only) and netInput.run (mv + depth,
        // via the separate unpremul pass inside NetInputAssembly::run) now
        // read from it.
        BOOL isFirstReconFrame = _netInput.isNextFrameFirst();
        _reconstruct.prepareHiddenInput(_ctx, _splatRProxy->getAuxTexture(), isFirstReconFrame,
                                         _reconstruct.getHiddenInputBuffer());

        // B2: assemble the 26-ch net input from this frame's proxy DLSS
        // attachments (hidden_in from B4's recurrence above). fg comes from
        // its own getFgTexture() (nullptr when !hasForegroundCoverage() --
        // NetInputAssembly::run() falls back to a dummy binding for that case).
        _netInput.run(_ctx, _splatRProxy->getOutputTexture(), _splatRProxy->getAuxTexture(),
                      _splatRProxy->hasForegroundCoverage() ? _splatRProxy->getFgTexture() : nullptr,
                      _reconstruct.getHiddenInputBuffer(),
                      peyeCur, prCur, puCur, pfCur, pfxCur, pfyCur,
                      0.5f * kProxyW, 0.5f * kProxyH, jxProxy, jyProxy);
        CFTimeInterval tp2 = CACurrentMediaTime();
        _msInput = (tp2 - tp1) * 1000.0;

        // B3: run the network on this frame's assembled input, own command
        // buffer (single-command-buffer fusion is a later optimization pass).
        {
            auto *unetCmdBuf = _ctx.beginCommandBuffer();
            _unet.encode(_ctx, unetCmdBuf, _netInput.getOutputBuffer(), _unetOutputBuffer);
        }
        CFTimeInterval tp3 = CACurrentMediaTime();
        _msNet = (tp3 - tp2) * 1000.0;

        // B4: reconstruct-with-memory.
        glm::vec3 teyeCur, trCur, tuCur, tfCur;
        glm::mat4 tviewCur, tprojCur;
        float tfxCur, tfyCur;
        computeOrbitCamera(_frame, kTargetW, kTargetH, ProxyRenderer::kMetalYConvention,
                            teyeCur, trCur, tuCur, tfCur, tviewCur, tprojCur, tfxCur, tfyCur);
        MTL::Texture *curDepth = _netInput.getDepthWrittenThisFrame();
        MTL::Texture *prevDepth = _netInput.getDepthFromPreviousFrame();
        BOOL wasFirst = _netInput.wasFirstFrame();

        if (_frame == kProxyDumpFrame && !_proxyDumped) {
            // Everything reconstruct.run() is about to consume this frame,
            // captured BEFORE the call (prevColor_/prevHidden_ read side,
            // before pingIndex_ flips) -- for reproducing this exact step in
            // PyTorch. proxy_f10_{color,mv,depth}.npy (dumpProxyFrame),
            // netinput_f10.npy's trailing hiddenChannels columns (this
            // frame's hidden_in), and unet_output_f10.npy (packed
            // kernel/blend/hidden_raw) are the other half of "everything
            // reconstruct consumes" -- already dumped by the calls below.
            [self dumpReconInputsWithPrevDepth:prevDepth jitterTargetX:jxTarget jitterTargetY:jyTarget];
        }

        _reconstruct.run(_ctx, _splatRProxy->getOutputTexture(), _splatRProxy->getAuxTexture(), curDepth,
                          prevDepth, wasFirst, _unetOutputBuffer, _reconOutputTex, teyeCur, trCur, tuCur, tfCur,
                          tfxCur, tfyCur, 0.5f * kTargetW, 0.5f * kTargetH, jxTarget, jyTarget);
        CFTimeInterval tp4 = CACurrentMediaTime();
        _msRecon = (tp4 - tp3) * 1000.0;

        // LUX_PSNR_FRAMES: continuous rollout PSNR capture -- runs entirely
        // inside the normal Reconstruction branch (no display-mode switch,
        // no _netInput.reset()/_reconstruct.reset()), so the recurrent
        // history stays exactly as continuous as it is in every other mode.
        // The extra Target full-res render + Bicubic upscale below are the
        // ONLY things this feature adds to the frame; both are read back to
        // CPU purely for the PSNR/RMS math, never fed back into the chain.
        if (_psnrFrames > 0 && _frame >= _psnrStartFrame && _frame < _psnrStartFrame + _psnrFrames) {
            [self capturePsnrFrame:_frame eyeCur:teyeCur rCur:trCur uCur:tuCur fCur:tfCur viewCur:tviewCur
                            projCur:tprojCur fxCur:tfxCur fyCur:tfyCur tCur:tCur];
        }

        if (_frame < 16 && !_seqDumped[_frame]) {
            _seqDumped[_frame] = YES;
            [self dumpSeqFrame:_reconOutputTex tag:@"reconseq" frame:_frame];
        }

        if (_frame == kProxyDumpFrame && !_proxyDumped) {
            _proxyDumped = YES;
            NSLog(@"[SplatView] B1 dump: frame=%d prevFrame=%d jitterTargetPx=(%.6f,%.6f) "
                  @"jitterProxyPx=(%.6f,%.6f) tCur=%.6f tPrev=%.6f eyeCur=(%.6f,%.6f,%.6f)",
                  _frame, prevFrame, jxTarget, jyTarget, jxProxy, jyProxy, tCur, tPrev,
                  peyeCur.x, peyeCur.y, peyeCur.z);
            [self dumpProxyFrame];
            [self dumpNetInput];
            [self dumpUnetOutput];
            // ReconstructPass has already run every frame from 0 (real history
            // in prevColor_/prevHidden_ by now), so frame 10's own reconstruction
            // is directly comparable to a from-scratch PyTorch replay of frames 0..10.
            _reconDumped = YES;
            [self dumpReconOutput];
        } else if (_frame == kProxyDumpFrame + 1 && !_proxyDumped2) {
            // Consecutive-frame proxy dump (flicker/order-stability check for
            // the single-command-buffer change, f7fde2c): compare against
            // proxy_f10's own color/depth -- with only 0.6deg of camera motion
            // between them, per-pixel splat ordering (and hence composited
            // color) should be nearly identical; a flip in visible ordering
            // would show up as a localized color/depth discontinuity here that
            // isn't explained by the small camera delta.
            _proxyDumped2 = YES;
            [self dumpProxyFrameWithTag:@"proxy_f11"];
        }
    }

    // --- Display (all modes): cheap compute-kernel copy/upscale (or, for
    // Reconstruction, a straight blit) into the drawable. Target now goes
    // through this same path as Proxy/Bicubic (scale=1, i.e. an
    // un-premultiply "copy") since MetalSplatLuxcRenderer has no
    // renderToDrawable of its own (headless-only backend).
    {
        CFTimeInterval td0 = CACurrentMediaTime();
        @autoreleasepool {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            CA::MetalDrawable *drawable = _ctx.metalLayer->nextDrawable();
            if (drawable) {
                auto *cmdBuf = _ctx.beginCommandBuffer();
                if (_displayMode == DisplayModeReconstruction) {
                    // _reconOutputTex is already target-res RGBA16Float --
                    // straight blit into the (same-format) drawable.
                    auto *blit = cmdBuf->blitCommandEncoder();
                    blit->copyFromTexture(_reconOutputTex, 0, 0, MTL::Origin(0, 0, 0),
                                           MTL::Size(kTargetW, kTargetH, 1), drawable->texture(), 0, 0,
                                           MTL::Origin(0, 0, 0));
                    blit->endEncoding();
                } else {
                    MTL::Texture *srcTex = (_displayMode == DisplayModeTarget)
                                                ? _splatRTarget->getOutputTexture()
                                                : _splatRProxy->getOutputTexture();
                    float scaleXY = (_displayMode == DisplayModeTarget)
                                        ? 1.0f
                                        : static_cast<float>(kTargetW) / static_cast<float>(kProxyW);
                    std::array<float, 2> scale = {scaleXY, scaleXY};
                    uint32_t bilinear = (_displayMode == DisplayModeBicubic) ? 1 : 0;
                    auto *enc = cmdBuf->computeCommandEncoder();
                    enc->setComputePipelineState(_upscalePipeline);
                    enc->setTexture(srcTex, 0);
                    enc->setTexture(drawable->texture(), 1);
                    enc->setBytes(&bilinear, sizeof(bilinear), 0);
                    enc->setBytes(scale.data(), sizeof(float) * 2, 1);
                    MTL::Size grid(kTargetW, kTargetH, 1);
                    NS::UInteger tew = _upscalePipeline->threadExecutionWidth();
                    NS::UInteger maxT = _upscalePipeline->maxTotalThreadsPerThreadgroup();
                    NS::UInteger th = maxT / tew;
                    if (th == 0) th = 1;
                    MTL::Size tg(tew, th, 1);
                    enc->dispatchThreads(grid, tg);
                    enc->endEncoding();
                }
                cmdBuf->presentDrawable(drawable);
                cmdBuf->commit();
                cmdBuf->waitUntilCompleted();
                if (_wantScreenshot) {
                    _wantScreenshot = NO;
                    [self saveScreenshotFromTexture:drawable->texture()];
                }
                // Bicubic-vs-Target PSNR baseline (task spec): the drawable
                // already holds the final displayed (un-premultiplied,
                // alpha=1) image at this point -- dumpSeqFrame's own
                // rgb/alpha divide is a no-op on alpha=1, safe to reuse.
                if (_displayMode == DisplayModeBicubic && _frame < 16 && !_seqDumped[_frame]) {
                    _seqDumped[_frame] = YES;
                    [self dumpSeqFrame:drawable->texture() tag:@"bicubic" frame:_frame];
                }
            }
            pool->release();
        }
        _msDisplay = (CACurrentMediaTime() - td0) * 1000.0;
    }

    _frame = (_frame + 1) % _loopFrames;

    // Auto-dump a screenshot every ~3s too (no UI-automation tap injection
    // available over devicectl -- see -handleTap: for the interactive path).
    // Suppressed during a LUX_PSNR_FRAMES capture window: it's an extra
    // synchronous GPU readback right in the middle of the very rollout being
    // measured -- one of the task's own bisection hypotheses for what an
    // app-side decay bug could be, so keep it out of the capture entirely
    // rather than risk it being a confound.
    BOOL psnrWindowActive = _psnrFrames > 0 && _frame >= _psnrStartFrame && _frame < _psnrStartFrame + _psnrFrames;
    if (!psnrWindowActive && _frame % (int)(kSimFps * 3.0f) == 0) {
        _wantScreenshot = YES;
    }

    self.fpsFrameCount += 1;
    CFTimeInterval now = CACurrentMediaTime();
    CFTimeInterval elapsed = now - self.fpsWindowStart;
    // "other": whole-frame wall time (tick-to-tick, i.e. the real
    // CADisplayLink period) minus everything this method itself timed --
    // display-link scheduling slack, UIKit/dispatch overhead, and any
    // remaining un-timed CPU work in this method.
    double summedStages = _msMorph + _msTarget + _msProxy + _msInput + _msNet + _msRecon + _msDisplay;
    double msOther = _msFrameWall - summedStages;
    if (elapsed >= 0.5) {
        self.fps = self.fpsFrameCount / elapsed;
        self.fpsFrameCount = 0;
        self.fpsWindowStart = now;
        self.hudLabel.text = [NSString stringWithFormat:
            @"%s | %u splats | frame %d/%d | %.1f fps (frame=%.1fms)\n"
            @"morph=%.1fms target=%.1fms proxy=%.1fms input=%.1fms net=%.1fms recon=%.1fms disp=%.1fms other=%.1fms",
            kDisplayModeNames[_displayMode], _scene.getSplatData().num_splats, _frame, _loopFrames, self.fps,
            _msFrameWall, _msMorph, _msTarget, _msProxy, _msInput, _msNet, _msRecon, _msDisplay, msOther];
        NSLog(@"[SplatView] mode=%s fps=%.1f frame=%d/%d | wall=%.2fms morph=%.2fms target=%.2fms proxy=%.2fms "
              @"input=%.2fms net=%.2fms recon=%.2fms display=%.2fms other=%.2fms summed=%.2fms",
              kDisplayModeNames[_displayMode], self.fps, _frame, _loopFrames,
              _msFrameWall, _msMorph, _msTarget, _msProxy, _msInput, _msNet, _msRecon, _msDisplay, msOther,
              summedStages);
    }
}

- (void)handleTap:(UITapGestureRecognizer *)tap {
    _displayMode = static_cast<DisplayMode>((_displayMode + 1) % DisplayModeCount);
    NSLog(@"[SplatView] display mode -> %s", kDisplayModeNames[_displayMode]);
    _wantScreenshot = YES;
}

// B1 validation dump: replicates metal_main.cpp's `--output-aux` aux-dump
// logic exactly (same DlssIO/MetalScreenshot calls, same un-premultiply
// convention) against _splatRProxy's own colour/motion/expected-depth
// targets, writing <Documents>/proxy_f10_{color,depth,mv}.npy +
// _color.png so it can be pulled with `devicectl device copy from` and
// compared against `lux-playground-metal --output-aux` run on the Mac with
// the identical camera/jitter (logged to NSLog just before this call).
- (void)dumpProxyFrame {
    [self dumpProxyFrameWithTag:@"proxy_f10"];
}

// tag is the Documents filename prefix (e.g. "proxy_f10", "proxy_f11" for
// the flicker-check consecutive-frame dump -- see the kProxyDumpFrame+1
// call site).
- (void)dumpProxyFrameWithTag:(NSString *)tag {
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *prefix = [docPaths.firstObject stringByAppendingPathComponent:tag];
    std::string p = std::string(prefix.UTF8String);
    uint32_t w = kProxyW, h = kProxyH;
    try {
        MetalScreenshot::saveTextureToPNG(_ctx, _splatRProxy->getOutputTexture(), w, h, p + "_color.png");

        {
            auto raw = MetalScreenshot::readTextureRaw(_ctx, _splatRProxy->getOutputTexture(), w, h, 8);
            std::vector<uint8_t> unusedRgba8;
            auto colorF32 = DlssIO::convertRgba16fColorAttachment(raw, w, h, unusedRgba8);
            DlssIO::writeNpyFloat32(p + "_color.npy", colorF32, {h, w, 4});
        }

        // lux 6ed0334: depth AND mv now share one packed getAuxTexture()
        // (mv.x*a, mv.y*a, depth*a, a) -- one read produces both dumps, each
        // channel un-premultiplied by THIS texture's own alpha (.w, index 3),
        // exactly metal_main.cpp's --output-aux convention (docs/
        // rendering-engines.md). readTextureAsFloats() is precision-agnostic
        // (RGBA32Float today; an `aux_precision: half` RGBA16Float variant is
        // expected soon -- see its own comment).
        if (_splatRProxy->hasExpectedDepth() || _splatRProxy->hasMotionVectors()) {
            constexpr uint32_t C = ProxyRenderer::kAuxChannels;  // 4
            std::vector<float> chans = readTextureAsFloats(_ctx, _splatRProxy->getAuxTexture(), w, h, C);
            std::vector<float> auxAlpha(static_cast<size_t>(w) * h);
            for (size_t i = 0; i < auxAlpha.size(); ++i) auxAlpha[i] = chans[i * C + 3];

            if (_splatRProxy->hasExpectedDepth()) {
                std::vector<float> depthPremul(static_cast<size_t>(w) * h);
                for (size_t i = 0; i < depthPremul.size(); ++i) depthPremul[i] = chans[i * C + 2];
                auto depth = DlssIO::unpremultiplyByAlpha(depthPremul, auxAlpha, w, h, 1);
                DlssIO::writeNpyFloat32(p + "_depth.npy", depth, {h, w});
            }

            if (_splatRProxy->hasMotionVectors()) {
                std::vector<float> mvPremul(static_cast<size_t>(w) * h * 2);
                for (size_t i = 0; i < auxAlpha.size(); ++i) {
                    mvPremul[i * 2 + 0] = chans[i * C + 0];
                    mvPremul[i * 2 + 1] = chans[i * C + 1];
                }
                auto mv = DlssIO::unpremultiplyByAlpha(mvPremul, auxAlpha, w, h, 2);
                DlssIO::writeNpyFloat32(p + "_mv.npy", mv, {h, w, 2});
            }
        }

        // fg: a separate attachment (fg*alpha, 0, 0, alpha) -- own alpha, own
        // un-premultiply, independent of the aux texture above. RGBA16Float
        // today (kept safe per getFgTexture()'s own comment: fg is [0,1]-
        // bounded like alpha itself), but read format-agnostically anyway.
        if (_splatRProxy->hasForegroundCoverage()) {
            const uint32_t C = _splatRProxy->getFgChannels();  // always 4 -- lux 5d5630c made this a
                                                                // runtime accessor (was kFgChannels)
            std::vector<float> chans = readTextureAsFloats(_ctx, _splatRProxy->getFgTexture(), w, h, C);
            std::vector<float> fgPremul(static_cast<size_t>(w) * h);
            std::vector<float> fgAlpha(static_cast<size_t>(w) * h);
            for (size_t i = 0; i < fgPremul.size(); ++i) {
                fgPremul[i] = chans[i * C + 0];
                fgAlpha[i] = chans[i * C + 3];
            }
            auto fg = DlssIO::unpremultiplyByAlpha(fgPremul, fgAlpha, w, h, 1);
            DlssIO::writeNpyFloat32(p + "_fg.npy", fg, {h, w});
        }

        NSLog(@"[SplatView] B1 dump written: %@_{color.png,color.npy,depth.npy,mv.npy,fg.npy}", prefix);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.hudLabel.text = [self.hudLabel.text stringByAppendingString:@"\n[B1 proxy dump saved]"];
        });
    } catch (const std::exception &e) {
        NSLog(@"[SplatView] B1 dump failed: %s", e.what());
    }
}

// B2 validation dump: reads _netInput's packed fp16 NHWC output buffer back
// to CPU, converts to float32 (DlssIO::halfToFloat -- already linked, no new
// fp16 dependency), and writes <Documents>/netinput_f10.npy
// [netH, netW, 18+hiddenChannels] for comparison against
// mobiledlss.train.model.build_input (+ sphere_uv, disocclusion_mask, and
// SceneTexture.forward folded in -- see NetInputAssembly.mm) on the same
// dumped B1 frame, via the mobiledlss venv.
- (void)dumpNetInput {
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *outPath = [docPaths.firstObject stringByAppendingPathComponent:@"netinput_f10.npy"];
    try {
        uint32_t netW = _netInput.getNetW(), netH = _netInput.getNetH(), ch = _netInput.getChannels();
        MTL::Buffer *buf = _netInput.getOutputBuffer();
        const uint16_t *halfData = static_cast<const uint16_t *>(buf->contents());
        size_t count = static_cast<size_t>(netW) * netH * ch;
        std::vector<float> f32(count);
        for (size_t i = 0; i < count; i++) f32[i] = DlssIO::halfToFloat(halfData[i]);
        DlssIO::writeNpyFloat32(std::string(outPath.UTF8String), f32, {netH, netW, ch});
        NSLog(@"[SplatView] B2 dump written: %@ (%u x %u x %u)", outPath, netH, netW, ch);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.hudLabel.text = [self.hudLabel.text stringByAppendingString:@"\n[B2 net-input dump saved]"];
        });
    } catch (const std::exception &e) {
        NSLog(@"[SplatView] B2 dump failed: %s", e.what());
    }
}

// B3 validation dump: raw packed UNet output (kernel logits + blend logits +
// hidden_raw, all pre-activation -- matches ParamPredUNet.forward_packed's
// single concatenated tensor) at the padded net resolution, for comparison
// against PyTorch on the mobiledlss venv.
- (void)dumpUnetOutput {
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *outPath = [docPaths.firstObject stringByAppendingPathComponent:@"unet_output_f10.npy"];
    try {
        uint32_t netW = _unet.getNetW(), netH = _unet.getNetH(), ch = _unet.getOutChannels();
        const uint16_t *halfData = static_cast<const uint16_t *>(_unetOutputBuffer->contents());
        size_t count = static_cast<size_t>(netW) * netH * ch;
        std::vector<float> f32(count);
        for (size_t i = 0; i < count; i++) f32[i] = DlssIO::halfToFloat(halfData[i]);
        DlssIO::writeNpyFloat32(std::string(outPath.UTF8String), f32, {netH, netW, ch});
        NSLog(@"[SplatView] B3 dump written: %@ (%u x %u x %u)", outPath, netH, netW, ch);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.hudLabel.text = [self.hudLabel.text stringByAppendingString:@"\n[B3 unet-output dump saved]"];
        });
    } catch (const std::exception &e) {
        NSLog(@"[SplatView] B3 dump failed: %s", e.what());
    }
}

// B4 validation dump: reconstruct output at frame 10 (RGBA16Float, target
// res) -- read back via the existing MetalScreenshot/DlssIO helpers (same
// un-premultiply-free path as a plain colour texture -- reconstruct's
// output is already straight, non-premultiplied RGB, alpha=1).
- (void)dumpReconOutput {
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *prefix = [docPaths.firstObject stringByAppendingPathComponent:@"recon_f10"];
    std::string p = std::string(prefix.UTF8String);
    try {
        MetalScreenshot::saveTextureToPNG(_ctx, _reconOutputTex, kTargetW, kTargetH, p + ".png");
        auto raw = MetalScreenshot::readTextureRaw(_ctx, _reconOutputTex, kTargetW, kTargetH, 8);
        std::vector<uint8_t> unusedRgba8;
        auto colorF32 = DlssIO::convertRgba16fColorAttachment(raw, kTargetW, kTargetH, unusedRgba8);
        DlssIO::writeNpyFloat32(p + ".npy", colorF32, {kTargetH, kTargetW, 4});
        NSLog(@"[SplatView] B4 dump written: %@.{png,npy}", prefix);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.hudLabel.text = [self.hudLabel.text stringByAppendingString:@"\n[B4 recon dump saved]"];
        });
    } catch (const std::exception &e) {
        NSLog(@"[SplatView] B4 dump failed: %s", e.what());
    }
}

// Frame-10 "everything reconstruct.run() consumes this frame" dump, taken
// right BEFORE the call (so prevColor_/prevHidden_/prevDepth are the
// history side, not yet overwritten by this frame's own output) --
// completing what dumpProxyFrame/dumpNetInput/dumpUnetOutput already cover
// (proxy colour/mv/depth, hidden_in, packed kernel/blend/hidden_raw logits)
// so the whole reconstruct step can be reproduced bit-for-bit in PyTorch.
- (void)dumpReconInputsWithPrevDepth:(MTL::Texture *)prevDepthTex jitterTargetX:(float)jx jitterTargetY:(float)jy {
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *prefix = [docPaths.firstObject stringByAppendingPathComponent:@"recon_in_f10"];
    std::string p = std::string(prefix.UTF8String);
    try {
        // history-in: last frame's own composited output (target res, RGBA16Float,
        // already-straight non-premultiplied RGB per ReconstructPass's own output
        // convention -- same read path as dumpReconOutput).
        {
            auto raw = MetalScreenshot::readTextureRaw(_ctx, _reconstruct.getPrevColorTexture(), kTargetW, kTargetH, 8);
            std::vector<uint8_t> unusedRgba8;
            auto colorF32 = DlssIO::convertRgba16fColorAttachment(raw, kTargetW, kTargetH, unusedRgba8);
            DlssIO::writeNpyFloat32(p + "_prevcolor.npy", colorF32, {kTargetH, kTargetW, 4});
        }
        // prev hidden (target res, NHWC fp16 -> f32).
        {
            uint32_t hiddenChannels = _unet.getHiddenChannels();
            const uint16_t *halfData = static_cast<const uint16_t *>(_reconstruct.getPrevHiddenBuffer()->contents());
            size_t count = static_cast<size_t>(kTargetW) * kTargetH * hiddenChannels;
            std::vector<float> f32(count);
            for (size_t i = 0; i < count; i++) f32[i] = DlssIO::halfToFloat(halfData[i]);
            DlssIO::writeNpyFloat32(p + "_prevhidden.npy", f32, {kTargetH, kTargetW, hiddenChannels});
        }
        // prev proxy depth (proxy res, R32Float, unpremultiplied -- NetInputAssembly's
        // own ping-pong history, the same tensor reconstruct.run()'s disocclusion
        // check reads as `prevDepthTex`).
        {
            auto raw = MetalScreenshot::readTextureRaw(_ctx, prevDepthTex, kProxyW, kProxyH, 4);
            std::vector<float> depth(static_cast<size_t>(kProxyW) * kProxyH);
            std::memcpy(depth.data(), raw.data(), raw.size());
            DlssIO::writeNpyFloat32(p + "_prevproxydepth.npy", depth, {kProxyH, kProxyW});
        }
        // jitter (raw target-pixel units, as passed to reconstruct.run()/apply_kernel).
        DlssIO::writeNpyFloat32(p + "_jitter.npy", {jx, jy}, {2});
        NSLog(@"[SplatView] B4-inputs dump written: %@_{prevcolor,prevhidden,prevproxydepth,jitter}.npy", prefix);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.hudLabel.text = [self.hudLabel.text stringByAppendingString:@"\n[B4 recon-inputs dump saved]"];
        });
    } catch (const std::exception &e) {
        NSLog(@"[SplatView] B4-inputs dump failed: %s", e.what());
    }
}

// 16-frame orbit PSNR test (task spec): dumps frame `frame`'s straight
// (un-premultiplied) RGB colour from `tex` (any target-res RGBA16Float
// texture -- works for both _splatRTarget's premultiplied render output and
// _reconOutputTex's already-straight one, since convertRgba16fColorAttachment
// does the alpha division itself) as <Documents>/seq_<tag>_f<frame>.npy.
// Run once with LUX_START_MODE=Target and once with LUX_START_MODE=
// Reconstruction (separate app launches -- Target's full-res pass and the
// Reconstruction chain are mutually exclusive per-mode now), pull both sets,
// and PSNR-compare pairwise per frame index in Python (same deterministic
// orbit camera per frame index in both runs).
- (void)dumpSeqFrame:(MTL::Texture *)tex tag:(NSString *)tag frame:(int)frame {
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *outPath = [docPaths.firstObject
        stringByAppendingPathComponent:[NSString stringWithFormat:@"seq_%@_f%d.npy", tag, frame]];
    try {
        auto raw = MetalScreenshot::readTextureRaw(_ctx, tex, kTargetW, kTargetH, 8);
        std::vector<uint8_t> unusedRgba8;
        auto colorF32 = DlssIO::convertRgba16fColorAttachment(raw, kTargetW, kTargetH, unusedRgba8);
        DlssIO::writeNpyFloat32(std::string(outPath.UTF8String), colorF32, {kTargetH, kTargetW, 4});
    } catch (const std::exception &e) {
        NSLog(@"[SplatView] seq dump (%s f%d) failed: %s", tag.UTF8String, frame, e.what());
    }
}

// LUX_PSNR_FRAMES (task spec): runs the Target full-res pass (same camera/
// morph convention as the DisplayModeTarget branch, just offscreen into
// _splatRTarget's own output texture -- never touches the drawable, never
// mutates any Reconstruction-chain state) and a Bicubic reference upscale of
// this frame's already-computed proxy colour (same upscale_proxy kernel/
// convention as the Bicubic display mode, into the offscreen
// _psnrBicubicTex), then reads all three target-res textures back to CPU
// (readTextureRaw + convertRgba16fColorAttachment -- same un-premultiply
// convention as dumpSeqFrame, works for both _splatRTarget's premultiplied
// output and _reconOutputTex's already-straight one) and logs/records PSNR
// + the mean blend alpha/w_m + hidden-state RMS for this frame. Called from
// -tick: strictly AFTER _reconstruct.run() for this frame, so
// _reconstruct.getBlendDebugBuffer()/getPrevHiddenBuffer() both read what
// run() just wrote (see ReconstructPass.h's getPrevHiddenBuffer() doc for
// the before-vs-after pingIndex_ distinction).
- (void)capturePsnrFrame:(int)frame eyeCur:(glm::vec3)eyeCur rCur:(glm::vec3)rCur uCur:(glm::vec3)uCur
                    fCur:(glm::vec3)fCur viewCur:(glm::mat4)viewCur projCur:(glm::mat4)projCur
                   fxCur:(float)fxCur fyCur:(float)fyCur tCur:(float)tCur {
    try {
        // --- Target (full-res, offscreen). ---
        _splatRTarget->updateCameraExplicit(eyeCur, viewCur, projCur, fxCur, fyCur);
        _splatRTarget->setJitter(0.0f, 0.0f);
        if (_splatRTarget->hasMotion()) {
            _splatRTarget->setMorphTime(tCur);
        }
        @autoreleasepool {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            _splatRTarget->render(_ctx);
            pool->release();
        }

        // --- Bicubic reference: upscale_proxy(bilinear=1, 2x) from this
        // frame's own proxy colour into the offscreen scratch texture. ---
        {
            auto *cmdBuf = _ctx.beginCommandBuffer();
            auto *enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(_upscalePipeline);
            enc->setTexture(_splatRProxy->getOutputTexture(), 0);
            enc->setTexture(_psnrBicubicTex, 1);
            uint32_t bilinear = 1;
            std::array<float, 2> scale = {static_cast<float>(kTargetW) / static_cast<float>(kProxyW),
                                           static_cast<float>(kTargetH) / static_cast<float>(kProxyH)};
            enc->setBytes(&bilinear, sizeof(bilinear), 0);
            enc->setBytes(scale.data(), sizeof(float) * 2, 1);
            MTL::Size grid(kTargetW, kTargetH, 1);
            NS::UInteger tew = _upscalePipeline->threadExecutionWidth();
            NS::UInteger maxT = _upscalePipeline->maxTotalThreadsPerThreadgroup();
            NS::UInteger th = maxT / tew;
            if (th == 0) th = 1;
            enc->dispatchThreads(grid, MTL::Size(tew, th, 1));
            enc->endEncoding();
            cmdBuf->commit();
            cmdBuf->waitUntilCompleted();
        }

        // --- Readback + PSNR. ---
        std::vector<uint8_t> unusedRgba8;
        auto targetRaw = MetalScreenshot::readTextureRaw(_ctx, _splatRTarget->getOutputTexture(), kTargetW, kTargetH, 8);
        auto targetF32 = DlssIO::convertRgba16fColorAttachment(targetRaw, kTargetW, kTargetH, unusedRgba8);
        auto reconRaw = MetalScreenshot::readTextureRaw(_ctx, _reconOutputTex, kTargetW, kTargetH, 8);
        auto reconF32 = DlssIO::convertRgba16fColorAttachment(reconRaw, kTargetW, kTargetH, unusedRgba8);
        auto bicubicRaw = MetalScreenshot::readTextureRaw(_ctx, _psnrBicubicTex, kTargetW, kTargetH, 8);
        auto bicubicF32 = DlssIO::convertRgba16fColorAttachment(bicubicRaw, kTargetW, kTargetH, unusedRgba8);

        float psnrRecon = psnrRgb(reconF32, targetF32, kTargetW, kTargetH);
        float psnrBicubic = psnrRgb(bicubicF32, targetF32, kTargetW, kTargetH);

        // --- Mean blend alpha (wS) / w_m (wM), post-disocclusion-renorm. ---
        double sumAlpha = 0.0, sumWm = 0.0;
        {
            const uint16_t *half = static_cast<const uint16_t *>(_reconstruct.getBlendDebugBuffer()->contents());
            size_t n = static_cast<size_t>(kTargetW) * kTargetH;
            for (size_t i = 0; i < n; i++) {
                sumAlpha += DlssIO::halfToFloat(half[i * 2 + 0]);
                sumWm += DlssIO::halfToFloat(half[i * 2 + 1]);
            }
        }
        float meanAlpha = static_cast<float>(sumAlpha / static_cast<double>(kTargetW) / static_cast<double>(kTargetH));
        float meanWm = static_cast<float>(sumWm / static_cast<double>(kTargetW) / static_cast<double>(kTargetH));

        // --- Hidden-state RMS (target-res raw hidden state just written by
        // this frame's run()). ---
        float hiddenRms;
        {
            uint32_t hiddenChannels = _unet.getHiddenChannels();
            const uint16_t *half = static_cast<const uint16_t *>(_reconstruct.getPrevHiddenBuffer()->contents());
            size_t n = static_cast<size_t>(kTargetW) * kTargetH * hiddenChannels;
            double sumSq = 0.0;
            for (size_t i = 0; i < n; i++) {
                float v = DlssIO::halfToFloat(half[i]);
                sumSq += static_cast<double>(v) * v;
            }
            hiddenRms = static_cast<float>(std::sqrt(sumSq / static_cast<double>(n)));
        }

        int relIdx = frame - _psnrStartFrame + 1;  // 1-indexed within the capture window
        _psnrLogFrame.push_back(relIdx);
        _psnrLogRecon.push_back(psnrRecon);
        _psnrLogBicubic.push_back(psnrBicubic);
        _psnrLogAlpha.push_back(meanAlpha);
        _psnrLogWm.push_back(meanWm);
        _psnrLogHiddenRms.push_back(hiddenRms);

        NSLog(@"[SplatView] PSNR frame=%d/%d recon=%.2fdB bicubic=%.2fdB alpha=%.4f w_m=%.4f hiddenRMS=%.4f",
              relIdx, _psnrFrames, psnrRecon, psnrBicubic, meanAlpha, meanWm, hiddenRms);

        if (relIdx == _psnrFrames && !_psnrDone) {
            _psnrDone = YES;
            [self finishPsnrCapture];
        }
    } catch (const std::exception &e) {
        NSLog(@"[SplatView] PSNR capture (frame %d) failed: %s", frame, e.what());
    }
}

// Writes <Documents>/psnr_rollout_<N>.csv (frame,psnr_recon_db,psnr_bicubic_db,
// mean_alpha,mean_w_m,hidden_rms -- one row per captured frame, 1-indexed)
// and logs a one-line summary at the task's own report frames (1,4,8,16,24,
// 32,48,64,96).
- (void)finishPsnrCapture {
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *outPath = [docPaths.firstObject
        stringByAppendingPathComponent:[NSString stringWithFormat:@"psnr_rollout_%d.csv", _psnrFrames]];
    NSMutableString *csv =
        [NSMutableString stringWithString:@"frame,psnr_recon_db,psnr_bicubic_db,mean_alpha,mean_w_m,hidden_rms\n"];
    for (size_t i = 0; i < _psnrLogFrame.size(); i++) {
        [csv appendFormat:@"%d,%.4f,%.4f,%.6f,%.6f,%.6f\n", _psnrLogFrame[i], _psnrLogRecon[i], _psnrLogBicubic[i],
                           _psnrLogAlpha[i], _psnrLogWm[i], _psnrLogHiddenRms[i]];
    }
    NSError *werr = nil;
    [csv writeToFile:outPath atomically:YES encoding:NSUTF8StringEncoding error:&werr];
    if (werr) {
        NSLog(@"[SplatView] PSNR capture: failed to write %@: %@", outPath, werr);
    } else {
        NSLog(@"[SplatView] PSNR capture: wrote %@ (%lu frames)", outPath, (unsigned long)_psnrLogFrame.size());
    }

    static const int kReportFrames[] = {1, 4, 8, 16, 24, 32, 48, 64, 96};
    NSMutableString *reconSummary = [NSMutableString string];
    NSMutableString *bicubicSummary = [NSMutableString string];
    for (int rf : kReportFrames) {
        if (rf > _psnrFrames) continue;
        size_t idx = static_cast<size_t>(rf - 1);
        if (idx >= _psnrLogFrame.size()) continue;
        [reconSummary appendFormat:@"f%d=%.2f ", rf, _psnrLogRecon[idx]];
        [bicubicSummary appendFormat:@"f%d=%.2f ", rf, _psnrLogBicubic[idx]];
    }
    NSLog(@"[SplatView] PSNR capture COMPLETE. recon dB: %@", reconSummary);
    NSLog(@"[SplatView] PSNR capture COMPLETE. bicubic dB: %@", bicubicSummary);
    dispatch_async(dispatch_get_main_queue(), ^{
        self.hudLabel.text = [self.hudLabel.text stringByAppendingString:@"\n[PSNR rollout capture complete]"];
    });
}

- (void)saveScreenshotFromTexture:(MTL::Texture *)texture {
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *outPath = [docPaths.firstObject stringByAppendingPathComponent:@"screenshot.png"];
    try {
        MetalScreenshot::saveTextureToPNG(_ctx, texture, kTargetW, kTargetH, std::string(outPath.UTF8String));
        NSLog(@"[SplatView] screenshot saved: %@", outPath);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.hudLabel.text = [self.hudLabel.text stringByAppendingString:@"\n[screenshot saved]"];
        });
    } catch (const std::exception &e) {
        NSLog(@"[SplatView] screenshot failed: %s", e.what());
    }
}

@end
