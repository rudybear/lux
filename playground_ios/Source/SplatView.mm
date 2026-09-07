// Stage A: live orbit-camera splat rendering on device.
//
// Reuses playground_cpp/src's Metal pipeline classes exactly as the macOS
// interactive loop (metal_main.cpp) does -- MetalContext (initHeadless(),
// then pointed at our own CAMetalLayer instead of a GLFW window),
// MetalSceneManager::loadScene()/hasSplatData()/getSplatData(), and
// MetalSplatRenderer::{init,updateCameraExplicit,setPreviousCameraExplicit,
// setMorphTime,seedPreviousMorphTime,renderToDrawable}. No modifications to
// any of those files -- see playground_ios/README.md.
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
#include "metal_splat_renderer.h"
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
// training-time renderer, so the proxy pass uses it instead of the
// hand-written path. Flip this typedef (+ the two backend-specific spots
// marked below: init()'s extra shaderBase arg, kMetalYConvention,
// kExpectedDepthChannels -- all already backend-agnostic via this alias/
// the class's own static constants) back to MetalSplatRenderer as a
// fallback if the luxc path ever regresses.
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
// (see task notes: stride4 first for size).
static NSString *const kSceneAssetName = @"juggle_p0.8_stride4";

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

// Nearest/bilinear 2x upscale of the proxy colour texture straight into the
// drawable -- un-premultiplies alpha (the proxy render is premultiplied),
// forces alpha=1 (opaque display). "Bicubic" mode currently uses this same
// kernel's bilinear path as a placeholder (TODO: real bicubic) -- labelled
// "Bilinear" in the HUD/log rather than falsely claiming bicubic.
static const char *kUpscaleMSL = R"(
#include <metal_stdlib>
using namespace metal;
kernel void upscale_proxy(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           constant uint& bilinear [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint srcW = src.get_width(), srcH = src.get_height();
    float4 rgba;
    if (bilinear == 0) {
        uint sx = min(gid.x / 2, srcW - 1);
        uint sy = min(gid.y / 2, srcH - 1);
        rgba = src.read(uint2(sx, sy));
    } else {
        float fx = (float(gid.x) + 0.5) / 2.0 - 0.5;
        float fy = (float(gid.y) + 0.5) / 2.0 - 0.5;
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
// proxy-pixel jitter (what MetalSplatRenderer::setJitter expects when
// applied to a proxy-res renderer) must scale by kProxyScale themselves
// (mobiledlss/train/data.py's "jitter_proxy = jitter / S" convention).
static void taaJitterTargetPx(int frame, int period, float &jxOut, float &jyOut) {
    const int i = (frame % period) + 1;
    jxOut = haltonSeq(i, 2) - 0.5f;
    jyOut = haltonSeq(i, 3) - 0.5f;
}

@interface SplatView () {
    MetalContext _ctx;
    MetalSceneManager _scene;
    std::unique_ptr<MetalSplatRenderer> _splatR;    // target-res, live drawable (Stage A, hand path -- fine to keep as-is, just a display)
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
    int _frame;
    int _loopFrames;
    BOOL _ready;
    BOOL _wantScreenshot;
    BOOL _proxyDumped;
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
        _displayMode = DisplayModeReconstruction;

        CAMetalLayer *metalLayer = (CAMetalLayer *)self.layer;
        // MetalSplatRenderer's render pipeline hardcodes color attachment 0
        // to RGBA16Float (metal_splat_renderer.cpp's createPipelines(), to
        // match its own offscreen colorTarget_ -- see the class comment on
        // RGBA16Float there). renderToDrawable() renders straight into
        // whatever texture the drawable holds with that same pipeline, so
        // the CAMetalLayer's own pixelFormat must match it (BGRA8Unorm, the
        // CAMetalLayer default, silently reinterprets the half-float bytes
        // as 8-bit UNORM and comes out as garbage -- discovered via an
        // on-device screenshot before this fix).
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
        NSLog(@"[SplatView] loading scene: %@", path);
        _scene.loadScene(std::string(path.UTF8String));

        if (!_scene.hasSplatData()) {
            *errorOut = "scene has no KHR_gaussian_splatting data";
            return NO;
        }

        _splatR = std::make_unique<MetalSplatRenderer>();
        _splatR->init(_ctx, _scene.getSplatData(), kTargetW, kTargetH);

        // B1: a second renderer instance dedicated to the proxy-res (480x270)
        // DLSS-attachment pass -- ProxyRenderer::init() fixes its offscreen
        // colorTarget_/motionTarget_/expectedDepthTarget_ at the given
        // width/height, so this can't share _splatR's instance. luxc needs
        // the compiled pipeline base (examples/gaussian_splat_dlss.*.spv,
        // bundled as a folder reference) -- the same pipeline
        // gaussian_splat_dlss.lux compiles to for the Vulkan/hand-Metal
        // parity comparison in commit 08cfe8e.
        _splatRProxy = std::make_unique<ProxyRenderer>();
        _splatRProxy->init(_ctx, _scene.getSplatData(), "examples/gaussian_splat_dlss", kProxyW, kProxyH);

        // B2: net input assembly (bg-sphere UV + texture sample + disocclusion
        // + box-pool). texture.npy/bg_sphere.npy exported by
        // demo/ios_assets/export_ios_weights.py (mobiledlss repo).
        NSString *texPath = [[NSBundle mainBundle] pathForResource:@"texture" ofType:@"npy"];
        NSString *spherePath = [[NSBundle mainBundle] pathForResource:@"bg_sphere" ofType:@"npy"];
        if (!texPath || !spherePath) {
            *errorOut = "bundle resource not found: texture.npy / bg_sphere.npy";
            return NO;
        }
        _netInput.init(_ctx, std::string(texPath.UTF8String), std::string(spherePath.UTF8String),
                        kProxyW, kProxyH, /*paramStride=*/2, /*hiddenChannels=*/8,
                        /*depthAlphaOffset=*/ProxyRenderer::kExpectedDepthChannels - 1);

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
        }

        if (_splatR->hasMotion()) {
            float dur = _splatR->animationDuration();
            _loopFrames = std::max(1, static_cast<int>(std::round(dur * kSimFps)));
        }
        NSLog(@"[SplatView] ready: %u splats, loopFrames=%d", _scene.getSplatData().num_splats, _loopFrames);
        return YES;
    } catch (const std::exception &e) {
        *errorOut = e.what();
        return NO;
    }
}

- (void)startDisplayLink {
    // Main-thread CAMetalLayer wiring (device + the metal-cpp bridge pointer
    // MetalSplatRenderer::renderToDrawable/nextDrawable() need), now that
    // _ctx.device exists (created off-thread in loadSceneAndInitRenderer).
    CAMetalLayer *metalLayer = (CAMetalLayer *)self.layer;
    metalLayer.device = (__bridge id<MTLDevice>)_ctx.device;
    _ctx.metalLayer = (__bridge CA::MetalLayer *)metalLayer;

    self.displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(tick:)];
    self.displayLink.preferredFramesPerSecond = 30;
    [self.displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
    self.fpsWindowStart = CACurrentMediaTime();
    self.fpsFrameCount = 0;
}

- (void)tick:(CADisplayLink *)link {
    if (!_ready || !_splatR) return;

    int prevFrame = (_frame > 0) ? (_frame - 1) : 0;
    float tCur = _splatR->hasMotion() ? _splatR->frameToTime(_frame) : 0.0f;
    float tPrev = _splatR->hasMotion() ? _splatR->frameToTime(prevFrame) : 0.0f;

    _msTarget = 0.0;

    // --- Target mode ONLY: full-res (960x540) hand-path splat render,
    // straight to the drawable (renderToDrawable presents it itself). Per
    // coordinator: this expensive pass (CPU sort ~19ms + full 960x540 draw)
    // must NOT run in Proxy/Bicubic/Reconstruction modes.
    if (_displayMode == DisplayModeTarget) {
        CFTimeInterval t0 = CACurrentMediaTime();
        glm::vec3 eyeCur, eyePrev, rCur, uCur, fCur, rPrev, uPrev, fPrev;
        glm::mat4 viewCur, viewPrev, projCur, projPrev;
        float fxCur, fyCur, fxPrev, fyPrev;
        computeOrbitCamera(_frame, kTargetW, kTargetH, MetalSplatRenderer::kMetalYConvention,
                            eyeCur, rCur, uCur, fCur, viewCur, projCur, fxCur, fyCur);
        computeOrbitCamera(prevFrame, kTargetW, kTargetH, MetalSplatRenderer::kMetalYConvention,
                            eyePrev, rPrev, uPrev, fPrev, viewPrev, projPrev, fxPrev, fyPrev);
        _splatR->updateCameraExplicit(eyeCur, viewCur, projCur, fxCur, fyCur);
        _splatR->setPreviousCameraExplicit(viewPrev, projPrev);
        if (_splatR->hasMotion()) {
            _splatR->setMorphTime(tCur);
            _splatR->seedPreviousMorphTime(tPrev);
        }
        @autoreleasepool {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            CA::MetalDrawable *drawable = _ctx.metalLayer->nextDrawable();
            if (drawable) {
                _splatR->renderToDrawable(_ctx, drawable);
                if (_wantScreenshot) {
                    _wantScreenshot = NO;
                    [self saveScreenshotFromTexture:drawable->texture()];
                }
            }
            pool->release();
        }
        _msTarget = (CACurrentMediaTime() - t0) * 1000.0;
    }

    // --- Proxy pass (B1) + net input assembly (B2) + net (B3): ALWAYS runs,
    // regardless of display mode, so the recurrent hidden state (B4) stays
    // continuous across mode switches -- only the extra full-res Target
    // pass above is mode-gated.
    glm::vec3 peyeCur, peyePrev, prCur, puCur, pfCur, prPrev, puPrev, pfPrev;
    glm::mat4 pviewCur, pviewPrev, pprojCur, pprojPrev;
    float pfxCur, pfyCur, pfxPrev, pfyPrev;
    computeOrbitCamera(_frame, kProxyW, kProxyH, ProxyRenderer::kMetalYConvention,
                        peyeCur, prCur, puCur, pfCur, pviewCur, pprojCur, pfxCur, pfyCur);
    computeOrbitCamera(prevFrame, kProxyW, kProxyH, ProxyRenderer::kMetalYConvention,
                        peyePrev, prPrev, puPrev, pfPrev, pviewPrev, pprojPrev, pfxPrev, pfyPrev);

    // mobiledlss/train/data.py convention: jitter_proxy = jitter_target / S.
    float jxTarget, jyTarget;
    taaJitterTargetPx(_frame, /*period=*/16, jxTarget, jyTarget);
    float jxProxy = jxTarget * kProxyScale;
    float jyProxy = jyTarget * kProxyScale;

    _splatRProxy->updateCameraExplicit(peyeCur, pviewCur, pprojCur, pfxCur, pfyCur);
    _splatRProxy->setPreviousCameraExplicit(pviewPrev, pprojPrev);
    if (_splatRProxy->hasMotion()) {
        _splatRProxy->setMorphTime(tCur);
        _splatRProxy->seedPreviousMorphTime(tPrev);
    }
    _splatRProxy->setJitter(jxProxy, jyProxy);

    CFTimeInterval tp0 = CACurrentMediaTime();
    @autoreleasepool {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        _splatRProxy->render(_ctx);
        pool->release();
    }
    CFTimeInterval tp1 = CACurrentMediaTime();
    _msProxy = (tp1 - tp0) * 1000.0;

    // B4 step 0 (must run BEFORE B2/B3 this frame): warp+downsample last
    // frame's raw hidden state into this frame's net-res hidden_in.
    BOOL isFirstReconFrame = _netInput.isNextFrameFirst();
    _reconstruct.prepareHiddenInput(_ctx, _splatRProxy->getMotionTexture(), isFirstReconFrame,
                                     _reconstruct.getHiddenInputBuffer());

    // B2: assemble the 26-ch net input from this frame's proxy DLSS
    // attachments (hidden_in from B4's recurrence above).
    _netInput.run(_ctx, _splatRProxy->getOutputTexture(), _splatRProxy->getExpectedDepthTexture(),
                  _splatRProxy->getMotionTexture(), _reconstruct.getHiddenInputBuffer(),
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

    // B4: reconstruct-with-memory -- ALWAYS runs (own offscreen target,
    // _reconOutputTex) so the hidden/prevColor history stays continuous
    // across display-mode switches, exactly like the proxy/net passes above.
    {
        glm::vec3 teyeCur, teyePrev, trCur, tuCur, tfCur, trPrev, tuPrev, tfPrev;
        glm::mat4 tviewCur, tviewPrev, tprojCur, tprojPrev;
        float tfxCur, tfyCur, tfxPrev, tfyPrev;
        computeOrbitCamera(_frame, kTargetW, kTargetH, MetalSplatRenderer::kMetalYConvention,
                            teyeCur, trCur, tuCur, tfCur, tviewCur, tprojCur, tfxCur, tfyCur);
        MTL::Texture *curDepth = _netInput.getDepthWrittenThisFrame();
        MTL::Texture *prevDepth = _netInput.getDepthFromPreviousFrame();
        BOOL wasFirst = _netInput.wasFirstFrame();
        _reconstruct.run(_ctx, _splatRProxy->getOutputTexture(), _splatRProxy->getMotionTexture(), curDepth,
                          prevDepth, wasFirst, _unetOutputBuffer, _reconOutputTex, teyeCur, trCur, tuCur, tfCur,
                          tfxCur, tfyCur, 0.5f * kTargetW, 0.5f * kTargetH, jxTarget, jyTarget);
    }
    CFTimeInterval tp4 = CACurrentMediaTime();
    _msRecon = (tp4 - tp3) * 1000.0;

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
    }

    // --- Display (Proxy/Bicubic/Reconstruction): cheap upscale of the
    // proxy colour texture straight into the drawable. Target mode already
    // presented its own drawable above.
    if (_displayMode != DisplayModeTarget) {
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
                    uint32_t bilinear = (_displayMode == DisplayModeBicubic) ? 1 : 0;
                    auto *enc = cmdBuf->computeCommandEncoder();
                    enc->setComputePipelineState(_upscalePipeline);
                    enc->setTexture(_splatRProxy->getOutputTexture(), 0);
                    enc->setTexture(drawable->texture(), 1);
                    enc->setBytes(&bilinear, sizeof(bilinear), 0);
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
            }
            pool->release();
        }
        _msDisplay = (CACurrentMediaTime() - td0) * 1000.0;
    } else {
        _msDisplay = 0.0;
    }

    _frame = (_frame + 1) % _loopFrames;

    // Auto-dump a screenshot every ~3s too (no UI-automation tap injection
    // available over devicectl -- see -handleTap: for the interactive path).
    if (_frame % (int)(kSimFps * 3.0f) == 0) {
        _wantScreenshot = YES;
    }

    self.fpsFrameCount += 1;
    CFTimeInterval now = CACurrentMediaTime();
    CFTimeInterval elapsed = now - self.fpsWindowStart;
    if (elapsed >= 0.5) {
        self.fps = self.fpsFrameCount / elapsed;
        self.fpsFrameCount = 0;
        self.fpsWindowStart = now;
        self.hudLabel.text = [NSString stringWithFormat:
            @"%s | %u splats | frame %d/%d | %.1f fps\n"
            @"target=%.1fms proxy=%.1fms input=%.1fms net=%.1fms recon=%.1fms disp=%.1fms",
            kDisplayModeNames[_displayMode], _scene.getSplatData().num_splats, _frame, _loopFrames, self.fps,
            _msTarget, _msProxy, _msInput, _msNet, _msRecon, _msDisplay];
        NSLog(@"[SplatView] mode=%s fps=%.1f frame=%d/%d | target=%.2fms proxy=%.2fms input=%.2fms "
              @"net=%.2fms recon=%.2fms display=%.2fms total=%.2fms",
              kDisplayModeNames[_displayMode], self.fps, _frame, _loopFrames,
              _msTarget, _msProxy, _msInput, _msNet, _msRecon, _msDisplay,
              _msTarget + _msProxy + _msInput + _msNet + _msRecon + _msDisplay);
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
    NSArray<NSString *> *docPaths =
        NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *prefix = [docPaths.firstObject stringByAppendingPathComponent:@"proxy_f10"];
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

        if (_splatRProxy->hasExpectedDepth()) {
            constexpr uint32_t C = ProxyRenderer::kExpectedDepthChannels;  // luxc: 4 (RGBA32Float)
            auto raw = MetalScreenshot::readTextureRaw(_ctx, _splatRProxy->getExpectedDepthTexture(), w, h, C * 4);
            std::vector<float> chans(static_cast<size_t>(w) * h * C);
            std::memcpy(chans.data(), raw.data(), raw.size());
            std::vector<float> depthPremul(static_cast<size_t>(w) * h);
            std::vector<float> depthAlpha(static_cast<size_t>(w) * h);
            for (size_t i = 0; i < depthPremul.size(); ++i) {
                depthPremul[i] = chans[i * C + 0];
                depthAlpha[i] = chans[i * C + (C - 1)];
            }
            auto depth = DlssIO::unpremultiplyByAlpha(depthPremul, depthAlpha, w, h, 1);
            DlssIO::writeNpyFloat32(p + "_depth.npy", depth, {h, w});
        }

        if (_splatRProxy->hasMotionVectors()) {
            auto raw = MetalScreenshot::readTextureRaw(_ctx, _splatRProxy->getMotionTexture(), w, h, 16);
            std::vector<float> rgba(static_cast<size_t>(w) * h * 4);
            std::memcpy(rgba.data(), raw.data(), raw.size());
            std::vector<float> mvPremul(static_cast<size_t>(w) * h * 2);
            std::vector<float> mvAlpha(static_cast<size_t>(w) * h);
            for (size_t i = 0; i < mvAlpha.size(); ++i) {
                mvPremul[i * 2 + 0] = rgba[i * 4 + 0];
                mvPremul[i * 2 + 1] = rgba[i * 4 + 1];
                mvAlpha[i] = rgba[i * 4 + 3];
            }
            auto mv = DlssIO::unpremultiplyByAlpha(mvPremul, mvAlpha, w, h, 2);
            DlssIO::writeNpyFloat32(p + "_mv.npy", mv, {h, w, 2});
        }

        NSLog(@"[SplatView] B1 dump written: %@_{color.png,color.npy,depth.npy,mv.npy}", prefix);
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
