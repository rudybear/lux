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
#include "dlss_io.h"
#include "metal_screenshot.h"

#include <glm/glm.hpp>
#include <array>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

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
static void computeOrbitCamera(int frame, uint32_t width, uint32_t height,
                                glm::vec3 &eyeOut, glm::mat4 &viewGlOut,
                                glm::mat4 &projOut, float &fxOut, float &fyOut) {
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
        0.01f, 1000.0f, /*metalYConvention=*/true);

    eyeOut = eye;
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
    std::unique_ptr<MetalSplatRenderer> _splatR;       // target-res, live drawable (Stage A)
    std::unique_ptr<MetalSplatRenderer> _splatRProxy;  // proxy-res, offscreen DLSS attachments (B1)
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
        // DLSS-attachment pass -- MetalSplatRenderer::init() fixes its
        // offscreen colorTarget_/motionTarget_/expectedDepthTarget_ at the
        // given width/height, so this can't share _splatR's instance.
        _splatRProxy = std::make_unique<MetalSplatRenderer>();
        _splatRProxy->init(_ctx, _scene.getSplatData(), kProxyW, kProxyH);

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

    glm::vec3 eyeCur, eyePrev;
    glm::mat4 viewCur, viewPrev, projCur, projPrev;
    float fxCur, fyCur, fxPrev, fyPrev;

    int prevFrame = (_frame > 0) ? (_frame - 1) : 0;
    computeOrbitCamera(_frame, kTargetW, kTargetH, eyeCur, viewCur, projCur, fxCur, fyCur);
    computeOrbitCamera(prevFrame, kTargetW, kTargetH, eyePrev, viewPrev, projPrev, fxPrev, fyPrev);

    float tCur = _splatR->hasMotion() ? _splatR->frameToTime(_frame) : 0.0f;
    float tPrev = _splatR->hasMotion() ? _splatR->frameToTime(prevFrame) : 0.0f;

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

    // --- B1: proxy-res (480x270) DLSS-attachment pass, jittered, MV/morph
    // history seeded exactly like the target-res pass above. Offscreen
    // (render(), not renderToDrawable()) -- correct-size aux attachments
    // only populate when colorTex == colorTarget_ at the renderer's own
    // width/height (see MetalSplatRenderer::renderToTarget's `includeAux`).
    {
        glm::vec3 peyeCur, peyePrev;
        glm::mat4 pviewCur, pviewPrev, pprojCur, pprojPrev;
        float pfxCur, pfyCur, pfxPrev, pfyPrev;
        computeOrbitCamera(_frame, kProxyW, kProxyH, peyeCur, pviewCur, pprojCur, pfxCur, pfyCur);
        computeOrbitCamera(prevFrame, kProxyW, kProxyH, peyePrev, pviewPrev, pprojPrev, pfxPrev, pfyPrev);

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

        @autoreleasepool {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            _splatRProxy->render(_ctx);
            pool->release();
        }

        if (_frame == kProxyDumpFrame && !_proxyDumped) {
            _proxyDumped = YES;
            NSLog(@"[SplatView] B1 dump: frame=%d prevFrame=%d jitterTargetPx=(%.6f,%.6f) "
                  @"jitterProxyPx=(%.6f,%.6f) tCur=%.6f tPrev=%.6f eyeCur=(%.6f,%.6f,%.6f)",
                  _frame, prevFrame, jxTarget, jyTarget, jxProxy, jyProxy, tCur, tPrev,
                  peyeCur.x, peyeCur.y, peyeCur.z);
            [self dumpProxyFrame];
        }
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
            @"lux splat live | %u splats | frame %d/%d | %.1f fps",
            _scene.getSplatData().num_splats, _frame, _loopFrames, self.fps];
        NSLog(@"[SplatView] fps=%.1f frame=%d/%d", self.fps, _frame, _loopFrames);
    }
}

- (void)handleTap:(UITapGestureRecognizer *)tap {
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
            constexpr uint32_t C = MetalSplatRenderer::kExpectedDepthChannels;  // 2: depth*alpha, alpha
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
