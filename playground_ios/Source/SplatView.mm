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

// Toggle to "juggle_p0.8_stride2" once app size / load time budget allows
// (see task notes: stride4 first for size).
static NSString *const kSceneAssetName = @"juggle_p0.8_stride4";

// Builds the OpenCV-convention world->camera viewmat + GL view/proj matrices
// for orbit frame `frame`, exactly matching
// mobiledlss/datagen/camera.py::look_at + orbit_path + intrinsics, and the
// DlssIO conversion metal_main.cpp's --camera-json path uses.
static void computeOrbitCamera(int frame, glm::vec3 &eyeOut, glm::mat4 &viewGlOut,
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

    const float fy = 0.5f * static_cast<float>(kTargetH) / tanf(glm::radians(kFovYDeg) * 0.5f);
    const float fx = fy;
    const float cx = 0.5f * static_cast<float>(kTargetW);
    const float cy = 0.5f * static_cast<float>(kTargetH);
    const glm::mat4 proj = DlssIO::buildIntrinsicsProjection(
        fx, fy, cx, cy, static_cast<float>(kTargetW), static_cast<float>(kTargetH),
        0.01f, 1000.0f, /*metalYConvention=*/true);

    eyeOut = eye;
    viewGlOut = viewGl;
    projOut = proj;
    fxOut = fx;
    fyOut = fy;
}

@interface SplatView () {
    MetalContext _ctx;
    MetalSceneManager _scene;
    std::unique_ptr<MetalSplatRenderer> _splatR;
    int _frame;
    int _loopFrames;
    BOOL _ready;
    BOOL _wantScreenshot;
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
    computeOrbitCamera(_frame, eyeCur, viewCur, projCur, fxCur, fyCur);
    computeOrbitCamera(prevFrame, eyePrev, viewPrev, projPrev, fxPrev, fyPrev);

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
