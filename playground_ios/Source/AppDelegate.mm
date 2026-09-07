// Minimal AppDelegate: no storyboard. Puts a full-screen SplatView (see
// SplatView.h/.mm) up as the root view -- it owns the whole lux Metal splat
// pipeline (MetalContext/MetalSceneManager/MetalSplatRenderer, reused as-is
// from playground_cpp/src) and drives it with a CADisplayLink.
#import "AppDelegate.h"
#import "SplatView.h"

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    self.window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];

    SplatView *view = [[SplatView alloc] initWithFrame:self.window.bounds];
    view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;

    UIViewController *vc = [[UIViewController alloc] init];
    vc.view = view;

    self.window.rootViewController = vc;
    [self.window makeKeyAndVisible];

    // Keep the screen awake while this demo is in the foreground -- the
    // device's own Auto-Lock setting isn't honored reliably (Low Power Mode
    // forces a short timeout), which was interrupting on-device fps runs.
    // This only disables the idle timer; it does not change
    // suspend/terminate behavior when the app is backgrounded.
    [UIApplication sharedApplication].idleTimerDisabled = YES;

    return YES;
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
    // Re-assert on every foreground transition -- UIKit resets
    // idleTimerDisabled to NO when the app resigns active/backgrounds.
    [UIApplication sharedApplication].idleTimerDisabled = YES;
}

@end
