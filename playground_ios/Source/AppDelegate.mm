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

    return YES;
}

@end
