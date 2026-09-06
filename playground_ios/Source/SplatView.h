#import <UIKit/UIKit.h>

// Full-screen live view of the lux Metal Gaussian-splat renderer
// (playground_cpp/src/metal_splat_renderer.*, hand-written MSL path), driven
// by a synthetic orbit camera (mobiledlss/datagen/camera.py::orbit_path
// convention) around the pruned juggle scene. Owns the whole Metal pipeline
// internally -- see SplatView.mm. Kept as a plain Objective-C interface
// (no C++ types) so it can be imported from AppDelegate.mm without pulling
// in the lux C++ headers there too.
@interface SplatView : UIView

@end
