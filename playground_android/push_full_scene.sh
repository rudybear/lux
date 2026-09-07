#!/usr/bin/env bash
# Full-scene Target switch (prep, docs/rendering-engines.md): pushes
# juggle_full_stride4.glb (336,568 gaussians, unpruned, 149MB) to
# files/scene/juggle_full_stride4.glb and drops the full_scene_target.txt
# marker android_main.cpp's initRenderer() checks for -- see AppState's
# fullSceneTargetRenderer/useFullSceneTarget field comment. Deliberately a
# SEPARATE script from push_assets.sh (not folded into its default run):
# this asset is 149MB, an order of magnitude bigger than everything else
# push_assets.sh pushes combined, and the marker flips real, currently-
# unvalidated-on-device behavior (Target switches scenes) -- both should
# only happen on purpose, not on every routine asset sync.
#
# Usage: ./push_full_scene.sh          (push + drop the marker, i.e. "go")
#        ./push_full_scene.sh --off    (remove the marker only, revert to
#                                        the pruned scene for Target --
#                                        does NOT delete the now-pushed glb,
#                                        harmless to leave staged)
set -euo pipefail

ADB="${ADB:-/opt/homebrew/bin/adb}"
PKG=com.lux.playgroundandroid
MOBILEDLSS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../mobiledlss" && pwd)"
TMP_STAGE="/data/local/tmp/lux_stage"

if [[ "${1:-}" == "--off" ]]; then
    echo "=== removing full_scene_target.txt marker (reverting Target to the pruned scene) ==="
    "$ADB" shell run-as "$PKG" rm -f files/full_scene_target.txt
    echo "done -- force-stop + relaunch the app for this to take effect."
    exit 0
fi

push_one() {
    local local_path="$1" remote_rel="$2"
    local tmp_name
    tmp_name="$(basename "$remote_rel")"
    "$ADB" push "$local_path" "$TMP_STAGE/$tmp_name"
    "$ADB" shell run-as "$PKG" mkdir -p "files/$(dirname "$remote_rel")"
    "$ADB" shell "cat $TMP_STAGE/$tmp_name | run-as $PKG sh -c 'cat > files/$remote_rel'"
    "$ADB" shell rm -f "$TMP_STAGE/$tmp_name"
}

echo "=== staging dir ==="
"$ADB" shell mkdir -p "$TMP_STAGE"

echo "=== pushing full (unpruned) scene: juggle_full_stride4.glb (149MB -- this will take a while) ==="
push_one "$MOBILEDLSS_ROOT/demo/ios_assets/juggle_full_stride4.glb" "scene/juggle_full_stride4.glb"

echo "=== dropping full_scene_target.txt marker ==="
"$ADB" shell "run-as $PKG sh -c 'echo 1 > files/full_scene_target.txt'"

echo "done -- force-stop + relaunch the app for this to take effect (initRenderer() only checks the marker at startup)."
