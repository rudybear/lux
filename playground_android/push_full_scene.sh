#!/usr/bin/env bash
# Full-scene Target switch (docs/rendering-engines.md, GPU-pipelining task
# goal 1): juggle_full_stride4.glb (336,568 gaussians, unpruned, 149MB) is
# now pushed by push_assets.sh's DEFAULT run (files/scene/
# juggle_full_stride4.glb) and android_main.cpp's initRenderer() loads it
# for Target unconditionally unless the "pruned_target.txt" OPT-OUT marker
# is present -- see AppState's fullSceneTargetRenderer/useFullSceneTarget
# field comment. This script is now just a convenience for two things that
# don't need a full push_assets.sh run:
#
#   ./push_full_scene.sh            re-push just the 149MB glb (e.g. after
#                                    a corrupted/interrupted transfer)
#   ./push_full_scene.sh --pruned   drop the pruned_target.txt marker,
#                                    OPTING OUT back to the old pruned-scene
#                                    Target (e.g. for a quick A/B) without
#                                    touching the already-pushed glb
#   ./push_full_scene.sh --full     remove the pruned_target.txt marker,
#                                    opting back IN to the full-scene Target
#                                    (the default -- undoes --pruned)
#
# Either flag needs a force-stop + relaunch of the app to take effect
# (initRenderer() only checks the marker at startup).
set -euo pipefail

ADB="${ADB:-/opt/homebrew/bin/adb}"
PKG=com.lux.playgroundandroid
MOBILEDLSS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../mobiledlss" && pwd)"
TMP_STAGE="/data/local/tmp/lux_stage"

if [[ "${1:-}" == "--pruned" ]]; then
    echo "=== dropping pruned_target.txt marker (opting OUT of the full-scene Target) ==="
    "$ADB" shell "run-as $PKG sh -c 'echo 1 > files/pruned_target.txt'"
    echo "done -- force-stop + relaunch the app for this to take effect."
    exit 0
fi

if [[ "${1:-}" == "--full" ]]; then
    echo "=== removing pruned_target.txt marker (opting back IN to the full-scene Target, the default) ==="
    "$ADB" shell run-as "$PKG" rm -f files/pruned_target.txt
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

echo "done -- this is now the DEFAULT Target scene (push_assets.sh pushes it routinely too); no marker needed."
