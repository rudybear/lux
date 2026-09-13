#!/usr/bin/env bash
# perf-ablation variant of playground_android/push_assets.sh: pushes a
# CHOSEN compiled pipeline base + scene .glb (instead of always
# gaussian_splat_dlss + juggle_p0.8_stride4.glb), and writes perf_config.txt
# so android_main.cpp (this perf-copy's own, not playground_android/'s)
# picks them up at startup. Skips the Stage 3 input-assembly/scene-memory
# assets entirely -- the perf harness never initializes that stage (see
# android_main.cpp's initRenderer() comment).
#
# Usage: ./push_assets.sh [pipeline_base] [scene_glb_path]
#   pipeline_base   examples/<name> relative to lux-4dgs root, no extension
#                   (default: examples/gaussian_splat_dlss)
#   scene_glb_path  absolute path to the .glb to push
#                   (default: mobiledlss/demo/ios_assets/juggle_p0.8_stride4.glb)
set -euo pipefail

ADB="${ADB:-/opt/homebrew/bin/adb}"
PKG=com.lux.playgroundandroidperf
LUX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILEDLSS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../mobiledlss" && pwd)"
TMP_STAGE="/data/local/tmp/lux_stage_perf"

PIPELINE_BASE="${1:-examples/gaussian_splat_dlss}"
SCENE_GLB="${2:-$MOBILEDLSS_ROOT/demo/ios_assets/juggle_p0.8_stride4.glb}"
SCENE_NAME="$(basename "$SCENE_GLB")"

push_one() {
    local local_path="$1" remote_rel="$2"
    local tmp_name
    tmp_name="$(basename "$remote_rel")"
    "$ADB" push "$local_path" "$TMP_STAGE/$tmp_name" >/dev/null
    "$ADB" shell run-as "$PKG" mkdir -p "files/$(dirname "$remote_rel")"
    "$ADB" shell "cat $TMP_STAGE/$tmp_name | run-as $PKG sh -c 'cat > files/$remote_rel'"
    "$ADB" shell rm -f "$TMP_STAGE/$tmp_name"
}

push_one_if_exists() {
    local local_path="$1" remote_rel="$2"
    if [ -f "$local_path" ]; then
        push_one "$local_path" "$remote_rel"
    else
        echo "  (skip, not present: $local_path)"
    fi
}

echo "=== staging dir ==="
"$ADB" shell mkdir -p "$TMP_STAGE"

echo "=== pushing radix sort shaders ==="
for f in histogram prefix_sum scatter; do
    push_one "$LUX_ROOT/shaders/radix_sort/$f.comp.spv" "shaders/radix_sort/$f.comp.spv"
done

echo "=== pushing pipeline: $PIPELINE_BASE ==="
for ext in comp.spv comp.json vert.spv vert.json frag.spv frag.json; do
    push_one "$LUX_ROOT/$PIPELINE_BASE.$ext" "$PIPELINE_BASE.$ext"
done
# morph.comp.{spv,json} only exist for `motion: keyframes` pipelines.
push_one_if_exists "$LUX_ROOT/$PIPELINE_BASE.morph.comp.spv" "$PIPELINE_BASE.morph.comp.spv"
push_one_if_exists "$LUX_ROOT/$PIPELINE_BASE.morph.comp.json" "$PIPELINE_BASE.morph.comp.json"

echo "=== pushing scene: $SCENE_GLB ==="
push_one "$SCENE_GLB" "scene/$SCENE_NAME"

echo "=== writing perf_config.txt (scene rel path, shader base rel path) ==="
CFG="$(mktemp)"
printf 'scene/%s\n%s\n' "$SCENE_NAME" "$PIPELINE_BASE" > "$CFG"
push_one "$CFG" "perf_config.txt"
rm -f "$CFG"

echo "done. Remote layout (internal files dir):"
"$ADB" shell run-as "$PKG" find files -type f
