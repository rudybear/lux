#!/usr/bin/env bash
# Pushes the luxc-compiled shaders + pruned scene onto the device, laid out
# under the app's INTERNAL files dir (getFilesDir(), /data/user/0/<pkg>/files)
# with the SAME relative paths the desktop CLI (playground_cpp/src/main.cpp)
# expects relative to its cwd (examples/<base>.*.spv,
# shaders/radix_sort/*.spv) -- android_main.cpp chdir()s here at startup so
# splat_renderer.cpp's SpvLoader calls (plain relative std::ifstream paths,
# unmodified from the desktop source) resolve unchanged.
#
# NOTE: we stage through /data/local/tmp + `run-as` rather than `adb push`
# straight into /sdcard/Android/data/<pkg>/files -- that path LOOKS
# world-writable via `adb shell ls -l` (rw-rw-rw-) but files the shell UID
# writes there aren't reliably readable by the app's own UID under Android
# 11+ scoped storage (cgltf_parse_file came back file_not_found on a file
# that objectively existed at the right size). run-as (the app is
# android:debuggable="true") copies as the app's own UID into its private
# internal storage instead, which has no such ambiguity.
set -euo pipefail

ADB="${ADB:-/opt/homebrew/bin/adb}"
PKG=com.lux.playgroundandroid
LUX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILEDLSS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../mobiledlss" && pwd)"
TMP_STAGE="/data/local/tmp/lux_stage"

push_one() {
    local local_path="$1" remote_rel="$2"
    local tmp_name
    tmp_name="$(basename "$remote_rel")"
    "$ADB" push "$local_path" "$TMP_STAGE/$tmp_name" >/dev/null
    "$ADB" shell run-as "$PKG" mkdir -p "files/$(dirname "$remote_rel")"
    "$ADB" shell "cat $TMP_STAGE/$tmp_name | run-as $PKG sh -c 'cat > files/$remote_rel'"
    "$ADB" shell rm -f "$TMP_STAGE/$tmp_name"
}

echo "=== staging dir ==="
"$ADB" shell mkdir -p "$TMP_STAGE"

echo "=== pushing radix sort shaders ==="
for f in histogram prefix_sum scatter; do
    push_one "$LUX_ROOT/shaders/radix_sort/$f.comp.spv" "shaders/radix_sort/$f.comp.spv"
done

echo "=== pushing gaussian_splat_dlss pipeline (compiled by luxc) ==="
for ext in comp.spv comp.json vert.spv vert.json frag.spv frag.json morph.comp.spv morph.comp.json; do
    push_one "$LUX_ROOT/examples/gaussian_splat_dlss.$ext" "examples/gaussian_splat_dlss.$ext"
done

echo "=== pushing pruned scene (juggle_p0.8_stride4.glb, ~136MB) ==="
push_one "$MOBILEDLSS_ROOT/demo/ios_assets/juggle_p0.8_stride4.glb" "scene/juggle_p0.8_stride4.glb"

echo "done. Remote layout (internal files dir):"
"$ADB" shell run-as "$PKG" find files -type f
