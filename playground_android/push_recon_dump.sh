#!/usr/bin/env bash
# Pushes a mobiledlss/tools/reconstruct_reference_dump.py dump directory
# onto the device for Stage 5's on-device runReconstructDump() validation
# pass (docs/rendering-engines.md) -- see android_main.cpp's initRenderer()
# comment. Only the files playground_cpp/src/reconstruct_runner.cpp's
# runReconstructDump() actually reads are staged (not build_input_f*.npy /
# ref_out_f*.npy / ref_hidden_f*.npy / memory_color_f*.npy /
# target_bguv_ref.npy / _memory_synth_clip.npz -- those are reference/debug
# files consumed only by the Mac-side Python comparison, kept local); the
# per-frame packed_params_f{t}.npy files alone are ~40MB each at this
# config (net 240x136 x 312ch fp32), so pushing everything (~1.3GB) would
# be needlessly slow.
#
# Usage: ./push_recon_dump.sh <local_dump_dir> <num_frames>
set -euo pipefail

ADB="${ADB:-/opt/homebrew/bin/adb}"
PKG=com.lux.playgroundandroid
LUX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_STAGE="/data/local/tmp/lux_stage"
LOCAL_DIR="$1"
NUM_FRAMES="$2"

push_one() {
    local local_path="$1" remote_rel="$2"
    local tmp_name
    tmp_name="$(basename "$remote_rel")"
    "$ADB" push "$local_path" "$TMP_STAGE/$tmp_name" >/dev/null
    "$ADB" shell run-as "$PKG" mkdir -p "files/$(dirname "$remote_rel")"
    "$ADB" shell "cat $TMP_STAGE/$tmp_name | run-as $PKG sh -c 'cat > files/$remote_rel'"
    "$ADB" shell rm -f "$TMP_STAGE/$tmp_name"
}

"$ADB" shell mkdir -p "$TMP_STAGE"

echo "=== pushing recon_dump meta + scene-memory assets ==="
push_one "$LOCAL_DIR/meta.json" "recon_dump/meta.json"
push_one "$LOCAL_DIR/bg_sphere.npy" "recon_dump/bg_sphere.npy"
push_one "$LOCAL_DIR/texture.npy" "recon_dump/texture.npy"
push_one "$LOCAL_DIR/memory_head.npz" "recon_dump/memory_head.npz"

echo "=== pushing per-frame reconstruct inputs (0..$((NUM_FRAMES-1))) ==="
for t in $(seq 0 $((NUM_FRAMES - 1))); do
    for name in proxy_color mv_proxy jitter packed_params disocc k_params cam_to_world; do
        push_one "$LOCAL_DIR/${name}_f${t}.npy" "recon_dump/${name}_f${t}.npy"
    done
    echo "  frame $t staged"
done

echo "=== pushing examples/reconstruct_mem_ps2 compiled pipeline ==="
for ext in warp.comp.spv apply.comp.spv bguv.comp.spv memory.comp.spv blend.comp.spv; do
    push_one "$LUX_ROOT/examples/reconstruct_mem_ps2.$ext" "examples/reconstruct_mem_ps2.$ext"
done

echo "done."
