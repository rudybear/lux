#!/usr/bin/env bash
# Reproducible build for the mobile-DLSS live-rendering demo (Stage A):
# cross-compiles playground_android's NativeActivity .so with the NDK's
# CMake toolchain file (reusing playground_cpp's Vulkan splat pipeline by
# reference), then hand-packages an APK with aapt2/apksigner (no Gradle) --
# same "no SDK/Gradle build" philosophy as bench/vulkan_android/build.sh in
# mobiledlss, just wrapped in an installable NativeActivity package since we
# need an on-screen swapchain instead of a headless CLI executable.
#
# Usage: ./build.sh [install|run]
#   (no args)  configure + build the .so + package signed APK
#   install    ...and adb install -r it
#   run        ...and adb install -r + launch it
set -euo pipefail

SDK_ROOT="${ANDROID_SDK_ROOT:-/opt/homebrew/share/android-commandlinetools}"
NDK="${ANDROID_NDK_HOME:-/opt/homebrew/share/android-ndk}"
JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home}"
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"
BUILD_TOOLS="$SDK_ROOT/build-tools/35.0.0"
PLATFORM_JAR="$SDK_ROOT/platforms/android-35/android.jar"
ADB="${ADB:-/opt/homebrew/bin/adb}"
API_LEVEL=29
ABI=arm64-v8a
PKG=com.lux.playgroundandroid

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SRC_DIR/build"
APK_STAGE="$BUILD_DIR/apk_stage"
OUT_APK_UNSIGNED="$BUILD_DIR/lux_playground.unsigned.apk"
OUT_APK_ALIGNED="$BUILD_DIR/lux_playground.aligned.apk"
OUT_APK="$BUILD_DIR/lux_playground.apk"
KEYSTORE="$BUILD_DIR/debug.keystore"

echo "=== [1/6] cmake configure (NDK r$(basename "$NDK" | grep -o '[0-9.]*' || true)) ==="
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="android-$API_LEVEL" \
    -DCMAKE_BUILD_TYPE=Release

echo "=== [2/6] cmake build ==="
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)"

SO_PATH="$BUILD_DIR/liblux_android.so"
if [ ! -f "$SO_PATH" ]; then
    echo "error: $SO_PATH not built" >&2
    exit 1
fi
echo "built: $SO_PATH ($(du -h "$SO_PATH" | cut -f1))"

echo "=== [3/6] stage APK contents ==="
rm -rf "$APK_STAGE"
mkdir -p "$APK_STAGE/lib/$ABI"
cp "$SO_PATH" "$APK_STAGE/lib/$ABI/liblux_android.so"

# libc++_shared.so must ship in the APK (NDK libs are not on the system
# library path) -- same requirement bench/vulkan_android's README notes for
# any C++-linked NDK binary that isn't fully static.
CXX_SHARED="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -f "$CXX_SHARED" ]; then
    cp "$CXX_SHARED" "$APK_STAGE/lib/$ABI/libc++_shared.so"
else
    echo "warning: libc++_shared.so not found at $CXX_SHARED" >&2
fi

echo "=== [4/6] aapt2 compile+link (no gradle) ==="
mkdir -p "$BUILD_DIR/aapt2_compiled"
"$BUILD_TOOLS/aapt2" link \
    -o "$OUT_APK_UNSIGNED" \
    --manifest "$SRC_DIR/manifest/AndroidManifest.xml" \
    -I "$PLATFORM_JAR" \
    --min-sdk-version "$API_LEVEL" --target-sdk-version 35

# Add native libs into the linked APK (aapt2 link only handles manifest+resources).
cd "$APK_STAGE" && zip -q -r "$OUT_APK_UNSIGNED" lib && cd "$SRC_DIR"

echo "=== [5/6] zipalign ==="
"$BUILD_TOOLS/zipalign" -f -p 4 "$OUT_APK_UNSIGNED" "$OUT_APK_ALIGNED"

echo "=== [6/6] sign (debug keystore) ==="
if [ ! -f "$KEYSTORE" ]; then
    "$JAVA_HOME/bin/keytool" -genkeypair -v -keystore "$KEYSTORE" \
        -alias androiddebugkey -storepass android -keypass android \
        -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=lux-android-debug, O=lux, C=US" >/dev/null
fi
"$BUILD_TOOLS/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:android \
    --key-pass pass:android --out "$OUT_APK" "$OUT_APK_ALIGNED"

echo "APK ready: $OUT_APK ($(du -h "$OUT_APK" | cut -f1))"

ACTION="${1:-}"
if [ "$ACTION" = "install" ] || [ "$ACTION" = "run" ]; then
    echo "=== adb install ==="
    "$ADB" install -r "$OUT_APK"
fi
if [ "$ACTION" = "run" ]; then
    echo "=== adb push assets ==="
    "$SRC_DIR/push_assets.sh"
    echo "=== launching ==="
    "$ADB" shell am start -n "$PKG/android.app.NativeActivity"
fi
