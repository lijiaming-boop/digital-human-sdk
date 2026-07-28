#!/bin/bash
# Rebuild pipeline_lipsync_test and run 30s fitting
set -e
cd /mnt/c/Users/27013/Desktop/digital-human-sdk

LOCAL_LIB="$HOME/.local/lib"
mkdir -p "$LOCAL_LIB"
[ ! -e "$LOCAL_LIB/libjpeg.so" ]   && ln -sf /usr/lib/x86_64-linux-gnu/libjpeg.so.8 "$LOCAL_LIB/libjpeg.so"
[ ! -e "$LOCAL_LIB/libsqlite3.so" ] && ln -sf /usr/lib/x86_64-linux-gnu/libsqlite3.so.0 "$LOCAL_LIB/libsqlite3.so"

# Fix CMake generated make deps for the libjpeg/libsqlite3 missing .so dev symlinks
for build_make in $(find build -name "build.make" 2>/dev/null); do
    sed -i \
        -e "s|/usr/lib/x86_64-linux-gnu/libjpeg\.so|$LOCAL_LIB/libjpeg.so|g" \
        -e "s|/usr/lib/x86_64-linux-gnu/libsqlite3\.so|$LOCAL_LIB/libsqlite3.so|g" \
        "$build_make"
done
echo "  -> Fixed build dependencies"

# Rebuild the target
cmake --build build --target pipeline_lipsync_test 2>&1 | tail -20
echo "=== BUILD DONE ==="

# Use WSL-native output dir for frame writes (drvfs writes JPEG slowly)
OUT_DIR="$HOME/dh_lipsync_run"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "=== RUNNING pipeline_lipsync_test 30s ==="
./build/bin/pipeline_lipsync_test \
    assets 30 25 zw_trimmed.mp3 0 "$OUT_DIR" 2>&1 | tail -80

echo "=== RUN DONE ==="
ls -la "$OUT_DIR"
