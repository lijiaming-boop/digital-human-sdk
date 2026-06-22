#!/bin/bash
# Build script for digital-human-sdk
# Works around missing .so dev symlinks in system (libjpeg.so, libsqlite3.so)

set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
LOCAL_LIB="$HOME/.local/lib"

# 1. 创建缺失的 .so 符号链接（无需 sudo）
mkdir -p "$LOCAL_LIB"
[ ! -e "$LOCAL_LIB/libjpeg.so" ]   && ln -sf /usr/lib/x86_64-linux-gnu/libjpeg.so.8 "$LOCAL_LIB/libjpeg.so"   && echo "  -> Created libjpeg.so link"
[ ! -e "$LOCAL_LIB/libsqlite3.so" ] && ln -sf /usr/lib/x86_64-linux-gnu/libsqlite3.so.0 "$LOCAL_LIB/libsqlite3.so" && echo "  -> Created libsqlite3.so link"

# 2. CMake 配置
cmake --preset vcpkg -B "$BUILD_DIR"

# 3. 修复 CMake 生成的 make 依赖（将不存在的 .so 路径替换为 ~/.local/lib 中的版本）
for build_make in $(find "$BUILD_DIR" -name "build.make"); do
    sed -i \
        -e "s|/usr/lib/x86_64-linux-gnu/libjpeg\.so|$LOCAL_LIB/libjpeg.so|g" \
        -e "s|/usr/lib/x86_64-linux-gnu/libsqlite3\.so|$LOCAL_LIB/libsqlite3.so|g" \
        "$build_make"
done
echo "  -> Fixed build dependencies"

# 4. 编译
cmake --build "$BUILD_DIR" "$@"
