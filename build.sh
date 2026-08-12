#!/bin/bash
# Build script for digital-human-sdk
# Portable Linux CPU-only build via the `linux-cpu-only` CMake preset.
#
# 设计目标（P0 工程基线）：
#   - 不修改用户系统目录（不再向 $HOME/.local/lib 写符号链接）；
#   - 不修改 CMake 生成的 build.make（避免依赖具体生成器实现）；
#   - 缺失的开发符号链接通过 CMAKE_LIBRARY_PATH 在构建目录内本地解决。
#
# 前置条件：
#   - 设置 VCPKG_ROOT 指向 vcpkg 检出目录；
#   - 已通过 vcpkg 安装 ncnn（CPU-only 即可）、opencv、ffmpeg、portaudio。
#
# 可选：若系统缺少 libjpeg.so / libsqlite3.so 开发符号链接，将本地补链目录
# 通过环境变量 DIGITAL_HUMAN_LOCAL_LIB 提供，脚本会将其作为 CMAKE_LIBRARY_PATH
# 传入，CMake 在 configure 阶段即可找到，无需后处理生成文件。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
PRESET="${PRESET:-linux-cpu-only}"

# 1. 校验 VCPKG_ROOT
if [ -z "${VCPKG_ROOT:-}" ] || [ ! -d "${VCPKG_ROOT}/scripts/buildsystems" ]; then
    echo "[ERROR] VCPKG_ROOT 未设置或无效。请执行: export VCPKG_ROOT=/path/to/vcpkg" >&2
    exit 1
fi

# 2. 准备本地依赖链接路径（可选，仅在缺失开发符号链接时使用）
EXTRA_CMAKE_ARGS=()
if [ -n "${DIGITAL_HUMAN_LOCAL_LIB:-}" ] && [ -d "${DIGITAL_HUMAN_LOCAL_LIB}" ]; then
    EXTRA_CMAKE_ARGS+=(-DCMAKE_LIBRARY_PATH="${DIGITAL_HUMAN_LOCAL_LIB}")
    echo "  -> Using local lib search path: ${DIGITAL_HUMAN_LOCAL_LIB}"
fi

# 3. CMake 配置（使用可移植 preset，binaryDir 固定为 build/）
echo "[1/2] Configure (preset: ${PRESET})"
cmake --preset "${PRESET}" -B "$BUILD_DIR" "${EXTRA_CMAKE_ARGS[@]}"

# 4. 编译
echo "[2/2] Build"
cmake --build "$BUILD_DIR" "$@"
