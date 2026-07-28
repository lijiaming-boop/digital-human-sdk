#!/usr/bin/env bash
# ============================================================================
# profile_flamegraph.sh — 用 perf 采样 digital-human-sdk 并生成火焰图
#
# 运行环境: WSL2 Ubuntu (perf 必须已安装: sudo apt install linux-tools-generic)
# 前置构建: build-profile 目录（RelWithDebInfo + -fno-omit-frame-pointer -g）
#
# 用法:
#   bash scripts/profile_flamegraph.sh [目标程序] [采样频率Hz]
# 示例:
#   bash scripts/profile_flamegraph.sh build-profile/bin/perf_benchmark 999
# 产物:
#   docs/perf/perf.data      原始采样数据
#   docs/perf/out.folded     折叠调用栈
#   docs/perf/flamegraph.svg 火焰图（浏览器打开可交互）
#   docs/perf/perf_report.txt perf report 文本摘要
# ============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

TARGET="${1:-build-profile/bin/perf_benchmark}"
FREQ="${2:-999}"                       # 999Hz 避免与内核 1000Hz tick 谐振
CGMODE="${3:-fp}"                       # fp=帧指针回栈(WSL2稳定) / dwarf=DWARF回栈
OUTDIR="$PROJECT_DIR/docs/perf"
FG="${FLAMEGRAPH_DIR:-$HOME/FlameGraph}"
mkdir -p "$OUTDIR"

# --- 定位 perf 二进制 ---
# WSL2 内核版本 (6.x) 与 apt 的 linux-tools (5.15) 不匹配，
# /usr/bin/perf 包装脚本会找不到对应版本，这里直接用已安装的实体二进制。
PERF="$(command -v perf || true)"
if [ -z "$PERF" ] || ! "$PERF" --version >/dev/null 2>&1; then
    PERF="$(ls /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf 2>/dev/null | sort -V | tail -1 || true)"
fi
[ -z "$PERF" ] && { echo "[FATAL] 找不到 perf，请先: sudo apt install linux-tools-generic"; exit 1; }
echo "[INFO] perf = $PERF ($($PERF --version 2>/dev/null || echo unknown))"
echo "[INFO] target = $TARGET  freq = ${FREQ}Hz"

# --- 采样 ---
# 默认 fp(帧指针)回栈：程序以 -fno-omit-frame-pointer 编译，WSL2 下比 dwarf 稳定
# （dwarf 每次采样要拷贝用户栈，WSL2 多线程时易报 "Bad address"）。
# -F: 采样频率  -g: 记录调用图  --: 后接被测程序
# 注: 非 root 下 mmap 受 perf_event_mlock_kb 限制，用默认缓冲大小最稳妥
echo "[STEP] perf record (call-graph=$CGMODE) ..."
"$PERF" record -F "$FREQ" --call-graph "$CGMODE" -o "$OUTDIR/perf.data" -- "$TARGET" \
    > "$OUTDIR/run_stdout.log" 2>&1 || {
        echo "[WARN] 程序退出码非 0，查看 $OUTDIR/run_stdout.log"; }

# --- 文本报告（快速定位热点函数）---
echo "[STEP] perf report (text) ..."
"$PERF" report -i "$OUTDIR/perf.data" --stdio --no-children 2>/dev/null \
    | head -80 > "$OUTDIR/perf_report.txt" || true

# --- 火焰图 ---
echo "[STEP] flamegraph ..."
"$PERF" script -i "$OUTDIR/perf.data" 2>/dev/null > "$OUTDIR/perf.script"
"$FG/stackcollapse-perf.pl" "$OUTDIR/perf.script" > "$OUTDIR/out.folded"
"$FG/flamegraph.pl" --title "digital-human-sdk perf_benchmark" \
    "$OUTDIR/out.folded" > "$OUTDIR/flamegraph.svg"

echo ""
echo "[DONE] 产物:"
echo "  火焰图:   $OUTDIR/flamegraph.svg   (浏览器打开)"
echo "  文本报告: $OUTDIR/perf_report.txt"
echo "  折叠栈:   $OUTDIR/out.folded"
