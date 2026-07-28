#!/bin/bash
# Run 60s + 30s silent baseline
set -e
cd /mnt/c/Users/27013/Desktop/digital-human-sdk

OUT1="$HOME/dh_60s"
rm -rf "$OUT1"
mkdir -p "$OUT1"
echo "=============================================="
echo "  Run 1: 60s zw_trimmed.mp3"
echo "=============================================="
./build/bin/pipeline_lipsync_test \
    assets 60 25 zw_trimmed.mp3 0 "$OUT1" 2>&1 | tail -35

OUT2="$HOME/dh_silent"
rm -rf "$OUT2"
mkdir -p "$OUT2"
echo "=============================================="
echo "  Run 2: 30s silent_30s.wav (baseline)"
echo "=============================================="
./build/bin/pipeline_lipsync_test \
    assets 30 25 silent_30s.wav 0 "$OUT2" 2>&1 | tail -35

echo "=== ALL DONE ==="
ls -la "$OUT1"
echo "---"
ls -la "$OUT2"
