#!/bin/bash
# Run pipeline_lipsync_test on voice_30s_16k_mono.wav (full 53s)
set -e
cd /mnt/c/Users/27013/Desktop/digital-human-sdk

OUT="$HOME/dh_voice"
rm -rf "$OUT"
mkdir -p "$OUT"
echo "=============================================="
echo "  Run: voice_30s_16k_mono.wav (full ~53s)"
echo "=============================================="
# Pass 53s to truncate to file length (file is 53.7s)
./build/bin/pipeline_lipsync_test \
    assets 53 25 voice_30s_16k_mono.wav 0 "$OUT" 2>&1 | tail -40

echo "=== DONE ==="
ls -la "$OUT"
