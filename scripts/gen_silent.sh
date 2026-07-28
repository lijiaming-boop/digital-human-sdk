#!/bin/bash
# Generate a 30s true-silent WAV for baseline comparison
set -e
ASSETS=/mnt/c/Users/27013/Desktop/digital-human-sdk/assets
ffmpeg -y -v error -f lavfi -i anullsrc=channel_layout=mono:sample_rate=16000 \
    -t 30 -c:a pcm_s16le "$ASSETS/silent_30s.wav"
echo "Generated silent_30s.wav"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$ASSETS/silent_30s.wav"
ls -la "$ASSETS/silent_30s.wav"
