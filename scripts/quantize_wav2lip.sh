#!/usr/bin/env bash
# Build FP16, INT8 and optional mixed-INT8 NCNN variants from one immutable FP32 model.
# Usage:
#   scripts/quantize_wav2lip.sh <fp32-param> <fp32-bin> <calibration-dir> <output-dir> [ncnn-tools-dir]
# calibration-dir must contain audio.list + face.list emitted by pipeline_lipsync_test.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 <fp32-param> <fp32-bin> <calibration-dir> <output-dir> [ncnn-tools-dir]" >&2
  exit 64
fi

SOURCE_PARAM=$1
SOURCE_BIN=$2
CALIB_DIR=$3
OUT_DIR=$4
TOOLS_DIR=${5:-}

tool() {
  if [[ -n "$TOOLS_DIR" ]]; then
    printf '%s/%s' "$TOOLS_DIR" "$1"
  else
    command -v "$1"
  fi
}

NCNN_OPTIMIZE=$(tool ncnnoptimize)
NCNN_TABLE=$(tool ncnn2table)
NCNN_INT8=$(tool ncnn2int8)

for file in "$SOURCE_PARAM" "$SOURCE_BIN" "$CALIB_DIR/audio.list" "$CALIB_DIR/face.list"; do
  [[ -f "$file" ]] || { echo "missing required file: $file" >&2; exit 66; }
done
for program in "$NCNN_OPTIMIZE" "$NCNN_TABLE" "$NCNN_INT8"; do
  [[ -x "$program" ]] || { echo "NCNN tool is unavailable: $program" >&2; exit 69; }
done

audio_count=$(wc -l < "$CALIB_DIR/audio.list")
face_count=$(wc -l < "$CALIB_DIR/face.list")
if [[ "$audio_count" -lt 100 || "$audio_count" -ne "$face_count" ]]; then
  echo "need at least 100 paired real samples; audio=$audio_count face=$face_count" >&2
  exit 65
fi

mkdir -p "$OUT_DIR"/{fp16,int8,int8-mixed}

# FP16 storage variant. Runtime still probes actual device capability and may retain FP32 arithmetic.
"$NCNN_OPTIMIZE" "$SOURCE_PARAM" "$SOURCE_BIN" \
  "$OUT_DIR/fp16/model.param" "$OUT_DIR/fp16/model.bin" 65536

# Keep a separately optimized FP32 graph as the sole source of the INT8 conversion.
"$NCNN_OPTIMIZE" "$SOURCE_PARAM" "$SOURCE_BIN" \
  "$OUT_DIR/int8/model-opt.param" "$OUT_DIR/int8/model-opt.bin" 0

"$NCNN_TABLE" "$OUT_DIR/int8/model-opt.param" "$OUT_DIR/int8/model-opt.bin" \
  "$CALIB_DIR/audio.list,$CALIB_DIR/face.list" "$OUT_DIR/int8/calibration.table" \
  shape=[16,80,1],[96,96,6] type=1 thread=4 method=kl

"$NCNN_INT8" "$OUT_DIR/int8/model-opt.param" "$OUT_DIR/int8/model-opt.bin" \
  "$OUT_DIR/int8/model.param" "$OUT_DIR/int8/model.bin" "$OUT_DIR/int8/calibration.table"

# Optional mixed precision: set MIXED_EXCLUDE to a comma-separated list of table-key prefixes
# after evidence from the quality regression identifies sensitive layers. Example:
# MIXED_EXCLUDE='/face_decoder/0/Conv,/output/Conv' scripts/quantize_wav2lip.sh ...
cp "$OUT_DIR/int8/model-opt.param" "$OUT_DIR/int8-mixed/model-opt.param"
cp "$OUT_DIR/int8/model-opt.bin" "$OUT_DIR/int8-mixed/model-opt.bin"
cp "$OUT_DIR/int8/calibration.table" "$OUT_DIR/int8-mixed/calibration.table"
if [[ -n "${MIXED_EXCLUDE:-}" ]]; then
  IFS=',' read -r -a excluded <<< "$MIXED_EXCLUDE"
  for prefix in "${excluded[@]}"; do
    # A commented scale line remains FP32 in ncnn2int8 mixed-precision conversion.
    sed -i -E "s|^(${prefix//\//\\/}[^[:space:]]*[[:space:]])|#\1|" \
      "$OUT_DIR/int8-mixed/calibration.table"
  done
  "$NCNN_INT8" "$OUT_DIR/int8-mixed/model-opt.param" "$OUT_DIR/int8-mixed/model-opt.bin" \
    "$OUT_DIR/int8-mixed/model.param" "$OUT_DIR/int8-mixed/model.bin" \
    "$OUT_DIR/int8-mixed/calibration.table"
else
  rm -f "$OUT_DIR/int8-mixed/model-opt.param" "$OUT_DIR/int8-mixed/model-opt.bin" \
        "$OUT_DIR/int8-mixed/calibration.table"
  rmdir "$OUT_DIR/int8-mixed"
fi

for variant in fp16 int8 int8-mixed; do
  [[ -f "$OUT_DIR/$variant/model.param" ]] || continue
  {
    echo "model_variant=$variant"
    echo "source_param_sha256=$(sha256sum "$SOURCE_PARAM" | awk '{print $1}')"
    echo "param_sha256=$(sha256sum "$OUT_DIR/$variant/model.param" | awk '{print $1}')"
    echo "bin_sha256=$(sha256sum "$OUT_DIR/$variant/model.bin" | awk '{print $1}')"
    echo "input_audio=audio_sequences:[16,80,1]"
    echo "input_face=face_sequences:[96,96,6]"
    echo "output=output"
  } > "$OUT_DIR/$variant/manifest.env"
done

echo "Created variants in: $OUT_DIR"
