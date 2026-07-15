#!/usr/bin/env python3
"""Generate test audio WAV and raw PCM files for pipeline testing."""
import wave
import struct
import math
import random
import os

sr = 16000
dur = 3.0
n = int(sr * dur)

samples = []
for i in range(n):
    t = i / sr
    s1 = 0.3 * math.sin(2 * math.pi * 440 * t)
    s2 = 0.15 * math.sin(2 * math.pi * 880 * t)
    if t < 2.0:
        freq = 200 + 300 * t
        s3 = 0.2 * math.sin(2 * math.pi * freq * t)
    else:
        s3 = 0
    noise = 0.01 * (random.random() - 0.5)
    samples.append(s1 + s2 + s3 + noise)

max_val = max(abs(max(samples)), abs(min(samples)), 0.01)
samples_norm = [s / max_val * 0.8 for s in samples]
samples_int = [int(s * 32767) for s in samples_norm]

out_dir = os.path.dirname(__file__)

# WAV
wav_path = os.path.join(out_dir, "test_16k_mono.wav")
packed = struct.pack("<" + "h" * len(samples_int), *samples_int)
with wave.open(wav_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(packed)
print(f"WAV: {wav_path} ({os.path.getsize(wav_path)} bytes)")

# Raw PCM (s16le)
raw_path = os.path.join(out_dir, "test_16k_mono.raw")
with open(raw_path, "wb") as f:
    f.write(packed)
print(f"RAW: {raw_path} ({os.path.getsize(raw_path)} bytes)")

print(f"Duration: {dur}s, SR: {sr}Hz, Samples: {n}")
