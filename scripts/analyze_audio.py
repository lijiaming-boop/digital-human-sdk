"""Detect silent vs speech segments in zw_trimmed.mp3.
Window 200ms (5 frames at 25fps), RMS threshold = 0.02.
"""
import numpy as np
import subprocess, json, sys, os

ASSET = '/mnt/c/Users/27013/Desktop/digital-human-sdk/assets/zw_trimmed.mp3'
OUT = '/tmp/audio_segments.json'

# Decode to 16k mono PCM
proc = subprocess.run(
    ['ffmpeg', '-v', 'error', '-i', ASSET,
     '-f', 's16le', '-ac', '1', '-ar', '16000', '-'],
    capture_output=True)
pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
sr = 16000
duration = len(pcm) / sr
print(f'PCM samples: {len(pcm)}, duration: {duration:.2f}s')

# Window 200ms hop 200ms
win = int(0.2 * sr)
hop = win
times, rms = [], []
for i in range(0, len(pcm) - win, hop):
    chunk = pcm[i:i+win]
    r = float(np.sqrt(np.mean(chunk*chunk)))
    times.append(i / sr); rms.append(r)

rms = np.array(rms)
# Silence threshold: dynamic. Use 10th percentile.
thr = float(np.percentile(rms, 20) * 1.2)
print(f'Silence threshold: {thr:.4f}')

# Segments
silent_segs, speech_segs = [], []
i = 0
while i < len(times):
    is_silent = rms[i] < thr
    j = i
    while j < len(times) and (rms[j] < thr) == is_silent:
        j += 1
    seg = {
        'start': round(times[i], 2),
        'end': round(times[j-1] + 0.2, 2),
        'duration': round(times[j-1] + 0.2 - times[i], 2),
        'rms_mean': round(float(np.mean(rms[i:j])), 4),
        'type': 'silent' if is_silent else 'speech'
    }
    if seg['duration'] >= 1.0:
        if is_silent:
            silent_segs.append(seg)
        else:
            speech_segs.append(seg)
    i = j

# Pick the longest of each within first 60s, and one each in 30-60s range
def pick_in_range(segs, lo, hi, n=2):
    in_range = [s for s in segs if s['end'] > lo and s['start'] < hi]
    in_range.sort(key=lambda s: -s['duration'])
    return in_range[:n]

result = {
    'sr': sr, 'duration': round(duration, 2),
    'silence_threshold': round(thr, 4),
    'silent_segments_all': silent_segs,
    'speech_segments_all': speech_segs,
    'silent_pick_0_30s': pick_in_range(silent_segs, 0, 30, 2),
    'speech_pick_0_30s': pick_in_range(speech_segs, 0, 30, 2),
    'silent_pick_30_60s': pick_in_range(silent_segs, 30, 60, 2),
    'speech_pick_30_60s': pick_in_range(speech_segs, 30, 60, 2),
}
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2)
print('Silent segments >=1s:', len(silent_segs))
print('Speech segments >=1s:', len(speech_segs))
print('Total silent duration:', round(sum(s['duration'] for s in silent_segs), 1), 's')
print('Total speech duration:', round(sum(s['duration'] for s in speech_segs), 1), 's')
print('--- Silent picks (0-30s) ---')
for s in result['silent_pick_0_30s']: print(s)
print('--- Speech picks (0-30s) ---')
for s in result['speech_pick_0_30s']: print(s)
print('--- Silent picks (30-60s) ---')
for s in result['silent_pick_30_60s']: print(s)
print('--- Speech picks (30-60s) ---')
for s in result['speech_pick_30_60s']: print(s)
print('Saved', OUT)
