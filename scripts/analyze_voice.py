"""Analyze voice_30s_16k_mono.wav: RMS profile, segment by segment."""
import numpy as np
import subprocess, json

ASSET = '/mnt/c/Users/27013/Desktop/digital-human-sdk/assets/voice_30s_16k_mono.wav'

proc = subprocess.run(
    ['ffmpeg', '-v', 'error', '-i', ASSET,
     '-f', 's16le', '-ac', '1', '-ar', '16000', '-'],
    capture_output=True)
pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
sr = 16000
duration = len(pcm) / sr
print(f'PCM samples: {len(pcm)}, duration: {duration:.2f}s')

# Window 200ms hop 200ms
win = int(0.2 * sr); hop = win
times, rms = [], []
for i in range(0, len(pcm) - win, hop):
    chunk = pcm[i:i+win]
    r = float(np.sqrt(np.mean(chunk*chunk)))
    times.append(i / sr); rms.append(r)
rms = np.array(rms)

print(f'RMS: min={rms.min():.4f} max={rms.max():.4f} mean={rms.mean():.4f} median={np.median(rms):.4f}')
print(f'Silence ratio (RMS<0.01): {(rms<0.01).mean()*100:.1f}%')
print(f'Silence ratio (RMS<0.02): {(rms<0.02).mean()*100:.1f}%')

# Per-5s segment RMS
print('\nPer-5s segment RMS:')
for seg in range(int(duration/5)+1):
    a = seg*5; b = a+5
    if a >= duration: break
    if b > duration: b = duration
    a_idx = int(a*5); b_idx = int(b*5)  # 5 segments per second (200ms window)
    if b_idx > len(rms): b_idx = len(rms)
    if a_idx >= len(rms): break
    seg_rms = rms[a_idx:b_idx]
    print(f'  {a}-{b}s: rms mean={seg_rms.mean():.4f} min={seg_rms.min():.4f} max={seg_rms.max():.4f}')

# Save downsampled time series
ds_t, ds_rms = [], []
for i in range(0, len(times), 3):  # every 600ms
    ds_t.append(round(times[i], 2))
    ds_rms.append(round(float(rms[i]), 4))

with open('/tmp/voice_rms.json', 'w') as f:
    json.dump({'t': ds_t, 'rms': ds_rms, 'duration': round(duration, 2),
               'rms_mean': float(rms.mean()), 'rms_max': float(rms.max())}, f)
print('\nSaved /tmp/voice_rms.json')
