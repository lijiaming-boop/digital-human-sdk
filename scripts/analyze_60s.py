"""Detailed analysis of 60s test:
- Per-5s segment correlation (mouth openness vs audio energy)
- Compare silent/low-energy segments vs speech/high-energy segments
- Compare zw_60s vs silent_30s baseline frame sets
"""
import csv, statistics as st, json, os, sys

zw_csv = '/home/hulushen/dh_60s/openness_energy.csv'
sil_csv = '/home/hulushen/dh_silent/openness_energy.csv'
zw_frames_dir = '/home/hulushen/dh_60s/frames'
sil_frames_dir = '/home/hulushen/dh_silent/frames'

# Load zw 60s data
zw_o, zw_e = [], []
with open(zw_csv) as f:
    for row in csv.DictReader(f):
        zw_o.append(float(row['openness']))
        zw_e.append(float(row['energy']))

# Load silent baseline
sil_o, sil_e = [], []
with open(sil_csv) as f:
    for row in csv.DictReader(f):
        sil_o.append(float(row['openness']))
        sil_e.append(float(row['energy']))

print('=== 60s zw_trimmed.mp3 ===')
print(f'  N = {len(zw_o)} frames')
print(f'  openness: min={min(zw_o):.2f} max={max(zw_o):.2f} mean={st.mean(zw_o):.2f} std={st.stdev(zw_o):.2f}')
print(f'  energy:   min={min(zw_e):.3f} max={max(zw_e):.3f} mean={st.mean(zw_e):.3f} std={st.stdev(zw_e):.3f}')

# Per-5s segments
print('\n=== Per-5s segment analysis (zw_60s) ===')
print(f'  {"segment":<10} {"o_mean":<8} {"o_std":<7} {"e_mean":<8} {"e_std":<7} {"r":<8}')
segs = []
for seg in range(12):
    a = seg * 125; b = a + 125
    if a >= len(zw_o): break
    if b > len(zw_o): b = len(zw_o)
    o3 = zw_o[a:b]; e3 = zw_e[a:b]
    L = len(o3)
    mo = sum(o3)/L; me = sum(e3)/L
    so = st.stdev(o3) if L > 1 else 0
    se = st.stdev(e3) if L > 1 else 0
    sxy = sum((o3[i]-mo)*(e3[i]-me) for i in range(L))
    sxx = sum((o3[i]-mo)**2 for i in range(L))
    syy = sum((e3[i]-me)**2 for i in range(L))
    r = sxy/((sxx*syy)**0.5) if sxx>0 and syy>0 else 0
    label = f'{seg*5}-{seg*5+5}s'
    print(f'  {label:<10} {mo:<8.2f} {so:<7.2f} {me:<8.3f} {se:<7.3f} {r:<+8.3f}')
    segs.append({'seg': seg, 'range': label, 'o_mean': round(mo,2),
                 'o_std': round(so,2), 'e_mean': round(me,3),
                 'e_std': round(se,3), 'r': round(r,3)})

# Aggregate: low-energy frames vs high-energy frames
print('\n=== Aggregate by energy level (zw_60s) ===')
e_threshold = st.median(zw_e)
low_o = [zw_o[i] for i in range(len(zw_o)) if zw_e[i] < e_threshold]
high_o = [zw_o[i] for i in range(len(zw_o)) if zw_e[i] >= e_threshold]
print(f'  Energy median = {e_threshold:.4f}')
print(f'  Low-energy  (n={len(low_o)}): openness mean={st.mean(low_o):.2f} std={st.stdev(low_o):.2f} min={min(low_o):.2f} max={max(low_o):.2f}')
print(f'  High-energy (n={len(high_o)}): openness mean={st.mean(high_o):.2f} std={st.stdev(high_o):.2f} min={min(high_o):.2f} max={max(high_o):.2f}')

# Silent baseline analysis
print('\n=== Silent baseline (30s silent_30s.wav) ===')
print(f'  N = {len(sil_o)} frames')
print(f'  openness: min={min(sil_o):.4f} max={max(sil_o):.4f} mean={st.mean(sil_o):.4f} std={st.stdev(sil_o) if len(sil_o)>1 else 0:.4f}')
print(f'  energy:   min={min(sil_e):.4f} max={max(sil_e):.4f}')
print(f'  >> Silent output is essentially STATIC (max-min = {max(sil_o)-min(sil_o):.4f})')

# Compare mean frame differences: silent should produce baseline mouth (closed)
# In zw_60s, low-energy frames should resemble silent baseline (mouth closed)
# In zw_60s, high-energy frames should show larger changes (mouth opening)
print('\n=== Comparison: silent baseline vs zw low/high energy ===')
print(f'  Silent baseline openness mean: {st.mean(sil_o):.4f}')
print(f'  zw low-energy openness mean:  {st.mean(low_o):.4f} (diff from silent = {st.mean(low_o)-st.mean(sil_o):+.2f})')
print(f'  zw high-energy openness mean: {st.mean(high_o):.4f} (diff from silent = {st.mean(high_o)-st.mean(sil_o):+.2f})')

# Direct frame comparison: take a few silent frames and a few high-energy frames
# Save sampled frames
import shutil
os.makedirs('/home/hulushen/dh_compare', exist_ok=True)
# Silent baseline samples
for i in [0, 100, 400, 749]:
    src = os.path.join(sil_frames_dir, f'f_{i:05d}.jpg')
    if os.path.exists(src):
        shutil.copy(src, f'/home/hulushen/dh_compare/silent_f{i:05d}.jpg')

# zw_60s low-energy samples (energy < median)
zw_low_idx = sorted([i for i in range(len(zw_e)) if zw_e[i] < e_threshold])
zw_high_idx = sorted([i for i in range(len(zw_e)) if zw_e[i] >= e_threshold], key=lambda i: -zw_e[i])
# Pick 4 evenly spaced from low and 4 from high
def pick_n(idxs, n):
    if len(idxs) <= n: return idxs
    step = len(idxs) / n
    return [idxs[int(i*step)] for i in range(n)]

for i in pick_n(zw_low_idx, 4):
    src = os.path.join(zw_frames_dir, f'f_{i:05d}.jpg')
    if os.path.exists(src):
        shutil.copy(src, f'/home/hulushen/dh_compare/zw_low_f{i:05d}.jpg')
for i in pick_n(zw_high_idx, 4):
    src = os.path.join(zw_frames_dir, f'f_{i:05d}.jpg')
    if os.path.exists(src):
        shutil.copy(src, f'/home/hulushen/dh_compare/zw_high_f{i:05d}.jpg')

print('\nSaved comparison frames to /home/hulushen/dh_compare/')
print(os.listdir('/home/hulushen/dh_compare/'))

# Save JSON
result = {
    'zw_60s': {
        'n': len(zw_o),
        'openness': {'min': min(zw_o), 'max': max(zw_o), 'mean': st.mean(zw_o), 'std': st.stdev(zw_o)},
        'energy': {'min': min(zw_e), 'max': max(zw_e), 'mean': st.mean(zw_e), 'std': st.stdev(zw_e)},
        'segments': segs,
        'low_energy_openness': {'n': len(low_o), 'mean': st.mean(low_o), 'std': st.stdev(low_o), 'min': min(low_o), 'max': max(low_o)},
        'high_energy_openness': {'n': len(high_o), 'mean': st.mean(high_o), 'std': st.stdev(high_o), 'min': min(high_o), 'max': max(high_o)},
    },
    'silent_30s': {
        'n': len(sil_o),
        'openness': {'min': min(sil_o), 'max': max(sil_o), 'mean': st.mean(sil_o), 'std': st.stdev(sil_o) if len(sil_o)>1 else 0},
    }
}
with open('/tmp/compare_result.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print('\nSaved /tmp/compare_result.json')
