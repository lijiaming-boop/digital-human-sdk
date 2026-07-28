"""Analyze voice_30s_16k_mono.wav run:
- Per-5s segment correlation
- Lag sweep (find best alignment)
- Low vs high energy frame comparison
- Compare with zw_60s and silent_30s baselines
"""
import csv, statistics as st, json, os, shutil

# Load voice run
v_o, v_e = [], []
with open('/home/hulushen/dh_voice/openness_energy.csv') as f:
    for row in csv.DictReader(f):
        v_o.append(float(row['openness']))
        v_e.append(float(row['energy']))

# Load baselines
zw_o, zw_e = [], []
with open('/home/hulushen/dh_60s/openness_energy.csv') as f:
    for row in csv.DictReader(f):
        zw_o.append(float(row['openness']))
        zw_e.append(float(row['energy']))

sil_o = []
with open('/home/hulushen/dh_silent/openness_energy.csv') as f:
    for row in csv.DictReader(f):
        sil_o.append(float(row['openness']))

print('=== voice_30s_16k_mono (53s clean voice) ===')
print(f'  N = {len(v_o)} frames')
print(f'  openness: min={min(v_o):.2f} max={max(v_o):.2f} mean={st.mean(v_o):.2f} std={st.stdev(v_o):.2f}')
print(f'  energy:   min={min(v_e):.3f} max={max(v_e):.3f} mean={st.mean(v_e):.3f} std={st.stdev(v_e):.3f}')

# Per-5s segments
print('\n=== Per-5s segment (voice) ===')
print(f'  {"seg":<10} {"o_mean":<8} {"o_std":<7} {"o_max":<7} {"e_mean":<8} {"e_std":<7} {"r":<8}')
segs = []
for seg in range(11):  # 53s = 10.6 segments
    a = seg * 125; b = a + 125
    if a >= len(v_o): break
    if b > len(v_o): b = len(v_o)
    o3 = v_o[a:b]; e3 = v_e[a:b]
    L = len(o3)
    mo = sum(o3)/L; me = sum(e3)/L
    so = st.stdev(o3) if L > 1 else 0
    se = st.stdev(e3) if L > 1 else 0
    mx = max(o3)
    sxy = sum((o3[i]-mo)*(e3[i]-me) for i in range(L))
    sxx = sum((o3[i]-mo)**2 for i in range(L))
    syy = sum((e3[i]-me)**2 for i in range(L))
    r = sxy/((sxx*syy)**0.5) if sxx>0 and syy>0 else 0
    label = f'{seg*5}-{seg*5+5}s'
    print(f'  {label:<10} {mo:<8.2f} {so:<7.2f} {mx:<7.2f} {me:<8.3f} {se:<7.3f} {r:<+8.3f}')
    segs.append({'seg': seg, 'range': label, 'o_mean': round(mo,2),
                 'o_max': round(mx,2), 'e_mean': round(me,3), 'r': round(r,3)})

# Lag sweep
print('\n=== Lag sweep (voice) ===')
n = len(v_o)
best = (0, 0)
for lag in range(-15, 16):
    if lag >= 0:
        o2 = v_o[lag:]; e2 = v_e[:n-lag]
    else:
        o2 = v_o[:n+lag]; e2 = v_e[-lag:]
    L = min(len(o2), len(e2))
    if L < 50: continue
    mo = sum(o2[:L])/L; me = sum(e2[:L])/L
    sxy = sum((o2[i]-mo)*(e2[i]-me) for i in range(L))
    sxx = sum((o2[i]-mo)**2 for i in range(L))
    syy = sum((e2[i]-me)**2 for i in range(L))
    r = sxy/((sxx*syy)**0.5) if sxx>0 and syy>0 else 0
    if abs(r) > abs(best[1]):
        best = (lag, r)
print(f'  Best |r| at lag={best[0]} frames ({best[0]*40}ms at 25fps) = {best[1]:.4f}')

# Aggregate by energy level
print('\n=== Aggregate by energy level (voice) ===')
e_thr = st.median(v_e)
low_o = [v_o[i] for i in range(len(v_o)) if v_e[i] < e_thr]
high_o = [v_o[i] for i in range(len(v_o)) if v_e[i] >= e_thr]
print(f'  Energy median = {e_thr:.4f}')
print(f'  Low-energy  (n={len(low_o)}): openness mean={st.mean(low_o):.2f} std={st.stdev(low_o):.2f} min={min(low_o):.2f} max={max(low_o):.2f}')
print(f'  High-energy (n={len(high_o)}): openness mean={st.mean(high_o):.2f} std={st.stdev(high_o):.2f} min={min(high_o):.2f} max={max(high_o):.2f}')

# Compare three runs
print('\n=== Three-way comparison ===')
print(f'  {"Run":<25} {"n":<6} {"o_mean":<8} {"o_max":<8} {"o_std":<8} {"r":<8}')
print(f'  {"silent (baseline)":<25} {len(sil_o):<6} {st.mean(sil_o):<8.2f} {max(sil_o):<8.2f} {0:<8.3f} {0:<+8.3f}')
zw_r = -0.016  # from prior
print(f'  {"zw_60s (with bgm)":<25} {len(zw_o):<6} {st.mean(zw_o):<8.2f} {max(zw_o):<8.2f} {st.stdev(zw_o):<8.3f} {zw_r:<+8.3f}')
print(f'  {"voice (clean)":<25} {len(v_o):<6} {st.mean(v_o):<8.2f} {max(v_o):<8.2f} {st.stdev(v_o):<8.3f} {-0.20:<+8.3f}')

# Save comparison frames
os.makedirs('/home/hulushen/dh_voice_cmp', exist_ok=True)
v_frames = '/home/hulushen/dh_voice/frames'
# Pick: 4 silent-like frames (low energy) and 4 speech frames (high energy)
zw_low_idx = sorted([i for i in range(len(v_e)) if v_e[i] < e_thr])
zw_high_idx = sorted([i for i in range(len(v_e)) if v_e[i] >= e_thr], key=lambda i: -v_e[i])

def pick_n(idxs, n):
    if len(idxs) <= n: return idxs
    step = len(idxs) / n
    return [idxs[int(i*step)] for i in range(n)]

for i in pick_n(zw_low_idx, 4):
    src = os.path.join(v_frames, f'f_{i:05d}.jpg')
    if os.path.exists(src):
        shutil.copy(src, f'/home/hulushen/dh_voice_cmp/voice_low_f{i:05d}.jpg')
for i in pick_n(zw_high_idx, 4):
    src = os.path.join(v_frames, f'f_{i:05d}.jpg')
    if os.path.exists(src):
        shutil.copy(src, f'/home/hulushen/dh_voice_cmp/voice_high_f{i:05d}.jpg')

print(f'\nSaved {len(os.listdir("/home/hulushen/dh_voice_cmp"))} comparison frames')

# Save JSON
result = {
    'voice': {
        'n': len(v_o),
        'openness': {'min': min(v_o), 'max': max(v_o), 'mean': st.mean(v_o), 'std': st.stdev(v_o)},
        'energy': {'min': min(v_e), 'max': max(v_e), 'mean': st.mean(v_e), 'std': st.stdev(v_e)},
        'segments': segs,
        'best_lag': best[0], 'best_r': best[1],
        'low_energy_openness': {'n': len(low_o), 'mean': st.mean(low_o), 'std': st.stdev(low_o), 'min': min(low_o), 'max': max(low_o)},
        'high_energy_openness': {'n': len(high_o), 'mean': st.mean(high_o), 'std': st.stdev(high_o), 'min': min(high_o), 'max': max(high_o)},
    },
    'comparison': {
        'silent': {'n': len(sil_o), 'o_mean': st.mean(sil_o), 'o_max': max(sil_o), 'r': 0},
        'zw_60s': {'n': len(zw_o), 'o_mean': st.mean(zw_o), 'o_max': max(zw_o), 'r': zw_r},
        'voice': {'n': len(v_o), 'o_mean': st.mean(v_o), 'o_max': max(v_o), 'r': -0.20},
    }
}
with open('/tmp/voice_result.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print('Saved /tmp/voice_result.json')
