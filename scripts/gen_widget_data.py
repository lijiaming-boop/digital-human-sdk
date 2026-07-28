import csv, statistics as st, json

o,e=[],[]
with open('/home/hulushen/dh_lipsync_run/openness_energy.csv') as f:
    r=csv.DictReader(f)
    for row in r:
        o.append(float(row['openness'])); e.append(float(row['energy']))

n=len(o)
# Downsample to ~60 points (every 12 frames, 0.48s)
ds_o, ds_e, ds_t = [], [], []
for i in range(0, n, 12):
    j = min(i+12, n)
    ds_o.append(sum(o[i:j])/len(o[i:j]))
    ds_e.append(sum(e[i:j])/len(e[i:j]))
    ds_t.append(round(i/25.0, 2))

# Normalize energy to openness range for comparison line
e_min, e_max = min(ds_e), max(ds_e)
o_min, o_max = min(ds_o), max(ds_o)
ds_e_scaled = [round(o_min + (v-e_min)/(e_max-e_min)*(o_max-o_min), 2) for v in ds_e]

# Per-segment
segs = []
for seg in range(6):
    a=seg*125; b=a+125
    if b>n: b=n
    if a>=n: break
    o3=o[a:b]; e3=e[a:b]
    L=len(o3)
    mo=sum(o3)/L; me=sum(e3)/L
    sxy=sum((o3[i]-mo)*(e3[i]-me) for i in range(L))
    sxx=sum((o3[i]-mo)**2 for i in range(L)); syy=sum((e3[i]-me)**2 for i in range(L))
    r=sxy/((sxx*syy)**0.5) if sxx>0 and syy>0 else 0
    segs.append({'seg': seg, 'range': f'{seg*5}-{seg*5+5}s', 'r': round(r, 3), 'o': round(mo, 2), 'e': round(me, 3)})

result = {
    'time_labels': ds_t,
    'openness': [round(v, 2) for v in ds_o],
    'energy_scaled': ds_e_scaled,
    'segments': segs,
    'metrics': {
        'frames_in': 750, 'frames_out': 750, 'dropped': 0,
        'content_fps': 25.0, 'target_fps': 25,
        'speed_x': 1.05, 'wall_s': 28.53, 'content_s': 30,
        'avg_inference_ms': 35.29, 'avg_render_ms': 17.28,
        'avg_audio_ms': 0.062,
        'openness_mean': 7.42, 'openness_max': 10.48, 'openness_min': 4.70,
        'r_frame': -0.181, 'r_smoothed': -0.206,
        'best_lag_frames': -2, 'best_lag_ms': -80, 'best_r': -0.226,
    }
}
print(json.dumps(result))
