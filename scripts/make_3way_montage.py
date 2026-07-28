"""Make 3-way comparison montage: silent / zw low-energy / voice low-energy / voice high-energy."""
import cv2, os, numpy as np

base = '/home/hulushen'
cells = [
    # (path, label)
    (f'{base}/dh_silent/frames/f_00000.jpg',  'silent t=0s'),
    (f'{base}/dh_silent/frames/f_00400.jpg',  'silent t=16s'),
    (f'{base}/dh_silent/frames/f_00749.jpg',  'silent t=30s'),
    (f'{base}/dh_60s/frames/f_00455.jpg',     'zw low t=18s'),
    (f'{base}/dh_60s/frames/f_01271.jpg',     'zw high t=51s'),
    (f'{base}/dh_voice/frames/f_00266.jpg',   'voice low t=11s'),
    (f'{base}/dh_voice/frames/f_00642.jpg',   'voice low t=26s'),
    (f'{base}/dh_voice/frames/f_00880.jpg',   'voice high t=35s'),
]

# Compute size
img = cv2.imread(cells[0][0])
H, W = img.shape[:2]
scale = 280.0 / W
cw = int(W * scale); ch = int(H * scale)
labh = 30
pad = 6
cols = 4; rows = 2
grid = np.zeros((rows*(ch+labh+pad)+pad, cols*(cw+pad)+pad, 3), dtype=np.uint8)

for i, (path, label) in enumerate(cells):
    r = i // cols; c = i % cols
    img = cv2.imread(path)
    if img is None: continue
    img = cv2.resize(img, (cw, ch))
    cell = np.zeros((ch+labh, cw, 3), dtype=np.uint8)
    cell[labh:, :] = img
    cv2.putText(cell, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    y0 = pad + r*(ch+labh+pad); x0 = pad + c*(cw+pad)
    grid[y0:y0+ch+labh, x0:x0+cw] = cell

out = '/home/hulushen/dh_voice_cmp/montage_3way.png'
cv2.imwrite(out, grid, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print('Saved', out, 'shape:', grid.shape)
