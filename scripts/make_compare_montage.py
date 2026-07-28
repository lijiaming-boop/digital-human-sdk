"""Make comparison montage: silent baseline + zw low-energy + zw high-energy frames."""
import cv2
import os
import numpy as np

CMP_DIR = '/home/hulushen/dh_compare'

# Layout: 3 rows x 4 cols
# Row 0: silent baseline (4 frames at t=0, 4s, 16s, 30s)
# Row 1: zw low-energy (4 frames)
# Row 2: zw high-energy (4 frames)
sil_files = sorted([f for f in os.listdir(CMP_DIR) if f.startswith('silent_')])
low_files = sorted([f for f in os.listdir(CMP_DIR) if f.startswith('zw_low_')])
high_files = sorted([f for f in os.listdir(CMP_DIR) if f.startswith('zw_high_')])

rows = [sil_files, low_files, high_files]
row_labels = ['silent baseline (RMS=0)', 'zw low-energy (RMS<0.17, music only)', 'zw high-energy (RMS>=0.17, speech+music)']

# Get frame size
img = cv2.imread(os.path.join(CMP_DIR, sil_files[0]))
H, W = img.shape[:2]
scale = 320.0 / W
cw = int(W * scale); ch = int(H * scale)
labh = 28
pad = 6
rows_n = 3; cols = 4
grid = np.zeros((rows_n*(ch+labh+pad)+pad, cols*(cw+pad)+pad, 3), dtype=np.uint8)

for ri, (files, label) in enumerate(zip(rows, row_labels)):
    for ci, fname in enumerate(files[:4]):
        img = cv2.imread(os.path.join(CMP_DIR, fname))
        img = cv2.resize(img, (cw, ch))
        # Extract frame index from filename
        idx = int(fname.split('_f')[1].split('.')[0])
        t = idx / 25.0
        cell = np.zeros((ch+labh, cw, 3), dtype=np.uint8)
        cell[labh:, :] = img
        cv2.putText(cell, f'f{idx} t={t:.1f}s', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y0 = pad + ri*(ch+labh+pad)
        x0 = pad + ci*(cw+pad)
        grid[y0:y0+ch+labh, x0:x0+cw] = cell
    # Row label on the leftmost column (overlay on first cell)
    # Actually just add label text below the grid later

out = '/home/hulushen/dh_compare/montage_compare.png'
cv2.imwrite(out, grid, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print('Saved', out, 'shape:', grid.shape)
