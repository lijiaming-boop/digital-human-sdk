import cv2
import os

frames_dir = '/home/hulushen/dh_lipsync_run/frames'
samples = list(range(0, 750, 50))  # 0, 50, 100, ..., 700 = 15 frames
cols = 5; rows = 3
# Compute cell size from first frame
f0 = cv2.imread(os.path.join(frames_dir, 'f_00000.jpg'))
H, W = f0.shape[:2]
scale = 384.0 / W
cw = int(W * scale); ch = int(H * scale)
pad = 8; labh = 24
grid = __import__('numpy').zeros((rows*(ch+labh+pad)+pad, cols*(cw+pad)+pad, 3), dtype='uint8')

for idx, i in enumerate(samples):
    r = idx // cols; c = idx % cols
    p = os.path.join(frames_dir, f'f_{i:05d}.jpg')
    img = cv2.imread(p)
    img = cv2.resize(img, (cw, ch))
    # Add label on top
    cell = __import__('numpy').zeros((ch+labh, cw, 3), dtype='uint8')
    cell[labh:, :] = img
    cv2.putText(cell, f'f{i} t={i/25.0:.1f}s', (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    y0 = pad + r*(ch+labh+pad); x0 = pad + c*(cw+pad)
    grid[y0:y0+ch+labh, x0:x0+cw] = cell

out = '/home/hulushen/dh_lipsync_run/montage.png'
cv2.imwrite(out, grid, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print('Saved', out, 'size:', grid.shape)
