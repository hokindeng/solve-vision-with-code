import numpy as np
from PIL import Image
import cv2
import os
import subprocess
from scipy.ndimage import label, binary_dilation

def main():
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/app/frames', exist_ok=True)

    img = np.array(Image.open('/app/first_frame.png'))
    # Determine background color by most frequent color
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    bg = colors[np.argmax(counts)]

    mask = ~np.all(img == bg, axis=-1)
    dilated_mask = binary_dilation(mask, iterations=10)
    labeled, num_features = label(dilated_mask)

    comps = []
    for i in range(1, num_features + 1):
        comp_mask = (labeled == i) & mask
        if not np.any(comp_mask):
            continue
        coords = np.argwhere(comp_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cy, cx = (y_min+y_max)/2.0, (x_min+x_max)/2.0
        
        area = (y_max - y_min + 1) * (x_max - x_min + 1)
        pixels = np.sum(comp_mask)
        density = pixels / area
        
        mean_color = img[comp_mask].mean(axis=0)
        
        comps.append({
            'mask': comp_mask,
            'cy': cy, 'cx': cx,
            'color': mean_color,
            'coords': coords,
            'pixels_val': img[comp_mask],
            'density': density
        })

    # Pair them by closest color
    pairs = []
    unpaired = list(comps)
    while len(unpaired) >= 2:
        c1 = unpaired.pop(0)
        best_dist = float('inf')
        best_idx = -1
        for i, c2 in enumerate(unpaired):
            dist = np.linalg.norm(c1['color'] - c2['color'])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx != -1:
            c2 = unpaired.pop(best_idx)
            
            if c1['density'] > c2['density']:
                obj, out = c1, c2
            else:
                obj, out = c2, c1
            pairs.append((obj, out))

    base_img = img.copy()
    for obj, _ in pairs:
        base_img[obj['mask']] = bg

    num_frames = 35

    for i in range(num_frames):
        t = i / (num_frames - 1)
        frame = base_img.copy()
        
        for obj, out in pairs:
            dy = t * (out['cy'] - obj['cy'])
            dx = t * (out['cx'] - obj['cx'])
            
            py = np.round(obj['coords'][:, 0] + dy).astype(int)
            px = np.round(obj['coords'][:, 1] + dx).astype(int)
            
            valid = (py >= 0) & (py < frame.shape[0]) & (px >= 0) & (px < frame.shape[1])
            frame[py[valid], px[valid]] = obj['pixels_val'][valid]
            
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
