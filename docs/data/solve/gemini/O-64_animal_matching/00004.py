import numpy as np
from PIL import Image
import cv2
import imageio
import os

def generate_video():
    img = np.array(Image.open('/app/first_frame.png').convert('RGB'))
    bg_color = img[0, 0].copy()
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    left_comps = []
    right_comps = []
    for i in range(1, num_labels):
        if stats[i][2] < 10 or stats[i][3] < 10:
            continue
        if stats[i][0] < 500:
            left_comps.append(i)
        elif stats[i][0] > 512:
            right_comps.append(i)

    matches = []
    for lc in left_comps:
        lw, lh = stats[lc][2], stats[lc][3]
        best_rc = None
        best_diff = float('inf')
        for rc in right_comps:
            rw, rh = stats[rc][2], stats[rc][3]
            diff = abs(lw - rw) + abs(lh - rh)
            if diff < best_diff:
                best_diff = diff
                best_rc = rc
        matches.append((lc, best_rc))

    base_img = img.copy()
    for lc, _ in matches:
        l_mask = (labels == lc)
        base_img[l_mask] = bg_color

    frames = []
    num_frames = 64

    for f in range(num_frames):
        t = f / (num_frames - 1)
        frame_img = base_img.copy()
        
        for lc, rc in matches:
            dx = stats[rc][0] - stats[lc][0]
            dy = stats[rc][1] - stats[lc][1]
            
            curr_dx = int(round(t * dx))
            curr_dy = int(round(t * dy))
            
            l_mask = (labels == lc)
            pixels = img[l_mask]
            
            y_indices, x_indices = np.where(l_mask)
            new_y = y_indices + curr_dy
            new_x = x_indices + curr_dx
            
            valid = (new_y >= 0) & (new_y < img.shape[0]) & (new_x >= 0) & (new_x < img.shape[1])
            
            frame_img[new_y[valid], new_x[valid]] = pixels[valid]
            
        frames.append(frame_img)

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
