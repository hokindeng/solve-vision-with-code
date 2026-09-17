import cv2
import numpy as np
import imageio
import os

def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )

def solve():
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    color_main = [255, 69, 0]
    color_out = [175, 0, 0]
    
    plat_main = [138, 43, 226]
    plat_out = [78, 0, 166]
    
    ball_mask = ((img == color_main).all(axis=2) | (img == color_out).all(axis=2)).astype(np.uint8)
    
    ball_crop = img[619:700, 376:457].copy()
    ball_mask_crop = ball_mask[619:700, 376:457].copy()
    
    clean_bg = img.copy()
    clean_bg[ball_mask == 1] = [255, 255, 255]
    
    pts = np.array([
        [421 + 16, 592 + 0],
        [421 + 43, 592 + 17],
        [421 + 27, 592 + 42],
        [421 + 0, 592 + 25]
    ], np.int32)
    
    cv2.fillPoly(clean_bg, [pts], plat_main)
    cv2.polylines(clean_bg, [pts], isClosed=True, color=plat_out, thickness=3)
    
    bg = img.copy()
    bg[ball_mask == 1] = clean_bg[ball_mask == 1]
    
    pts_arr = np.array([
        [416.0, 659.0],
        [442.7, 612.5],
        [473.6, 574.6],
        [508.0, 538.5],
        [540.1, 501.0],
        [565.9, 459.1],
        [585.0, 415.0],
        [602.1, 371.0]
    ])
    
    fine_curve = []
    for i in range(len(pts_arr) - 1):
        p0 = pts_arr[max(0, i - 1)]
        p1 = pts_arr[i]
        p2 = pts_arr[i + 1]
        p3 = pts_arr[min(len(pts_arr) - 1, i + 2)]
        for t in np.linspace(0, 1, 100, endpoint=False):
            fine_curve.append(catmull_rom(p0, p1, p2, p3, t))
    fine_curve.append(pts_arr[-1])
    fine_curve = np.array(fine_curve)
    
    dists = np.zeros(len(fine_curve))
    for i in range(1, len(fine_curve)):
        dists[i] = dists[i-1] + np.linalg.norm(fine_curve[i] - fine_curve[i-1])
        
    total_dist = dists[-1]
    
    frames = 64
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for f in range(frames):
        t = f / (frames - 1)
        eased_t = t * (2 - t)
        target_dist = eased_t * total_dist
        
        idx = np.searchsorted(dists, target_dist)
        if idx == 0:
            pos = fine_curve[0]
        elif idx >= len(dists):
            pos = fine_curve[-1]
        else:
            d0 = dists[idx-1]
            d1 = dists[idx]
            ratio = (target_dist - d0) / (d1 - d0)
            pos = fine_curve[idx-1] + ratio * (fine_curve[idx] - fine_curve[idx-1])
            
        frame = bg.copy()
        
        cx, cy = pos
        x = int(round(cx - 40))
        y = int(round(cy - 40))
        
        y1, y2 = y, y + ball_crop.shape[0]
        x1, x2 = x, x + ball_crop.shape[1]
        
        frame_roi = frame[y1:y2, x1:x2]
        frame_roi[ball_mask_crop == 1] = ball_crop[ball_mask_crop == 1]
                    
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
