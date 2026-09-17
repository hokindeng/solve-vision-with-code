import cv2
import numpy as np
import imageio
import os

def solve():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path}")
        
    # Find colors and shapes
    colors = np.unique(img.reshape(-1, 3), axis=0)
    
    best_ratio = -1
    best_cnt = None
    
    for c in colors:
        # Assuming background is white. 
        if np.all(c == 255): 
            continue
            
        mask = cv2.inRange(img, c, c)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours: 
            continue
            
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        
        # Filter out tiny noise contours
        if area < 1000:
            continue
            
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            continue
            
        ratio = area / hull_area
        if ratio > best_ratio:
            best_ratio = ratio
            best_cnt = cnt

    if best_cnt is None:
        raise ValueError("No valid shapes found")

    N = len(best_cnt)
    frames = []
    total_frames = 40
    
    for i in range(total_frames):
        out = img.copy()
        
        # Determine how many points of the contour to draw
        k = int((i / (total_frames - 1)) * N)
        
        if k > 1:
            if k == N:
                cv2.polylines(out, [best_cnt], True, (0, 0, 255), 5, lineType=cv2.LINE_AA)
            else:
                cv2.polylines(out, [best_cnt[:k]], False, (0, 0, 255), 5, lineType=cv2.LINE_AA)
                
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        frames.append(out_rgb)
        
    imageio.mimwrite(out_path, frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
