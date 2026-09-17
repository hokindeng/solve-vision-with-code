import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Find the circular region
    black_mask = np.all(img == [0, 0, 0], axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    region_center = None
    region_radius = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100000 < area < 150000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circ = 4 * np.pi * area / (perimeter * perimeter)
                if circ > 0.8:  # It's a circle
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    region_center = (int(x), int(y))
                    region_radius = int(r)
                    break
    
    if region_center is None:
        # Fallback if detection fails
        region_center = (755, 512)
        region_radius = 196
        
    # 2. Find colored objects
    mask = (~np.all(img == [255, 255, 255], axis=-1)) & (~np.all(img == [0, 0, 0], axis=-1))
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    target_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circ = 4 * np.pi * area / (perimeter * perimeter)
        if circ > 0.8: # It's a circle
            (x, y), r = cv2.minEnclosingCircle(cnt)
            dist = np.sqrt((x - region_center[0])**2 + (y - region_center[1])**2)
            if dist < region_radius:
                target_contours.append(cnt)
                
    # 3. Generate video
    num_frames = 40
    fps = 16
    out_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    for i in range(num_frames):
        progress = i / (num_frames - 1) if num_frames > 1 else 1.0
        frame = img.copy()
        
        for cnt in target_contours:
            k = int(len(cnt) * progress)
            if k > 1:
                is_closed = (k == len(cnt))
                cv2.polylines(frame, [cnt[:k]], isClosed=is_closed, color=(0, 255, 0), thickness=4)
                
        # convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
