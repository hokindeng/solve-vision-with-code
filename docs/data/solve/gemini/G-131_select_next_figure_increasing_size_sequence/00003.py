import cv2
import numpy as np
import imageio

img = cv2.imread('/app/first_frame.png')
fps = 16
total_frames = 60
frames = []

def get_arrow_frame(base_img, pt1, pt2, progress, text):
    frame = base_img.copy()
    if progress <= 0: return frame
    
    x1, y1 = pt1
    x2, y2 = pt2
    
    cur_x = int(x1 + (x2 - x1) * progress)
    
    if cur_x > x1:
        length = cur_x - x1
        tip = min(0.5, 15.0 / max(1, length))
        cv2.arrowedLine(frame, (x1, y1), (cur_x, y1), (0, 0, 0), 3, tipLength=tip)
        
    frame_with_text = frame.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = x1 + (x2 - x1) // 2 - tw // 2
    ty = y1 - 15
    cv2.putText(frame_with_text, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[:, :cur_x] = True
    frame[mask] = frame_with_text[mask]
    
    return frame

base = img.copy()

for f in range(total_frames):
    frame = base.copy()
    
    # Phase 1: Arrow 1 (Frames 0-11)
    if f < 12:
        p = f / 11.0
        frame = get_arrow_frame(frame, (267, 337), (387, 337), p, "+ size step")
    else:
        frame = get_arrow_frame(frame, (267, 337), (387, 337), 1.0, "+ size step")
        
    # Phase 2: Arrow 2 (Frames 12-23)
    if 12 <= f < 24:
        p = (f - 12) / 11.0
        frame = get_arrow_frame(frame, (456, 337), (559, 337), p, "+ size step")
    elif f >= 24:
        frame = get_arrow_frame(frame, (456, 337), (559, 337), 1.0, "+ size step")
        
    # Phase 3: Arrow 3 (Frames 24-35)
    if 24 <= f < 36:
        p = (f - 24) / 11.0
        frame = get_arrow_frame(frame, (646, 337), (723, 337), p, "+ size step")
    elif f >= 36:
        frame = get_arrow_frame(frame, (646, 337), (723, 337), 1.0, "+ size step")
        
    # Phase 4: Circle (Frames 36-59)
    if f >= 36:
        p = min(1.0, (f - 36) / 11.0)
        angle = int(360 * p)
        if angle > 0:
            cv2.ellipse(frame, (856, 854), (95, 95), 0, 0, angle, (0, 0, 255), 5)
            
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

import os
os.makedirs('/app/output', exist_ok=True)
imageio.mimwrite('/app/output/video.mp4', frames, fps=fps, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
