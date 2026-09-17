import cv2
import numpy as np
import imageio
import os
import subprocess

def create_video():
    first_frame = cv2.imread('/app/first_frame.png')
    clean_bg = first_frame.copy()
    
    # Erase Shape 4 completely (it's at x=106..241, y=665..700)
    # We erase a slightly larger area to be safe
    clean_bg[660:710, 100:250] = [255, 255, 255]

    os.makedirs('/app/output', exist_ok=True)

    frames = []
    
    # cx, cy for Shape 4
    cx = 173.5
    cy = 682.5

    for i in range(16):
        t = i / 15.0
        img = clean_bg.copy()
        
        if t <= 0.5:
            # Phase 1: Change size (Scale by 1.222, same border)
            p = t / 0.5
            w = 135 + (165 - 135) * p
            h = 35 + (43 - 35) * p
            border = 2
            fill = True
        else:
            # Phase 2: Become outline-only (Border thickens 2->6, outer size grows by 4)
            p = (t - 0.5) / 0.5
            w = 165 + 4 * p
            h = 43 + 4 * p
            border = 2 + 4 * p
            fill = False
            
        w = int(round(w))
        h = int(round(h))
        border = int(round(border))
        
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = x1 + w
        y2 = y1 + h
        
        # Draw the 4 borders
        # Top
        cv2.rectangle(img, (x1, y1), (x2 - 1, y1 + border - 1), (0, 100, 0), -1)
        # Bottom
        cv2.rectangle(img, (x1, y2 - border), (x2 - 1, y2 - 1), (0, 100, 0), -1)
        # Left
        cv2.rectangle(img, (x1, y1 + border), (x1 + border - 1, y2 - border - 1), (0, 100, 0), -1)
        # Right
        cv2.rectangle(img, (x2 - border, y1 + border), (x2 - 1, y2 - border - 1), (0, 100, 0), -1)
        
        if fill:
            # Fill center with white
            cv2.rectangle(img, (x1 + border, y1 + border), (x2 - 1 - border, y2 - 1 - border), (255, 255, 255), -1)
            
        frames.append(img)
        
    # We must ensure the very first frame exactly matches first_frame.png, 
    # except maybe for rounding errors, but we can just use the original frame
    # for i=0. Actually, our drawing logic is pixel perfect for i=0.
    frames[0] = first_frame.copy()

    # Save to MP4 using imageio
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        # Convert BGR to RGB for imageio
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)
    writer.close()

if __name__ == '__main__':
    create_video()
