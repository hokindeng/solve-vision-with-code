import cv2
import numpy as np
import imageio
import os

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    
    dx, dy, dw, dh = 108, 602, 161, 161
    cx = dx + dw / 2.0
    cy = dy + dh / 2.0
    
    D_patch = img[dy:dy+dh, dx:dx+dw].copy()
    
    frames = []
    
    for i in range(32):
        frame = img.copy()
        r = int(74 * (i / 31.0))
        if r > 0:
            cv2.circle(frame, (int(cx), int(cy)), r, (255, 255, 255), -1)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    outline_patch = D_patch.copy()
    cv2.circle(outline_patch, (int(dw/2), int(dh/2)), 74, (255, 255, 255), -1)
    mask = (cv2.cvtColor(outline_patch, cv2.COLOR_BGR2GRAY) < 250)
    
    base_empty = img.copy()
    base_empty[dy:dy+dh, dx:dx+dw] = (255, 255, 255)
    
    for i in range(32):
        t = i / 31.0
        curr_cx = cx + (512 - cx) * t
        curr_cy = cy + (512 - cy) * t
        
        frame = base_empty.copy()
        
        tl_x = int(round(curr_cx - dw / 2.0))
        tl_y = int(round(curr_cy - dh / 2.0))
        
        for y in range(dh):
            for x in range(dw):
                if mask[y, x]:
                    if 0 <= tl_y + y < 1024 and 0 <= tl_x + x < 1024:
                        frame[tl_y+y, tl_x+x] = outline_patch[y, x]
                        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
