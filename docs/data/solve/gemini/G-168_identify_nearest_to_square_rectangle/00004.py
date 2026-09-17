import cv2
import numpy as np
import imageio
import os

img = cv2.imread('/app/first_frame.png')
# Rectangles found:
rects = [
    (429, 104, 69, 68),
    (248, 251, 159, 61),
    (567, 350, 172, 97),
    (436, 525, 152, 61),
    (599, 675, 61, 145),
    (210, 685, 165, 132)
]

os.makedirs('/app/output', exist_ok=True)
writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 2

def draw_texts(frame, reveal_count, highlight_idx=-1):
    for i in range(reveal_count):
        x, y, w, h = rects[i]
        ratio = w / h
        text1 = f"{w}x{h}"
        text2 = f"W/H = {ratio:.2f}"
        
        size1, _ = cv2.getTextSize(text1, font, font_scale, thickness)
        size2, _ = cv2.getTextSize(text2, font, font_scale, thickness)
        
        tx1 = x + w//2 - size1[0]//2
        ty1 = y - 40
        tx2 = x + w//2 - size2[0]//2
        ty2 = y - 15
        
        if highlight_idx == i:
            color = (0, 0, 255) # Red text for the best one
        else:
            color = (0, 0, 0) # Black text
            
        # Draw white outline for better readability
        cv2.putText(frame, text1, (tx1, ty1), font, font_scale, (255, 255, 255), thickness+2)
        cv2.putText(frame, text2, (tx2, ty2), font, font_scale, (255, 255, 255), thickness+2)
        
        cv2.putText(frame, text1, (tx1, ty1), font, font_scale, color, thickness)
        cv2.putText(frame, text2, (tx2, ty2), font, font_scale, color, thickness)

for f in range(48):
    frame = img.copy()
    
    # 0 to 3: no text
    # 4 to 27: reveal text one by one (1 text every 4 frames)
    reveal_count = 0
    if f >= 4:
        reveal_count = min(6, (f - 4) // 4 + 1)
        
    highlight_idx = -1
    if f >= 30:
        highlight_idx = 0 # The first rect is the closest to 1:1
        
    draw_texts(frame, reveal_count, highlight_idx)
    
    if f >= 32:
        # Draw red circle around rects[0]
        x, y, w, h = rects[0]
        center = (x + w//2, y + h//2)
        radius = int(np.sqrt(w**2 + h**2) / 2) + 10
        
        # Animate from f=32 to f=40 (9 frames total, angles from 0 to 360)
        progress = min(1.0, (f - 32) / 8.0)
        end_angle_offset = int(360 * progress)
        
        if end_angle_offset > 0:
            cv2.ellipse(frame, center, (radius, radius), 0, -90, -90 + end_angle_offset, (0, 0, 255), 3)
            
    # Convert BGR to RGB for imageio
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    writer.append_data(frame_rgb)

writer.close()
