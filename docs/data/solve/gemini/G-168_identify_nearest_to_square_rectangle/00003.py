import cv2
import numpy as np
import imageio
import os
import math

img = cv2.imread('/app/first_frame.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:
        rects.append((x, y, w, h))

# Sort left to right
rects.sort(key=lambda r: r[0])

# Find the one closest to square
best_i = -1
best_diff = float('inf')
for i, (x, y, w, h) in enumerate(rects):
    ratio = w / h
    symmetric_diff = max(w/h, h/w) - 1
    if symmetric_diff < best_diff:
        best_diff = symmetric_diff
        best_i = i

fps = 16
total_frames = 48
frames = []

for f in range(total_frames):
    frame = img.copy()
    
    # 1. Compare step-by-step
    # Each rect gets 8 frames to appear.
    for i, (x, y, w, h) in enumerate(rects):
        if f >= i * 8:
            ratio = w / h
            text = f"w/h={w}/{h}={ratio:.2f}"
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + w // 2 - text_size[0] // 2
            text_y = y - 40
            
            # Make sure it doesn't go out of bounds
            text_x = max(10, min(frame.shape[1] - text_size[0] - 10, text_x))
            text_y = max(text_size[1] + 10, text_y)
            
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 2. Draw red circle (Frames 32 to 47)
    if f >= 32:
        x, y, w, h = rects[best_i]
        center = (x + w // 2, y + h // 2)
        radius = int(math.hypot(w / 2, h / 2)) + 10
        
        progress = min(1.0, (f - 32) / 10.0)
        angle = int(360 * progress)
        
        if angle > 0:
            cv2.ellipse(frame, center, (radius, radius), 0, -90, -90 + angle, (0, 0, 255), 4)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

os.makedirs('/app/output', exist_ok=True)
imageio.mimwrite('/app/output/video.mp4', frames, fps=fps, codec='libx264', pixelformat='yuv420p')
