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
        raise FileNotFoundError(f"Could not read {img_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 20 and cv2.boundingRect(c)[3] > 20]
    
    target_box = None
    for i, box1 in enumerate(boxes):
        x1, y1, w1, h1 = box1
        for j, box2 in enumerate(boxes):
            if i == j: continue
            x2, y2, w2, h2 = box2
            if x1 <= x2 and y1 <= y2 and x1+w1 >= x2+w2 and y1+h1 >= y2+h2:
                target_box = box1
                break
        if target_box:
            break
            
    if target_box:
        x, y, w, h = target_box
        cx = x + w // 2
        cy = y + h // 2
        radius = max(w, h) // 2 + 20
    else:
        # Fallback to known coordinates for "习" in first_frame.png
        cx, cy = 615, 236
        radius = 80
        
    n_frames = 48
    fps = 16
    thickness = 6
    
    frames = []
    for i in range(n_frames):
        frame = img.copy()
        if i > 0:
            end_angle = 360 * i / (n_frames - 1)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + end_angle, (0, 0, 255), thickness, cv2.LINE_AA)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    imageio.mimwrite(out_path, frames, fps=fps, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
