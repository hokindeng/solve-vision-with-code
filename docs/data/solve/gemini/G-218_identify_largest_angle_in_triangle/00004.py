import cv2
import numpy as np
import imageio
import math
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return
    
    c = max(contours, key=cv2.contourArea)
    
    approx = None
    for eps_mult in np.linspace(0.001, 0.1, 1000):
        epsilon = eps_mult * cv2.arcLength(c, True)
        temp_approx = cv2.approxPolyDP(c, epsilon, True)
        if len(temp_approx) == 3:
            approx = temp_approx
            break
            
    if approx is None:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
    
    if len(approx) < 3:
        return
        
    pts = [pt[0] for pt in approx[:3]]
    
    def dist(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        
    a = dist(pts[1], pts[2])
    b = dist(pts[0], pts[2])
    c_side = dist(pts[0], pts[1])
    
    angle_A = math.acos(max(min((b**2 + c_side**2 - a**2) / (2 * b * c_side), 1.0), -1.0))
    angle_B = math.acos(max(min((a**2 + c_side**2 - b**2) / (2 * a * c_side), 1.0), -1.0))
    angle_C = math.acos(max(min((a**2 + b**2 - c_side**2) / (2 * a * b), 1.0), -1.0))
    
    angles = [angle_A, angle_B, angle_C]
    max_idx = int(np.argmax(angles))
    
    target_pt = tuple(int(x) for x in pts[max_idx])
    
    radius = 40
    thickness = 5
    color = (255, 0, 0) # Red in RGB
    
    total_frames = 22
    frames = []
    
    for i in range(total_frames):
        frame = img_rgb.copy()
        
        if i >= 3:
            progress = (i - 3 + 1) / (total_frames - 3)
            angle = int(360 * progress)
            cv2.ellipse(frame, target_pt, (radius, radius), 0, 0, angle, color, thickness, cv2.LINE_AA)
            
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
