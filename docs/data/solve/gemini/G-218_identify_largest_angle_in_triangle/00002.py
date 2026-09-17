import cv2
import numpy as np
import os
import subprocess
import math

def get_largest_angle_vertex(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Just take the largest contour
    largest_cnt = max(contours, key=cv2.contourArea)
    approx = None
    for eps_mult in np.linspace(0.01, 0.1, 10):
        epsilon = eps_mult * cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
        if len(approx) == 3:
            break
            
    if approx is None or len(approx) != 3:
        # Fallback
        epsilon = 0.05 * cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
        approx = approx[:3]

    pts = [pt[0] for pt in approx]
    
    # Calculate side lengths
    def dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    dists = [
        dist(pts[1], pts[2]), # side opposite to pts[0]
        dist(pts[0], pts[2]), # side opposite to pts[1]
        dist(pts[0], pts[1])  # side opposite to pts[2]
    ]
    
    max_idx = np.argmax(dists)
    return (int(pts[max_idx][0]), int(pts[max_idx][1]))

def create_video():
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    
    center = get_largest_angle_vertex(img_path)
    print("Largest angle vertex:", center)
    
    radius = 40
    color = (0, 0, 255) # Red in BGR
    thickness = 4
    
    os.makedirs('/app/output', exist_ok=True)
    
    frames = []
    num_frames = 22
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            angle = int(360 * i / (num_frames - 1))
            cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, color, thickness, lineType=cv2.LINE_AA)
            
        frames.append(frame)
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = '/app/output/video.mp4'
    temp_path = '/app/output/temp_video.mp4'
    
    out = cv2.VideoWriter(temp_path, fourcc, 16.0, (img.shape[1], img.shape[0]))
    for f in frames:
        out.write(f)
    out.release()
    
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        out_path
    ], check=True)
    os.remove(temp_path)

if __name__ == '__main__':
    create_video()
