import cv2
import numpy as np
import math
import subprocess
import os
import shutil

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def get_angle(p1, p2, p3):
    # Angle at p2 using the cosine rule
    a = dist(p1, p2)
    b = dist(p2, p3)
    c = dist(p1, p3)
    cos_val = (a**2 + b**2 - c**2) / (2 * a * b + 1e-6)
    cos_val = max(min(cos_val, 1.0), -1.0)
    return math.degrees(math.acos(cos_val))

def solve():
    img_path = '/app/first_frame.png'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize to find the triangle
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours found.")
        return
        
    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # We expect a triangle
    pts = [pt[0] for pt in approx]
    
    if len(pts) < 3:
        print("Error: Not enough vertices found.")
        return
    
    # Calculate angles at each vertex
    angles = []
    num_pts = len(pts)
    for i in range(num_pts):
        p1 = pts[(i - 1) % num_pts]
        p2 = pts[i]
        p3 = pts[(i + 1) % num_pts]
        angles.append(get_angle(p1, p2, p3))
        
    # Find the vertex with the largest interior angle
    max_idx = np.argmax(angles)
    target_pt = pts[max_idx]
    
    # Slightly adjust the center inwards towards the centroid 
    # to account for line thickness (approx 7px wide -> 3.5px shift)
    centroid = np.mean(pts, axis=0)
    direction = centroid - target_pt
    norm = np.linalg.norm(direction)
    if norm > 0:
        center = target_pt + (direction / norm) * 3.5
    else:
        center = target_pt
        
    center = (int(round(center[0])), int(round(center[1])))
    
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, 'tmp_frames')
    os.makedirs(tmp_dir, exist_ok=True)
    
    num_frames = 22
    for i in range(num_frames):
        frame = img.copy()
        if i > 0:
            # Progressively draw the circle over the frames
            end_angle = i * 360 / (num_frames - 1)
            cv2.ellipse(frame, center, (40, 40), 0, 0, end_angle, (0, 0, 255), 4, cv2.LINE_AA)
        
        frame_path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, frame)
        
    # Generate the video
    cmd = [
        "ffmpeg", "-y", "-framerate", "16", "-i", os.path.join(tmp_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out_dir, "video.mp4")
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clean up temporary frames
    shutil.rmtree(tmp_dir)
    
if __name__ == '__main__':
    solve()
