import cv2
import numpy as np
import imageio
import os

def solve():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour which should be the triangle
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
    
    # Extract vertices
    if len(approx) >= 3:
        pts = approx.reshape(-1, 2)[:3]
    else:
        # Fallback if approximation fails
        pts = cv2.convexHull(c).squeeze()[:3]

    # Compute side lengths squared
    dists = []
    for i in range(3):
        p1 = pts[i]
        p2 = pts[(i+1)%3]
        dists.append(np.sum((p1 - p2)**2))
    
    # The largest angle is opposite to the longest side
    longest_side_idx = np.argmax(dists)
    target_vertex = pts[(longest_side_idx + 2) % 3]
    center = tuple(map(int, target_vertex))
    
    # Generate frames
    frames = []
    num_frames = 22
    radius = 40
    thickness = 5
    color = (255, 0, 0) # Red in RGB for imageio

    # Convert original to RGB for background
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for f in range(num_frames):
        frame = img_rgb.copy()
        
        # Calculate progress (from 0 to 1)
        progress = f / (num_frames - 1)
        angle = int(360 * progress)
        
        if angle > 0:
            cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, color, thickness, cv2.LINE_AA)
            
        frames.append(frame)

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None)

if __name__ == '__main__':
    solve()
