import cv2
import numpy as np
import os
import subprocess

# Points for 1, 2, 3, 4
points = [(445, 861), (193, 533), (523, 503), (534, 163)]
num_lines = 3
total_frames = 55
frames_per_line = (total_frames - 1) // num_lines # 18

# Read the first frame
base_img = cv2.imread('/app/first_frame.png')

# Create a mask for the dots (circles)
gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dots_mask_2d = np.zeros(base_img.shape[:2], dtype=np.uint8)
cv2.drawContours(dots_mask_2d, contours, -1, 255, -1)
# Convert 2D mask to boolean mask for numpy indexing
dots_mask = dots_mask_2d > 0

os.makedirs('/app/frames', exist_ok=True)

# Function to draw a partial line
def draw_partial_line(img, pt1, pt2, progress, color=(0, 0, 255), thickness=8):
    x = int(pt1[0] + (pt2[0] - pt1[0]) * progress)
    y = int(pt1[1] + (pt2[1] - pt1[1]) * progress)
    cv2.line(img, pt1, (x, y), color, thickness, lineType=cv2.LINE_AA)

frame_idx = 0
# Frame 0: original
cv2.imwrite(f'/app/frames/frame_{frame_idx:04d}.png', base_img)
frame_idx += 1

# current_img holds the fully drawn lines up to the current point
current_img = base_img.copy()

for i in range(num_lines):
    pt1 = points[i]
    pt2 = points[i+1]
    
    # We want to draw this line over `frames_per_line` frames
    for k in range(1, frames_per_line + 1):
        progress = k / frames_per_line
        img_copy = current_img.copy()
        draw_partial_line(img_copy, pt1, pt2, progress, thickness=8)
        
        # Restore dots
        img_copy[dots_mask] = base_img[dots_mask]
        
        cv2.imwrite(f'/app/frames/frame_{frame_idx:04d}.png', img_copy)
        frame_idx += 1
        
    # After the loop, the line is fully drawn, update current_img
    cv2.line(current_img, pt1, pt2, (0, 0, 255), 8, lineType=cv2.LINE_AA)
    current_img[dots_mask] = base_img[dots_mask]

# Now we have 55 frames (0 to 54)
os.makedirs('/app/output', exist_ok=True)

# Encode to mp4
subprocess.run([
    'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
], check=True)
