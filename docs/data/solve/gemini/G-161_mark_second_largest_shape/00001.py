import cv2
import numpy as np
import os
import subprocess

def create_video():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # Find the circles
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circles.append((x, y, radius))
        
    # Sort circles by radius from largest to smallest
    circles.sort(key=lambda x: x[2], reverse=True)
    
    if len(circles) < 2:
        print("Error: Less than 2 circles found.")
        return
        
    # Get the second largest circle
    target = circles[1]
    cx, cy, radius = target
    cx, cy = int(cx), int(cy)
    
    # We want to circle it in red. Let's add some margin to the radius
    draw_radius = int(radius + 15)
    
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 40
    
    for i in range(num_frames):
        # Calculate the angle for the sweep
        # Starts at 0, ends at 360
        end_angle = i * 360 / (num_frames - 1)
        
        frame = img.copy()
        if end_angle > 0:
            cv2.ellipse(frame, (cx, cy), (draw_radius, draw_radius), 0, -90, -90 + end_angle, (0, 0, 255), 8, cv2.LINE_AA)
            
        cv2.imwrite(f"{frames_dir}/frame_{i:04d}.png", frame)
        
    # Use ffmpeg to create the video
    os.makedirs('/app/output', exist_ok=True)
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', '16',
        '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    create_video()
