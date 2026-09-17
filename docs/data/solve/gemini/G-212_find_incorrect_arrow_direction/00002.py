import cv2
import numpy as np
import os
import subprocess

def create_video():
    img_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    # Read the original image
    base_img = cv2.imread(img_path)
    
    # The target arrow has its minimum enclosing circle at (397.5, 204.5) with radius 83.5
    # We use a slightly larger radius and rounded center
    cx, cy = 398, 204
    radius = 100
    thickness = 6
    
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 48
    fps = 16
    
    for i in range(num_frames):
        frame = base_img.copy()
        
        angle = 360 * i / (num_frames - 1)
        
        if angle > 0:
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + angle, (0, 0, 255), thickness, cv2.LINE_AA)
            
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:04d}.png"), frame)
        
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    create_video()
