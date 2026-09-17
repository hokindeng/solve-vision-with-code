import cv2
import numpy as np
import os
import subprocess
import shutil

def create_video():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        print("Error: Could not read /app/first_frame.png")
        return

    os.makedirs('/app/output', exist_ok=True)
    temp_frames_dir = '/app/temp_frames_run'
    os.makedirs(temp_frames_dir, exist_ok=True)

    # Source patches
    # Pink circle is at index 1 (0-indexed).
    pink_patch = img[452:572, 236:356].copy()
    
    # Question mark is around x=872. Target x starting at 811
    qm_patch = img[452:572, 811:931].copy()
    
    num_frames = 25
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        # Blend patches
        blended_patch = cv2.addWeighted(pink_patch, alpha, qm_patch, 1 - alpha, 0)
        
        # Create frame
        frame = img.copy()
        frame[452:572, 811:931] = blended_patch
        
        # Save frame
        cv2.imwrite(f"{temp_frames_dir}/frame_{i:04d}.png", frame)
        
    # Compile video using ffmpeg for reliable H.264 / yuv420p
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f"{temp_frames_dir}/frame_%04d.png",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clean up
    shutil.rmtree(temp_frames_dir)

if __name__ == "__main__":
    create_video()
