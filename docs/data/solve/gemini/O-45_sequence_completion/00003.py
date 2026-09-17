import cv2
import numpy as np
import os
import subprocess

def main():
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    
    os.makedirs('/tmp/frames', exist_ok=True)
    
    green_patch = img[462:562, 516:616].copy()
    qm_patch = img[462:562, 839:939].copy()
    
    N = 25
    for i in range(N):
        alpha = i / (N - 1)
        frame = img.copy()
        
        blended = cv2.addWeighted(green_patch, alpha, qm_patch, 1 - alpha, 0)
        frame[462:562, 839:939] = blended
        
        out_path = f'/tmp/frames/frame_{i:03d}.png'
        cv2.imwrite(out_path, frame)
    
    os.makedirs('/app/output', exist_ok=True)
    out_video = '/app/output/video.mp4'
    
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', '16',
        '-i', '/tmp/frames/frame_%03d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_video
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
