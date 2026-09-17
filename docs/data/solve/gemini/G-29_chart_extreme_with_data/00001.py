import cv2
import numpy as np
import os
import subprocess

def main():
    os.makedirs('/app/output', exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # The max value is 97.
    # The dot for 97 is located at bounding box: xmin=292, ymin=222, xmax=332, ymax=262
    xmin, ymin, xmax, ymax = 292, 222, 332, 262
    
    margin = 4
    x1 = xmin - margin
    y1 = ymin - margin
    x2 = xmax + margin
    y2 = ymax + margin
    
    W = x2 - x1
    H = y2 - y1
    total_length = 2 * (W + H)
    
    num_frames = 48
    fps = 16
    
    os.makedirs('/tmp/frames', exist_ok=True)
    
    color = (0, 0, 255) # Red
    thickness = 4
    
    for f in range(num_frames):
        frame = img.copy()
        
        progress = f / (num_frames - 1)
        current_length = progress * total_length
        
        if current_length > 0:
            l1 = min(current_length, W)
            cv2.line(frame, (x1, y1), (x1 + int(l1), y1), color, thickness)
        
        if current_length > W:
            l2 = min(current_length - W, H)
            cv2.line(frame, (x2, y1), (x2, y1 + int(l2)), color, thickness)
            
        if current_length > W + H:
            l3 = min(current_length - W - H, W)
            cv2.line(frame, (x2, y2), (x2 - int(l3), y2), color, thickness)
            
        if current_length > 2*W + H:
            l4 = min(current_length - 2*W - H, H)
            cv2.line(frame, (x1, y2), (x1, y2 - int(l4)), color, thickness)
            
        cv2.imwrite(f'/tmp/frames/frame_{f:04d}.png', frame)
        
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', '/tmp/frames/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
