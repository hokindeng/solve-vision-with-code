import cv2
import numpy as np
import subprocess
import os

def create_video():
    first_frame = cv2.imread('/app/first_frame.png')
    if first_frame is None:
        raise FileNotFoundError("Could not find /app/first_frame.png")
        
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for c in contours:
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            dots.append((cx, cy))
            
    dots.sort(key=lambda p: p[0])
    
    if not dots:
        print("No dots found!")
        return
        
    middle_idx = len(dots) // 2
    mcx, mcy = dots[middle_idx]
    
    num_frames = 22
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f"{first_frame.shape[1]}x{first_frame.shape[0]}",
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_path
    ]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    radius = 35
    thickness = 6
    
    for i in range(num_frames):
        frame = first_frame.copy()
        progress = i / (num_frames - 1)
        
        if progress > 0:
            end_angle = -90 + 360 * progress
            cv2.ellipse(frame, (mcx, mcy), (radius, radius), 0, -90, float(end_angle), (0, 0, 255), thickness, lineType=cv2.LINE_AA)
            
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    create_video()
