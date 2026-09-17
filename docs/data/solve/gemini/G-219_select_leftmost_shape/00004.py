import cv2
import numpy as np
import math
import subprocess
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read first_frame.png")
    
    height, width, _ = img.shape
    
    # Find leftmost shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            x, y, w, h = cv2.boundingRect(c)
            centers.append({
                'cx': x + w / 2,
                'cy': y + h / 2,
                'w': w,
                'h': h,
                'contour': c,
                'x': x
            })
            
    centers.sort(key=lambda item: item['x'])
    
    leftmost = centers[0]
    cx = int(leftmost['cx'])
    cy = int(leftmost['cy'])
    w = leftmost['w']
    h = leftmost['h']
    
    radius = int(math.hypot(w/2, h/2) + 15)
    
    num_frames = 48
    fps = 16
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # We will write to a temporary file and then re-encode with ffmpeg to match exactly the required H.264/yuv420p
    temp_path = '/app/temp_video.mp4'
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Calculate angle for arc
        end_angle = int((i / (num_frames - 1)) * 360)
        
        if end_angle > 0:
            # cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
            # angle=0 means standard orientation, startAngle=-90 means start at top
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + end_angle, (0, 0, 255), 6, cv2.LINE_AA)
            
        out.write(frame)
        
    out.release()
    
    # Convert using ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error', '-i', temp_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        out_path
    ], check=True)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__ == '__main__':
    create_video()
