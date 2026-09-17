import cv2
import numpy as np
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Find contours to locate the center dot
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dots.append((cX, cY))
            
    # Sort dots by x-coordinate
    dots.sort(key=lambda x: x[0])
    
    # Middle dot by count
    center_dot = dots[len(dots) // 2]
    
    os.makedirs('/app/output', exist_ok=True)
    temp_video = '/app/temp.mp4'
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), 16, (1024, 1024))
    
    num_frames = 22
    for i in range(num_frames):
        frame = img.copy()
        
        # Calculate how much of the circle to draw
        progress = i / (num_frames - 1)
        end_angle = float(-90 + progress * 360)
        
        if progress > 0:
            # Draw an arc that grows over time
            cv2.ellipse(frame, center_dot, (26, 26), 0, -90.0, end_angle, (0, 0, 255), 4, cv2.LINE_AA)
            
        out.write(frame)
        
    out.release()
    
    # re-encode with ffmpeg to h264 yuv420p
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_video, 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
        '/app/output/video.mp4'
    ], check=True, capture_output=True)
    
    if os.path.exists(temp_video):
        os.remove(temp_video)

if __name__ == "__main__":
    main()
