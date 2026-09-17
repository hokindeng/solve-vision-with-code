import cv2
import numpy as np
import os
import subprocess
import shutil

def find_middle_dot(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dots.append((cX, cY))
            
    dots.sort(key=lambda x: x[0])
    
    if not dots:
        raise ValueError("No dots found in the image.")
        
    mid_idx = len(dots) // 2
    return dots[mid_idx], image

def make_video():
    image_path = '/app/first_frame.png'
    mid_dot, base_img = find_middle_dot(image_path)
    
    center_x, center_y = mid_dot
    radius = 35
    thickness = 4
    color = (0, 0, 255) # BGR for red
    
    frames_dir = '/app/temp_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    os.makedirs('/app/output', exist_ok=True)
    out_video_path = '/app/output/video.mp4'
    
    num_frames = 22
    fps = 16
    
    for i in range(num_frames):
        frame = base_img.copy()
        angle = int((i / (num_frames - 1)) * 360)
        
        if angle > 0:
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, -90, -90 + angle, color, thickness, cv2.LINE_AA)
            
        cv2.imwrite(os.path.join(frames_dir, f'frame_{i:04d}.png'), frame)
        
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-crf', '10', '-pix_fmt', 'yuv420p', out_video_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    shutil.rmtree(frames_dir)

if __name__ == '__main__':
    make_video()
