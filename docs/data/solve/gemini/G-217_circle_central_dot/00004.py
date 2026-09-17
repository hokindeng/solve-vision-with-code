import cv2
import numpy as np
import os
import subprocess

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        print("Error: Could not read /app/first_frame.png")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            dots.append(c)
            
    dots.sort(key=lambda c: cv2.boundingRect(c)[0])
    
    if dots:
        middle_index = len(dots) // 2
        middle_dot = dots[middle_index]
        M = cv2.moments(middle_dot)
        if M['m00'] != 0:
            cx = int(round(M['m10']/M['m00']))
            cy = int(round(M['m01']/M['m00']))
        else:
            x, y, w, h = cv2.boundingRect(middle_dot)
            cx = x + w // 2
            cy = y + h // 2
        center = (cx, cy)
        
        _, _, w, h = cv2.boundingRect(middle_dot)
        radius = int(max(w, h) / 2) + 12
    else:
        # Fallback if no dots found
        center = (512, 512)
        radius = 28

    thickness = 4
    color = (0, 0, 255) # Red in BGR

    os.makedirs('/app/output', exist_ok=True)
    temp_dir = '/app/output/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    num_frames = 22
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            end_angle = i * 360 / (num_frames - 1)
            # Draw arc starting from top (270 degrees) and going clockwise
            cv2.ellipse(frame, center, (radius, radius), 0, 270, 270 + end_angle, color, thickness, cv2.LINE_AA)
            
        cv2.imwrite(f'{temp_dir}/frame_{i:04d}.png', frame)

    # Use ffmpeg to encode the frames
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', f'{temp_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '10',
        '/app/output/video.mp4'
    ], check=True)

    # Cleanup temp frames
    for i in range(num_frames):
        try:
            os.remove(f'{temp_dir}/frame_{i:04d}.png')
        except OSError:
            pass
    os.rmdir(temp_dir)

if __name__ == '__main__':
    generate_video()
