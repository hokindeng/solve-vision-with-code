import cv2
import numpy as np
import os
import subprocess
import shutil

def create_video():
    img_path = '/app/first_frame.png'
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/app/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    leftmost_c = min(contours, key=lambda c: cv2.boundingRect(c)[0])
    (x, y), radius = cv2.minEnclosingCircle(leftmost_c)
    center = (int(x), int(y))
    
    # Add padding to the radius for the red circle
    draw_radius = int(radius + 20)
    
    fps = 16
    total_frames = 48
    draw_frames = 40
    
    for i in range(total_frames):
        frame = img.copy()
        
        # Calculate progress of drawing the circle
        if i == 0:
            progress = 0
        elif i >= draw_frames:
            progress = 1.0
        else:
            # Ease in out for smooth drawing
            # Or just linear
            progress = i / draw_frames
            
        end_angle = -90 + progress * 360
        
        if progress > 0:
            # Draw arc starting from top (-90 degrees) to end_angle
            cv2.ellipse(frame, center, (draw_radius, draw_radius), 0, -90, end_angle, (0, 0, 255), 8, cv2.LINE_AA)
            
        out_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        cv2.imwrite(out_path, frame)
        
    out_video = os.path.join(out_dir, 'video.mp4')
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'scale=1024:1024',
        out_video
    ]
    subprocess.run(cmd, check=True)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"Video saved to {out_video}")
    
if __name__ == '__main__':
    create_video()
