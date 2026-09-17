import cv2
import numpy as np
import os
import subprocess

def create_video():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not find /app/first_frame.png")
    
    # Identify the middle dot. 
    # There are 11 dots. The 6th one is at (655, 512).
    # Next dot is at (705, 512). The gap between centers is 50.
    # Dot radius is ~18. So the circle radius should be around 26 
    # to fit between the dots perfectly.
    center = (655, 512)
    axes = (26, 26)
    
    os.makedirs('/app/output', exist_ok=True)
    
    num_frames = 22
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            # i = 0 -> end_angle = -90 (but we don't draw)
            # i = 21 -> end_angle = 270 (full circle)
            end_angle = -90 + (i / (num_frames - 1)) * 360
            
            # Using cv2.ellipse
            cv2.ellipse(frame, center, axes, 0, -90, end_angle, (0, 0, 255), 4, cv2.LINE_AA)
            
        cv2.imwrite(f'/app/output/frame_{i:02d}.png', frame)
        
    # generate video
    ffmpeg_cmd = [
        'ffmpeg', '-y', 
        '-framerate', '16', 
        '-i', '/app/output/frame_%02d.png', 
        '-c:v', 'libx264', 
        '-pix_fmt', 'yuv420p', 
        '/app/output/video.mp4'
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # clean up frames
    for i in range(num_frames):
        try:
            os.remove(f'/app/output/frame_{i:02d}.png')
        except OSError:
            pass

if __name__ == '__main__':
    create_video()
