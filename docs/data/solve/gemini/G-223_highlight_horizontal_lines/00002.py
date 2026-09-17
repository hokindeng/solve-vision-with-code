import cv2
import numpy as np
import math
import os
import subprocess
import shutil

def solve():
    img = cv2.imread('/app/first_frame.png')
    bg_color = img[0, 0]
    diff = cv2.absdiff(img, bg_color)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 10

    contours, _ = cv2.findContours(mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # horizontal means width is significantly larger than height
        if w > 1.5 * h:
            horizontal_lines.append((x, y, w, h))

    os.makedirs('/app/frames', exist_ok=True)
    num_frames = 48
    
    for f in range(num_frames):
        frame = img.copy()
        
        # Animate drawing the circles by increasing the angle
        angle = 360.0 * f / (num_frames - 1)
        
        for (x, y, w, h) in horizontal_lines:
            center = (int(x + w/2), int(y + h/2))
            radius = int(math.hypot(w/2, h/2)) + 15
            
            if angle > 0:
                cv2.ellipse(frame, center, (radius, radius), 0, -90, -90 + angle, (0, 0, 0), 6, cv2.LINE_AA)
                
        cv2.imwrite(f'/app/frames/frame_{f:03d}.png', frame)

    os.makedirs('/app/output', exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'scale=1024:1024',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    
    # clean up frames
    shutil.rmtree('/app/frames')

if __name__ == '__main__':
    solve()
