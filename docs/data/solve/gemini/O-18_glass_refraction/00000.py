import cv2
import numpy as np
import math
import subprocess

# 1. Load images
orig = cv2.imread('/app/first_frame.png')

# 2. Create cleaned image (angle annotation removed)
cleaned = orig.copy()
for y in range(512):
    for x in range(orig.shape[1]):
        b, g, r = orig[y, x]
        if b == 255 and g == 255 and r == 255:
            continue
        # Keep blue line
        if b > 150 and g < 100 and r < 100:
            continue
        # Keep normal line (gray)
        if 510 <= x <= 514 and abs(int(b) - 150) < 10 and abs(int(g) - 150) < 10 and abs(int(r) - 150) < 10:
            continue
        # Remove everything else above interface (the annotation)
        cleaned[y, x] = [255, 255, 255]

# 3. Calculate Snell's law
n1 = 1.00
n2 = 1.850
theta1 = 69.8 * math.pi / 180.0
sin_theta2 = (n1 * math.sin(theta1)) / n2
theta2 = math.asin(sin_theta2)
tan_theta2 = math.tan(theta2)

dy_total = 1024 - 512
dx_total = dy_total * tan_theta2

# 4. Generate video frames
frames_dir = '/app/frames'
subprocess.run(['mkdir', '-p', frames_dir])

total_frames = 70
fps = 16

for i in range(total_frames):
    # Fade out annotation over first 30 frames
    fade_len = 30
    if i < fade_len:
        alpha = 1.0 - (i / float(fade_len))
        frame_img = cv2.addWeighted(orig, alpha, cleaned, 1.0 - alpha, 0)
    else:
        frame_img = cleaned.copy()
        
    # Draw red refracted ray
    # Grow over first 65 frames
    grow_len = 65
    progress = min(1.0, i / float(grow_len))
    
    if progress > 0:
        x_curr = 512 + dx_total * progress
        y_curr = 512 + dy_total * progress
        
        shift = 4
        x0 = int(round(512 * (2**shift)))
        y0 = int(round(512 * (2**shift)))
        x1 = int(round(x_curr * (2**shift)))
        y1 = int(round(y_curr * (2**shift)))
        
        cv2.line(frame_img, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA, shift=shift)
        
    cv2.imwrite(f"{frames_dir}/frame_{i:04d}.png", frame_img)

# 5. Encode video
subprocess.run([
    'ffmpeg', '-y', '-framerate', str(fps), '-i', f"{frames_dir}/frame_%04d.png",
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
], check=True)
