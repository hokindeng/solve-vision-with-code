import cv2
import numpy as np
import os
import subprocess

# Ensure output directory exists
os.makedirs('/app/output', exist_ok=True)
os.makedirs('/tmp/frames', exist_ok=True)

# 1. Load first frame
image = cv2.imread('/app/first_frame.png')
if image is None:
    raise Exception("Could not read /app/first_frame.png")

# 2. Create base frame by removing Pac-Man (fill cell 0,0 with green)
base_frame = image.copy()
base_frame[2:255, 2:255] = [80, 175, 76]

# 3. Extract Pac-Man sprite
sprite = np.zeros((128, 128, 4), dtype=np.uint8)
y_min, y_max = 82, 210
x_min, x_max = 62, 190
patch = image[y_min:y_max, x_min:x_max]

for i in range(128):
    for j in range(128):
        c = patch[i, j]
        c_tuple = tuple(c)
        if c_tuple == (0, 255, 255): # Yellow
            sprite[i, j] = [0, 255, 255, 255]
        elif c_tuple == (255, 255, 255): # White
            sprite[i, j] = [255, 255, 255, 255]
        elif c_tuple == (0, 0, 0): # Black
            sprite[i, j] = [0, 0, 0, 255]
        elif c_tuple == (80, 175, 76): # Green
            sprite[i, j] = [0, 0, 0, 0]
        else:
            # anti-aliased black
            alpha_green = c[1] / 175.0
            alpha_black = 1.0 - alpha_green
            alpha_black = max(0.0, min(1.0, alpha_black))
            sprite[i, j] = [0, 0, 0, int(alpha_black * 255)]

# 4. Define the optimal path
path = [
    (0, 0), (1, 0), (2, 0), (3, 0), 
    (3, 1), (2, 1), (1, 1), (0, 1), 
    (0, 2), (0, 3), (1, 3), (2, 3), 
    (2, 2), (3, 2), (3, 3)
]
num_segments = len(path) - 1 # 14 segments
total_frames = 91

# Helper to rotate sprite
def rotate_sprite(spr, angle):
    if angle == 0:
        return spr
    elif angle == 90:
        return cv2.rotate(spr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(spr, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(spr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return spr

# 5. Generate frames
for f in range(total_frames):
    frame = base_frame.copy()
    
    # Calculate position
    progress = (f / (total_frames - 1)) * num_segments
    idx = int(progress)
    if idx >= num_segments:
        idx = num_segments - 1
        t = 1.0
    else:
        t = progress - idx
        
    r1, c1 = path[idx]
    r2, c2 = path[idx+1]
    
    cur_r = r1 + (r2 - r1) * t
    cur_c = c1 + (c2 - c1) * t
    
    # Calculate angle based on current segment direction
    dr = r2 - r1
    dc = c2 - c1
    if dr == 1:
        angle = 90
    elif dr == -1:
        angle = 270
    elif dc == 1:
        angle = 0
    elif dc == -1:
        angle = 180
    else:
        angle = 0 # Should not happen
        
    current_sprite = rotate_sprite(sprite, angle)
    
    # Calculate top-left coordinate for sprite
    # Base offset is 2 for grid line, plus 82/62 for Pac-Man's bounding box offset in the cell
    y_pos = int(round(cur_r * 256 + 82))
    x_pos = int(round(cur_c * 256 + 62))
    
    # Paste sprite
    for i in range(128):
        for j in range(128):
            alpha = current_sprite[i, j, 3] / 255.0
            if alpha > 0:
                frame[y_pos + i, x_pos + j] = (1 - alpha) * frame[y_pos + i, x_pos + j] + alpha * current_sprite[i, j, :3]
                
    # Save frame
    cv2.imwrite(f'/tmp/frames/frame_{f:04d}.png', frame)

# 6. Encode video with ffmpeg
# We need exactly: H.264, yuv420p, 1024x1024, 16 fps, about 91 frames
ffmpeg_cmd = [
    'ffmpeg', '-y', 
    '-framerate', '16', 
    '-i', '/tmp/frames/frame_%04d.png', 
    '-c:v', 'libx264', 
    '-pix_fmt', 'yuv420p', 
    '-vf', 'scale=1024:1024', 
    '/app/output/video.mp4'
]

subprocess.run(ffmpeg_cmd, check=True)
print("Video generated successfully at /app/output/video.mp4")
