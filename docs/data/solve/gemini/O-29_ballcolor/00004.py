import cv2
import numpy as np
import imageio
import os

# 1. Load initial frame
img = cv2.imread('/app/first_frame.png')
if img is None:
    raise FileNotFoundError("Could not find /app/first_frame.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Extract templates directly from the first frame
# We know a red ball is exactly at center (452, 456)
cx, cy = 452, 456
red_patch = img_rgb[cy-25:cy+26, cx-25:cx+26].copy()

# Masks for the ball
ball_mask = np.any(red_patch != [240, 240, 240], axis=-1)
interior_mask = np.all(red_patch == [255, 50, 50], axis=-1)

# Generate other color patches
cyan_patch = red_patch.copy()
cyan_patch[interior_mask] = [50, 180, 180]

purple_patch = red_patch.copy()
purple_patch[interior_mask] = [75, 0, 130]

navy_patch = red_patch.copy()
navy_patch[interior_mask] = [0, 0, 128]

# Initial ball center positions identified
R_balls = np.array([
    [452., 456.], [504., 456.], [452., 508.], [504., 508.], [452., 561.]
])
C_balls = np.array([
    [387., 206.], [387., 259.]
])
P_balls = np.array([
    [180., 489.], [233., 489.], [180., 542.], [233., 542.], [180., 594.], [233., 594.]
])
N_balls = np.array([
    [466., 755.], [519., 755.], [571., 755.], [466., 808.], [519., 808.], [571., 808.],
    [466., 860.], [519., 860.], [571., 860.], [466., 913.], [519., 913.], [571., 913.]
])

# 3. Create clean background by removing all balls
clean_bg = img_rgb.copy()
all_initial = np.vstack([R_balls, C_balls, P_balls, N_balls])
for x, y in all_initial:
    bx, by = int(round(x)), int(round(y))
    clean_bg[by-25:by+26, bx-25:bx+26] = [240, 240, 240]

def center(balls):
    return np.mean(balls, axis=0)

def draw_balls(bg, balls, patch, mask):
    for x, y in balls:
        bx, by = int(round(x)), int(round(y))
        y1, y2 = by - 25, by + 26
        x1, x2 = bx - 25, bx + 26
        # Bounds check just in case, though guaranteed to be inside 1024x1024
        if y1 >= 0 and y2 <= bg.shape[0] and x1 >= 0 and x2 <= bg.shape[1]:
            roi = bg[y1:y2, x1:x2]
            roi[mask] = patch[mask]

# Initialize active clusters
active_red = R_balls.copy()
cyan = C_balls.copy()
purple = P_balls.copy()
navy = N_balls.copy()

frames = []
total_frames = 75

for f in range(total_frames):
    # Animation timeline logic
    if f == 0:
        start_red = active_red.copy()
        v = center(cyan) - center(start_red)
    elif 0 < f <= 19:
        active_red = start_red + v * (f / 19.0)
    elif f == 20:
        active_red = np.vstack([active_red, cyan])
        cyan = np.empty((0, 2))
    elif f == 24:
        start_red = active_red.copy()
        v = center(purple) - center(start_red)
    elif 24 < f <= 43:
        active_red = start_red + v * ((f - 24) / 19.0)
    elif f == 44:
        active_red = np.vstack([active_red, purple])
        purple = np.empty((0, 2))
    elif f == 48:
        start_red = active_red.copy()
        v = center(navy) - center(start_red)
    elif 48 < f <= 67:
        active_red = start_red + v * ((f - 48) / 19.0)
    elif f == 68:
        active_red = np.vstack([active_red, navy])
        navy = np.empty((0, 2))
        
    # Draw frame
    frame = clean_bg.copy()
    if len(cyan) > 0:
        draw_balls(frame, cyan, cyan_patch, ball_mask)
    if len(purple) > 0:
        draw_balls(frame, purple, purple_patch, ball_mask)
    if len(navy) > 0:
        draw_balls(frame, navy, navy_patch, ball_mask)
    if len(active_red) > 0:
        # Drawing red balls last so they appear on top if they overlap
        draw_balls(frame, active_red, red_patch, ball_mask)
        
    frames.append(frame)

# 4. Save video
os.makedirs('/app/output', exist_ok=True)
output_path = '/app/output/video.mp4'
imageio.mimwrite(
    output_path, 
    frames, 
    fps=16, 
    macro_block_size=None, 
    format='FFMPEG', 
    codec='libx264', 
    pixelformat='yuv420p'
)
print(f"Video saved to {output_path}")

