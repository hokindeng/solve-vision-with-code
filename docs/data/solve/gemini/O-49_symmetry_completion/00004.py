import numpy as np
from PIL import Image
import os
import imageio

# Create output dir
os.makedirs('/app/output', exist_ok=True)

# Load first frame
img = Image.open('/app/first_frame.png').convert('RGB')
base_img = np.array(img)

active_color = np.array([217, 119, 6])
bg_color = np.array([248, 250, 252])

# Missing cells identified based on symmetry
missing_cells = [
    (1, 5),
    (3, 3), (3, 4), (3, 5),
    (5, 3)
]

def get_cell_bbox(r, c):
    y_start = [205, 308, 410, 513, 615, 717][r] + 1
    y_end = [306, 408, 511, 613, 715, 818][r] - 1
    x_start = [205, 308, 410, 513, 615, 717][c] + 1
    x_end = [306, 408, 511, 613, 715, 818][c] - 1
    return y_start, y_end, x_start, x_end

# Generate frames
frames = []
total_frames = 35
sweep_frames = 30
start_x = 512
end_x = 819

for i in range(total_frames):
    frame = base_img.copy()
    if i < sweep_frames:
        current_x = start_x + int((end_x - start_x) * (i / (sweep_frames - 1)))
    else:
        current_x = end_x
        
    for r, c in missing_cells:
        y1, y2, x1, x2 = get_cell_bbox(r, c)
        if current_x >= x1:
            fill_x = min(current_x, x2)
            
            mask = np.all(frame[y1:y2+1, x1:fill_x+1] == bg_color, axis=-1)
            frame[y1:y2+1, x1:fill_x+1][mask] = active_color
            
    frames.append(frame)

# Save as video with specific ffmpeg parameters
out_path = '/app/output/video.mp4'
imageio.mimwrite(out_path, frames, fps=16, codec='libx264', pixelformat='yuv420p')
print("Video saved!")
