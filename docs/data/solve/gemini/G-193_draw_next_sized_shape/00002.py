import cv2
import numpy as np
import imageio
import os

os.makedirs('/app/output', exist_ok=True)

img = cv2.imread('/app/first_frame.png')
patch = img[484:484+57, 409:409+58].copy()

# The mask of the shape
mask = (patch == [99, 85, 75]).all(axis=-1)

frames = []
total_frames = 60
start_f = 5
end_f = 55

for f in range(total_frames):
    frame = img.copy()
    
    if f < start_f:
        rows = 0
    elif f > end_f:
        rows = 57
    else:
        rows = int(57 * (f - start_f) / (end_f - start_f))
        
    if rows > 0:
        # We only copy the top `rows`
        for y in range(rows):
            for x in range(58):
                if mask[y, x]:
                    frame[484+y, 854+x] = [99, 85, 75]
                    
    # Convert BGR to RGB for imageio
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p', quality=10)
