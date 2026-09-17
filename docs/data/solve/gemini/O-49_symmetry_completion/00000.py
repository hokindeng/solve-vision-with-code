import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")

    r_min, r_max = 204, 819
    c_min, c_max = 204, 819

    cells_to_fill = [(1, 5), (2, 4), (3, 3), (3, 5), (5, 3), (5, 5)]

    bounds = [
        (2, 101),
        (105, 203),
        (207, 306),
        (310, 408),
        (412, 510),
        (514, 613)
    ]

    fill_color = [105, 150, 5]

    frames = []
    num_frames = 35

    for f in range(num_frames):
        frame = img.copy()
        
        # Animate from f=0 to f=30, holding for the last 5 frames
        progress = min(1.0, f / 30.0)
        
        if progress > 0:
            for r, c in cells_to_fill:
                r_start, r_end = bounds[r]
                c_start, c_end = bounds[c]
                
                r_center = (r_start + r_end) / 2.0
                c_center = (c_start + c_end) / 2.0
                
                r_len = r_end - r_start + 1
                c_len = c_end - c_start + 1
                
                curr_r_len = r_len * progress
                curr_c_len = c_len * progress
                
                cr_start = int(round(r_center - curr_r_len / 2.0))
                cr_end = int(round(r_center + curr_r_len / 2.0))
                cc_start = int(round(c_center - curr_c_len / 2.0))
                cc_end = int(round(c_center + curr_c_len / 2.0))
                
                cr_start = max(r_start, cr_start)
                cr_end = min(r_end, cr_end)
                cc_start = max(c_start, cc_start)
                cc_end = min(c_end, cc_end)
                
                if cr_end >= cr_start and cc_end >= cc_start:
                    global_r_start = r_min + cr_start
                    global_r_end = r_min + cr_end
                    global_c_start = c_min + cc_start
                    global_c_end = c_min + cc_end
                    
                    frame[global_r_start:global_r_end+1, global_c_start:global_c_end+1] = fill_color
                
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimsave('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1, quality=10)

if __name__ == '__main__':
    solve()
