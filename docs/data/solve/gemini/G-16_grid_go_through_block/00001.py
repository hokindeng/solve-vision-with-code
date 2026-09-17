import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    first_frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    clean_bg = first_frame_rgb.copy()
    r_start, c_start = 0, 4
    y_cell = 49 + r_start * 93
    x_cell = 49 + c_start * 93
    # Clean the start cell by filling it with the symmetric green color
    clean_bg[y_cell:y_cell+90, x_cell:x_cell+90] = [50, 200, 50]

    agent_patch = first_frame_rgb[70:70+47, 442:442+47].copy()
    mask_orange = np.all(agent_patch == [255, 165, 0], axis=-1)
    mask_dark = np.all(agent_patch == [200, 120, 0], axis=-1)
    agent_mask = mask_orange | mask_dark

    targets = [(0,4), (5,3), (5,2), (7,5), (2,8), (1,6)]
    def get_path(start, end):
        path = []
        r, c = start
        r_end, c_end = end
        while r != r_end:
            r += 1 if r_end > r else -1
            path.append((r, c))
        while c != c_end:
            c += 1 if c_end > c else -1
            path.append((r, c))
        return path

    full_path = [targets[0]]
    for i in range(1, len(targets)):
        full_path.extend(get_path(full_path[-1], targets[i]))

    frames = []
    frames_per_step = 4
    num_hold_start = 4
    num_hold_end = 102 - (len(full_path) - 1) * frames_per_step - num_hold_start

    def get_agent_pos(r, c):
        return 70 + r * 93, 70 + c * 93

    for _ in range(num_hold_start):
        frames.append(first_frame_rgb.copy())

    for i in range(len(full_path) - 1):
        r1, c1 = full_path[i]
        r2, c2 = full_path[i+1]
        y1, x1 = get_agent_pos(r1, c1)
        y2, x2 = get_agent_pos(r2, c2)
        
        for k in range(1, frames_per_step + 1):
            y = int(round(y1 + (y2 - y1) * k / frames_per_step))
            x = int(round(x1 + (x2 - x1) * k / frames_per_step))
            
            frame = clean_bg.copy()
            frame_patch = frame[y:y+47, x:x+47]
            frame_patch[agent_mask] = agent_patch[agent_mask]
            frames.append(frame)

    for _ in range(num_hold_end):
        frames.append(frames[-1].copy())

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == "__main__":
    solve()
