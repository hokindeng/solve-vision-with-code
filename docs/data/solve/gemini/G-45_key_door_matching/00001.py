import cv2
import numpy as np
import os
import subprocess
import shutil

def main():
    img = cv2.imread('/app/first_frame.png')
    grid_size = 15
    cell_size = 1024 / grid_size

    # 1. Agent offsets
    agent_ys, agent_xs = np.where(np.all(img == [0, 255, 0], axis=-1))
    agent_cy, agent_cx = int(1.5 * cell_size), int(1.5 * cell_size)
    agent_offsets = list(zip(agent_ys - agent_cy, agent_xs - agent_cx))

    # 2. Backgrounds
    clean_bg = img.copy()
    for y, x in zip(agent_ys, agent_xs):
        clean_bg[y, x] = [255, 255, 255]

    bg_with_key = clean_bg.copy()
    bg_without_key = clean_bg.copy()

    key_ys, key_xs = np.where(np.all(bg_without_key == [0, 255, 255], axis=-1))
    for y, x in zip(key_ys, key_xs):
        if int(x / cell_size) == 13 and int(y / cell_size) == 10:
            bg_without_key[y, x] = [255, 255, 255]

    # 3. Path
    path1 = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (3, 6), (3, 7), (2, 7), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 13), (3, 13), (3, 12), (3, 11), (4, 11), (5, 11), (5, 12), (5, 13), (6, 13), (7, 13), (7, 12), (7, 11), (8, 11), (9, 11), (10, 11), (11, 11), (11, 10), (11, 9), (12, 9), (13, 9), (13, 10)]
    path2 = [(13, 10), (13, 9), (13, 8), (13, 7), (13, 6), (13, 5), (12, 5), (11, 5), (10, 5), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (8, 9), (7, 9), (6, 9), (5, 9), (5, 8), (5, 7), (6, 7), (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (11, 2), (11, 1), (10, 1), (9, 1)]
    full_path = path1 + path2[1:]

    num_frames = 154
    total_steps = len(full_path) - 1 # 70

    frames_dir = '/app/temp_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    for i in range(num_frames):
        t = i / (num_frames - 1) * total_steps
        
        if t >= 37:
            frame = bg_without_key.copy()
        else:
            frame = bg_with_key.copy()
            
        idx = int(t)
        fraction = t - idx
        if idx < total_steps:
            p1 = full_path[idx]
            p2 = full_path[idx+1]
            cx1, cy1 = (p1[0] + 0.5) * cell_size, (p1[1] + 0.5) * cell_size
            cx2, cy2 = (p2[0] + 0.5) * cell_size, (p2[1] + 0.5) * cell_size
            cx = cx1 + fraction * (cx2 - cx1)
            cy = cy1 + fraction * (cy2 - cy1)
        else:
            p1 = full_path[total_steps]
            cx, cy = (p1[0] + 0.5) * cell_size, (p1[1] + 0.5) * cell_size
            
        draw_cx, draw_cy = int(round(cx)), int(round(cy))
        
        for dy, dx in agent_offsets:
            frame[draw_cy + dy, draw_cx + dx] = [0, 255, 0]
            
        cv2.imwrite(f'{frames_dir}/frame_{i:03d}.png', frame)

    os.makedirs('/app/output', exist_ok=True)
    out_file = '/app/output/video.mp4'
    if os.path.exists(out_file):
        os.remove(out_file)

    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_file
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    shutil.rmtree(frames_dir)

if __name__ == '__main__':
    main()
