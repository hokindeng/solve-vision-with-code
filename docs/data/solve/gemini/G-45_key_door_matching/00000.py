import cv2
import numpy as np
import os
import subprocess

def solve():
    img = cv2.imread('/app/first_frame.png')

    # 1. Background without agent
    bg_no_agent = img.copy()
    green_mask_full = (bg_no_agent[:, :, 0] == 0) & (bg_no_agent[:, :, 1] == 255) & (bg_no_agent[:, :, 2] == 0)
    bg_no_agent[green_mask_full] = [255, 255, 255]

    # 2. Background without agent and without red key
    bg_no_agent_no_key = bg_no_agent.copy()
    patch = bg_no_agent_no_key[894:949, 484:540]
    red_mask_patch = (patch[:, :, 0] == 0) & (patch[:, :, 1] == 0) & (patch[:, :, 2] == 255)
    patch[red_mask_patch] = [255, 255, 255]

    # 3. Agent mask
    agent_patch = img[81:123, 81:123].copy()
    agent_mask = (agent_patch[:, :, 0] == 0) & (agent_patch[:, :, 1] == 255) & (agent_patch[:, :, 2] == 0)

    grid_size = 1024 / 15

    # 4. Path logic
    path_to_key = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 6), (5, 7), (6, 7), (7, 7), (7, 6), (7, 5), (8, 5), (9, 5), (9, 4), (9, 3), (8, 3), (7, 3), (6, 3), (5, 3), (5, 2), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12), (13, 13), (12, 13), (11, 13), (10, 13), (9, 13), (8, 13), (7, 13)]
    path_to_door = [(7, 13), (8, 13), (9, 13), (10, 13), (11, 13), (12, 13), (13, 13), (13, 12), (13, 11), (13, 10), (13, 9), (13, 8), (13, 7), (13, 6), (13, 5), (13, 4), (13, 3), (13, 2), (13, 1), (12, 1), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (10, 11), (9, 11), (8, 11), (7, 11), (6, 11), (5, 11), (5, 12), (5, 13), (4, 13), (3, 13), (3, 12), (3, 11)]

    frames = []
    # 4 frames pause at start
    for _ in range(4):
        frames.append((path_to_key[0], False))

    # 50 segments
    for i in range(len(path_to_key) - 1):
        p1 = path_to_key[i]
        p2 = path_to_key[i+1]
        frames.append((p1, False))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        frames.append((mid, False))

    # 4 frames pause at key (key collected)
    for _ in range(4):
        frames.append((path_to_key[-1], True))

    # 42 segments
    for i in range(len(path_to_door) - 1):
        p1 = path_to_door[i]
        p2 = path_to_door[i+1]
        frames.append((p1, True))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        frames.append((mid, True))

    # 6 frames pause at door
    for _ in range(6):
        frames.append((path_to_door[-1], True))

    os.makedirs('/app/output/frames', exist_ok=True)
    
    for idx, (pos, key_collected) in enumerate(frames):
        x, y = pos
        bg = bg_no_agent_no_key if key_collected else bg_no_agent
        frame = bg.copy()
        
        px = round(x * grid_size + grid_size / 2)
        py = round(y * grid_size + grid_size / 2)
        
        y1 = py - 21
        y2 = py + 21
        x1 = px - 21
        x2 = px + 21
        
        frame[y1:y2, x1:x2][agent_mask] = [0, 255, 0]
        
        cv2.imwrite(f'/app/output/frames/frame_{idx:03d}.png', frame)
        
    # generate video
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-framerate', '16',
        '-i', '/app/output/frames/frame_%03d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(ffmpeg_cmd, check=True)

if __name__ == '__main__':
    solve()
