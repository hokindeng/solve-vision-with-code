import cv2
import numpy as np
from collections import deque
import os
import subprocess
import shutil

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    centers = [int(np.round((i + 0.5) * 1024 / 15)) for i in range(15)]
    
    maze = np.zeros((15, 15), dtype=int)
    start_pos = None
    end_pos = None
    
    for r in range(15):
        for c in range(15):
            y, x = centers[r], centers[c]
            color = img[y, x]
            if np.all(color == [30, 30, 30]):
                maze[r, c] = 1 # Wall
            else:
                maze[r, c] = 0 # Path
                if np.all(color == [50, 200, 50]):
                    start_pos = (r, c)
                elif np.all(color == [50, 50, 200]):
                    end_pos = (r, c)
    
    # Fallback to bounding box for red flag if not detected by center
    if end_pos is None:
        red_y, red_x = np.where(np.all(img == [50, 50, 200], axis=-1))
        if len(red_y) > 0:
            red_center_y = int(red_y.mean())
            red_center_x = int(red_x.mean())
            end_r = int(red_center_y * 15 / 1024)
            end_c = int(red_center_x * 15 / 1024)
            end_pos = (end_r, end_c)

    if start_pos is None:
        green_y, green_x = np.where(np.all(img == [50, 200, 50], axis=-1))
        if len(green_y) > 0:
            green_center_y = int(green_y.mean())
            green_center_x = int(green_x.mean())
            start_r = int(green_center_y * 15 / 1024)
            start_c = int(green_center_x * 15 / 1024)
            start_pos = (start_r, start_c)

    # Solve maze using BFS
    queue = deque([(start_pos, [start_pos])])
    visited = set([start_pos])
    path = None
    
    while queue:
        (r, c), p = queue.popleft()
        if (r, c) == end_pos:
            path = p
            break
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 15 and 0 <= nc < 15 and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), p + [(nr, nc)]))
                
    if path is None:
        print("No path found!")
        return

    # Clear previous frames if any
    frames_dir = '/tmp/frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    
    total_frames = 63
    anim_frames = 55 # Draw the line progressively over 55 frames
    
    non_white_mask = ~ (img == [255, 255, 255]).all(axis=-1)
    
    for f in range(total_frames):
        if f <= anim_frames:
            progress = f / anim_frames
        else:
            progress = 1.0
            
        frame = img.copy()
        total_segments = len(path) - 1
        current_len = progress * total_segments
        
        full_segments = int(current_len)
        fraction = current_len - full_segments
        
        # Draw full segments
        for i in range(full_segments):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            pt1 = (centers[c1], centers[r1])
            pt2 = (centers[c2], centers[r2])
            cv2.line(frame, pt1, pt2, (50, 200, 50), thickness=20, lineType=cv2.LINE_AA)
            
        # Draw partial segment
        if full_segments < total_segments and fraction > 0:
            r1, c1 = path[full_segments]
            r2, c2 = path[full_segments+1]
            pt1 = (centers[c1], centers[r1])
            pt2 = (centers[c2], centers[r2])
            
            # Interpolate
            pt_interp = (
                int(pt1[0] + (pt2[0] - pt1[0]) * fraction),
                int(pt1[1] + (pt2[1] - pt1[1]) * fraction)
            )
            cv2.line(frame, pt1, pt_interp, (50, 200, 50), thickness=20, lineType=cv2.LINE_AA)
            
        # Apply mask to restore non-white pixels (walls, markers)
        frame[non_white_mask] = img[non_white_mask]
        
        cv2.imwrite(os.path.join(frames_dir, f'frame_{f:04d}.png'), frame)

    # Encode video
    os.makedirs('/app/output', exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
