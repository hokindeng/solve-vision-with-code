import cv2
import numpy as np
import os
import subprocess
from collections import deque

def solve_maze():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Parse the grid robustly
    grid = np.zeros((15, 15), dtype=int)
    
    # Pre-calculate masks to find markers
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    for row in range(15):
        for col in range(15):
            y0, y1 = int(row * 1024 / 15), int((row + 1) * 1024 / 15)
            x0, x1 = int(col * 1024 / 15), int((col + 1) * 1024 / 15)
            
            # Check markers first
            cell_green_pixels = np.count_nonzero(mask_green[y0:y1, x0:x1])
            cell_red_pixels = np.count_nonzero(mask_red[y0:y1, x0:x1])
            
            if cell_green_pixels > 100:
                grid[row, col] = 2
            elif cell_red_pixels > 100:
                grid[row, col] = 3
            else:
                # Differentiate white pathway from black wall
                cell = img[y0:y1, x0:x1]
                mean_color = np.mean(cell, axis=(0, 1))
                b, g, r = mean_color
                if r > 128 and g > 128 and b > 128:
                    grid[row, col] = 1
                else:
                    grid[row, col] = 0

    start = None
    end = None
    for r in range(15):
        for c in range(15):
            if grid[r, c] == 2:
                start = (r, c)
            elif grid[r, c] == 3:
                end = (r, c)
                
    if start is None or end is None:
        raise ValueError("Start or end not found!")

    # 2. Run BFS to find the path
    queue = deque([(start, [start])])
    visited = set([start])
    path = []

    while queue:
        (r, c), p = queue.popleft()
        if (r, c) == end:
            path = p
            break
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 15 and 0 <= nc < 15 and grid[nr, nc] != 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), p + [(nr, nc)]))
                
    if not path:
        raise ValueError("No path found!")

    # 3. Use masks to keep original markers on top of the path
    mask = (mask_green > 0) | (mask_red > 0)
    original_pixels = img[mask].copy()

    # 4. Render frames
    os.makedirs('/tmp/frames', exist_ok=True)
    
    total_frames = 71
    num_segments = len(path) - 1
    
    thickness = 20
    color = (50, 200, 50) # Matching green
    
    for i in range(total_frames):
        frame = img.copy()
        
        progress = i / (total_frames - 1)
        current_distance = progress * num_segments
        segment_index = int(current_distance)
        segment_progress = current_distance - segment_index
        
        if segment_index >= num_segments:
            segment_index = num_segments - 1
            segment_progress = 1.0
            
        # Draw full segments
        for j in range(segment_index):
            r1, c1 = path[j]
            r2, c2 = path[j+1]
            x1 = int((c1 + 0.5) * 1024 / 15)
            y1 = int((r1 + 0.5) * 1024 / 15)
            x2 = int((c2 + 0.5) * 1024 / 15)
            y2 = int((r2 + 0.5) * 1024 / 15)
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            cv2.circle(frame, (x1, y1), thickness // 2, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x2, y2), thickness // 2, color, -1, cv2.LINE_AA)
            
        # Draw partial segment
        if segment_progress > 0 or segment_index > 0:
            r1, c1 = path[segment_index]
            r2, c2 = path[segment_index+1]
            x1 = int((c1 + 0.5) * 1024 / 15)
            y1 = int((r1 + 0.5) * 1024 / 15)
            x2 = int((c2 + 0.5) * 1024 / 15)
            y2 = int((r2 + 0.5) * 1024 / 15)
            
            px = int(x1 + (x2 - x1) * segment_progress)
            py = int(y1 + (y2 - y1) * segment_progress)
            
            if segment_progress > 0:
                cv2.line(frame, (x1, y1), (px, py), color, thickness, cv2.LINE_AA)
                cv2.circle(frame, (px, py), thickness // 2, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x1, y1), thickness // 2, color, -1, cv2.LINE_AA)
            
        # Restore markers on top
        frame[mask] = original_pixels
        
        cv2.imwrite(f'/tmp/frames/frame_{i:03d}.png', frame)
        
    os.makedirs('/app/output', exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/tmp/frames/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    solve_maze()
