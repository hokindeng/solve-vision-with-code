import numpy as np
from PIL import Image
import cv2
import imageio
import os
from collections import deque

def main():
    img = Image.open("/app/first_frame.png").convert("RGB")
    base_arr = np.array(img)

    CELL_SIZE = 1024 / 15.0

    # Extract grid and find start/end
    grid = []
    start = None
    end = None
    max_green = 0
    max_red = 0

    for r in range(15):
        row_chars = []
        for c in range(15):
            y0, y1 = int(r * CELL_SIZE), int((r + 1) * CELL_SIZE)
            x0, x1 = int(c * CELL_SIZE), int((c + 1) * CELL_SIZE)
            patch = base_arr[y0:y1, x0:x1]
            
            white = ((patch[:,:,0] == 255) & (patch[:,:,1] == 255) & (patch[:,:,2] == 255)).sum()
            green = ((patch[:,:,0] == 50) & (patch[:,:,1] == 200) & (patch[:,:,2] == 50)).sum()
            red = ((patch[:,:,0] == 200) & (patch[:,:,1] == 50) & (patch[:,:,2] == 50)).sum()
            
            total_path = white + green + red
            if total_path > (y1-y0)*(x1-x0) * 0.2:
                row_chars.append(".")
            else:
                row_chars.append("#")
                
            if green > max_green:
                max_green = green
                start = (r, c)
            if red > max_red:
                max_red = red
                end = (r, c)
                
        grid.append(row_chars)

    if not start or not end:
        print("Could not find start or end.")
        return

    # BFS
    q = deque([(start, [start])])
    visited = set([start])
    path = None

    while q:
        curr, p = q.popleft()
        if curr == end:
            path = p
            break
        
        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 15 and 0 <= nc < 15 and grid[nr][nc] == '.':
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), p + [(nr, nc)]))

    if not path:
        print("No path found.")
        return

    def get_center(r, c):
        return (int((c + 0.5) * CELL_SIZE), int((r + 0.5) * CELL_SIZE))

    centers = [get_center(r, c) for r, c in path]

    total_frames = 60
    frames = [base_arr.copy()]
    path_color = (50, 200, 50)
    thickness = 20
    radius = thickness // 2

    # Masks for start and end
    end_y0, end_y1 = int(end[0] * CELL_SIZE), int((end[0] + 1) * CELL_SIZE)
    end_x0, end_x1 = int(end[1] * CELL_SIZE), int((end[1] + 1) * CELL_SIZE)
    end_cell = base_arr[end_y0:end_y1, end_x0:end_x1].copy()
    end_cell_white = (end_cell[:,:,0] == 255) & (end_cell[:,:,1] == 255) & (end_cell[:,:,2] == 255)
    end_cell_mask = ~end_cell_white

    start_y0, start_y1 = int(start[0] * CELL_SIZE), int((start[0] + 1) * CELL_SIZE)
    start_x0, start_x1 = int(start[1] * CELL_SIZE), int((start[1] + 1) * CELL_SIZE)
    start_cell = base_arr[start_y0:start_y1, start_x0:start_x1].copy()
    start_cell_white = (start_cell[:,:,0] == 255) & (start_cell[:,:,1] == 255) & (start_cell[:,:,2] == 255)
    start_cell_mask = ~start_cell_white

    anim_frames = 50
    total_segments = max(1, len(path) - 1)

    for i in range(1, total_frames):
        curr_arr = base_arr.copy()
        
        progress = min(total_segments, (i / anim_frames) * total_segments)
        last_idx = int(progress)
        frac = progress - last_idx
        
        for j in range(last_idx):
            pt1 = centers[j]
            pt2 = centers[j+1]
            cv2.line(curr_arr, pt1, pt2, path_color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(curr_arr, pt1, radius, path_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(curr_arr, pt2, radius, path_color, -1, lineType=cv2.LINE_AA)
            
        if last_idx < len(path) - 1 and frac > 0:
            pt1 = centers[last_idx]
            pt2 = centers[last_idx+1]
            curr_pt = (
                int(pt1[0] + (pt2[0] - pt1[0]) * frac),
                int(pt1[1] + (pt2[1] - pt1[1]) * frac)
            )
            cv2.line(curr_arr, pt1, curr_pt, path_color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(curr_arr, pt1, radius, path_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(curr_arr, curr_pt, radius, path_color, -1, lineType=cv2.LINE_AA)
            
        if progress > 0 and len(centers) > 0:
            cv2.circle(curr_arr, centers[0], radius, path_color, -1, lineType=cv2.LINE_AA)
            
        curr_end_cell = curr_arr[end_y0:end_y1, end_x0:end_x1]
        np.copyto(curr_end_cell, end_cell, where=end_cell_mask[:,:,None])
        
        curr_start_cell = curr_arr[start_y0:start_y1, start_x0:start_x1]
        np.copyto(curr_start_cell, start_cell, where=start_cell_mask[:,:,None])
        
        frames.append(curr_arr)

    os.makedirs("/app/output", exist_ok=True)
    imageio.mimwrite("/app/output/video.mp4", frames, fps=16, codec="libx264", pixelformat="yuv420p")

if __name__ == "__main__":
    main()
