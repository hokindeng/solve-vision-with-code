import cv2
import numpy as np
import imageio
from collections import deque
import os

def solve():
    # Load image
    img = cv2.imread('/app/first_frame.png')
    
    # Grid bounding lines (determined via analysis)
    grid_lines = [48, 141, 234, 327, 420, 513, 606, 699, 792, 885, 978]

    # Extract the agent and its mask
    # The start cell is at row 0, col 3, which is y in 49:141, x in 328:420
    cell = img[49:141, 328:420]
    
    # Find non-blue pixels in this cell
    non_blue = np.any(cell != [255, 100, 0], axis=-1).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_blue)
    
    # Find the agent component by finding the largest area that is not the background (0)
    agent_label = -1
    max_area = -1
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            agent_label = i
            
    agent_mask = (labels == agent_label)
    
    # The agent bounding box inside the cell is 21,21 to 68,68
    agent_patch = cell[21:21+47, 21:21+47].copy()
    mask_patch = agent_mask[21:21+47, 21:21+47]

    # Clean the background image (remove the agent from the start cell)
    cleaned_img = img.copy()
    start_y1, start_x1 = 49, 328
    for y in range(47):
        for x in range(47):
            if mask_patch[y, x]:
                cleaned_img[start_y1 + 21 + y, start_x1 + 21 + x] = [255, 100, 0]

    # Define the 10x10 grid with obstacles and start/end points
    grid = [
        ". . . S . . . . . .".split(),
        ". . . . . . . . . .".split(),
        ". . . . . . . X . .".split(),
        ". . . . . . . . . .".split(),
        ". . . . . X . . . .".split(),
        ". . . . . . . . . .".split(),
        ". X . . . . . . . X".split(),
        ". . . . . . . . . .".split(),
        ". . . X . . . . E .".split(),
        ". . . . X . . . . .".split()
    ]

    start = (0, 3)
    end = (8, 8)

    # BFS to find the shortest path
    queue = deque([(start, [start])])
    visited = set([start])
    path = []

    while queue:
        (r, c), p = queue.popleft()
        if (r, c) == end:
            path = p
            break
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and (nr, nc) not in visited:
                if grid[nr][nc] != 'X':
                    visited.add((nr, nc))
                    queue.append(((nr, nc), p + [(nr, nc)]))

    if not path:
        print("Error: No path found!")
        return

    # Calculate precise coordinates for placing the agent
    def get_pos(r, c):
        return grid_lines[r] + 22, grid_lines[c] + 22

    def draw_agent(base_img, y, x):
        frame = base_img.copy()
        y, x = int(round(y)), int(round(x))
        for dy in range(47):
            for dx in range(47):
                if mask_patch[dy, dx]:
                    frame[y + dy, x + dx] = agent_patch[dy, dx]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frames = []

    # Frame 0 to 4 (5 frames) - initial wait
    start_y, start_x = get_pos(path[0][0], path[0][1])
    for _ in range(5):
        frames.append(draw_agent(cleaned_img, start_y, start_x))

    # Animate transitions (4 frames per step)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        y1, x1 = get_pos(r1, c1)
        y2, x2 = get_pos(r2, c2)
        
        for step in range(1, 5):
            t = step / 4.0
            cur_y = y1 + (y2 - y1) * t
            cur_x = x1 + (x2 - x1) * t
            frames.append(draw_agent(cleaned_img, cur_y, cur_x))

    # Frames 57 to 61 (5 frames) - final wait
    end_y, end_x = get_pos(path[-1][0], path[-1][1])
    for _ in range(5):
        frames.append(draw_agent(cleaned_img, end_y, end_x))
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    solve()
