import cv2
import numpy as np
import imageio
import os

def solve():
    # 1. Load original first frame
    img_path = '/app/first_frame.png'
    first_frame = cv2.imread(img_path)
    if first_frame is None:
        raise ValueError(f"Could not load {img_path}")

    # 2. Precompute grid boundaries
    x_b = [int(round(47 + 93.2 * i)) for i in range(11)]
    y_b = [int(round(47 + 93.2 * i)) for i in range(11)]

    # 3. Extract agent sprite and create clean background
    # Agent is at start cell (7,0)
    r, c = 7, 0
    cell = first_frame[y_b[r]:y_b[r+1], x_b[c]:x_b[c+1]]
    
    agent_color1 = np.array([0, 150, 200])
    agent_color2 = np.array([0, 200, 255])
    mask1 = np.all(cell == agent_color1, axis=-1)
    mask2 = np.all(cell == agent_color2, axis=-1)
    agent_mask_2d = mask1 | mask2

    # Find bounding box of agent
    rows = np.any(agent_mask_2d, axis=1)
    cols = np.any(agent_mask_2d, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Extract sprite
    agent_sprite = cell[rmin:rmax+1, cmin:cmax+1].copy()
    agent_sprite_mask = agent_mask_2d[rmin:rmax+1, cmin:cmax+1].copy()

    # Create clean background
    clean_bg = first_frame.copy()
    clean_cell = clean_bg[y_b[r]:y_b[r+1], x_b[c]:x_b[c+1]]
    clean_cell[agent_mask_2d] = [255, 100, 0] # Replace with blue background

    # 4. Define path (from BFS)
    path = [(7, 0), (6, 0), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0), 
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
    
    # 5. Calculate waypoints
    waypoints = []
    for pr, pc in path:
        waypoints.append((x_b[pc] + cmin, y_b[pr] + rmin))

    # 6. Render video
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    total_frames = 58
    total_segments = len(path) - 1 # 12

    h, w = agent_sprite.shape[:2]

    for f in range(total_frames):
        t = f / float(total_frames - 1) # 0.0 to 1.0
        current_dist = t * total_segments
        idx = int(current_dist)
        frac = current_dist - idx

        if idx >= total_segments:
            x, y = waypoints[-1]
        else:
            x = waypoints[idx][0] + frac * (waypoints[idx+1][0] - waypoints[idx][0])
            y = waypoints[idx][1] + frac * (waypoints[idx+1][1] - waypoints[idx][1])

        x = int(round(x))
        y = int(round(y))

        frame = clean_bg.copy()
        roi = frame[y:y+h, x:x+w]
        
        # In case the region exceeds image boundaries (should not happen here), safely paste
        roi[agent_sprite_mask] = agent_sprite[agent_sprite_mask]

        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()
    print("Video saved to /app/output/video.mp4")

if __name__ == '__main__':
    solve()
