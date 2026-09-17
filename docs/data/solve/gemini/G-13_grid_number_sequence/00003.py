import cv2
import numpy as np
import os
import subprocess

def create_video():
    # Load first frame
    first_frame = cv2.imread('/app/first_frame.png')
    
    # 1. Extract agent patch (the orange circle)
    orange_mask = np.all(first_frame == [0, 165, 255], axis=-1)
    rows, cols = np.where(orange_mask)
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    
    agent_crop = first_frame[min_r:max_r+1, min_c:max_c+1].copy()
    agent_mask = orange_mask[min_r:max_r+1, min_c:max_c+1]
    
    # Center offset of the agent
    off_y = (max_r - min_r) // 2
    off_x = (max_c - min_c) // 2
    
    # 2. Create clean background
    # The start cell is at r=3, c=5. We fill it with green to erase the agent.
    clean_bg = first_frame.copy()
    # Cell bounds for r=3, c=5
    y1, y2 = 102*3 + 2, 102*3 + 102  # 308 to 408
    x1, x2 = 102*5 + 2, 102*5 + 102  # 512 to 612
    clean_bg[y1:y2, x1:x2] = [0, 255, 0]
    
    # 3. Define waypoints
    # Waypoints in (x, y)
    waypoints = [
        (561, 357),  # Start: (3, 5)
        (51, 357),   # T1: (3, 0)
        (357, 357),  # intermediate: (3, 3)
        (357, 561),  # T2: (5, 3)
        (51, 561),   # T3: (5, 0)
        (153, 561),  # intermediate: (5, 1)
        (153, 255),  # intermediate: (2, 1)
        (765, 255)   # End: (2, 7)
    ]
    
    # Calculate distances
    dists = []
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i+1]
        dist = np.hypot(x2 - x1, y2 - y1)
        dists.append(dist)
    
    total_dist = sum(dists)
    num_frames = 89
    
    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/output/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    for i in range(num_frames):
        # Calculate current distance
        curr_dist = i * total_dist / (num_frames - 1)
        
        # Find which segment we are on
        accum_dist = 0
        cx, cy = waypoints[-1] # Default to end
        
        for j, d in enumerate(dists):
            if accum_dist + d >= curr_dist - 1e-5: # subtract small epsilon for float precision
                # We are on segment j
                rem = curr_dist - accum_dist
                ratio = rem / d if d > 0 else 0
                x1, y1 = waypoints[j]
                x2, y2 = waypoints[j+1]
                cx = x1 + (x2 - x1) * ratio
                cy = y1 + (y2 - y1) * ratio
                break
            accum_dist += d
            
        cx, cy = int(round(cx)), int(round(cy))
        
        # Draw on frame
        frame = clean_bg.copy()
        
        # Paste agent
        y_min = cy - off_y
        y_max = cy + (max_r - min_r) - off_y + 1
        x_min = cx - off_x
        x_max = cx + (max_c - min_c) - off_x + 1
        
        frame[y_min:y_max, x_min:x_max][agent_mask] = agent_crop[agent_mask]
        
        # Save frame
        cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', frame)
        
    # Generate video
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)

if __name__ == '__main__':
    create_video()
