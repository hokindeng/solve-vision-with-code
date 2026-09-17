import cv2
import numpy as np
import imageio
import math
import os

def solve():
    # Load first frame
    image = cv2.imread('/app/first_frame.png')
    
    # Agent mask (Blue and Dark Blue)
    agent_mask = np.logical_or(
        np.all(image == [255, 0, 0], axis=-1),
        np.all(image == [139, 0, 0], axis=-1)
    )
    
    # Clean frame by removing the agent (fill with Green)
    clean_frame = image.copy()
    clean_frame[agent_mask] = [0, 128, 0]
    
    # Get original agent pixels and coordinates
    y_orig, x_orig = np.where(agent_mask)
    colors = image[y_orig, x_orig]
    
    # Node centers
    G = (806, 468)
    W2 = (476, 754)
    R = (586, 216)
    
    # Offsets of agent pixels relative to G
    dx = x_orig - G[0]
    dy = y_orig - G[1]
    
    # Distances
    dist1 = math.hypot(W2[0] - G[0], W2[1] - G[1])
    dist2 = math.hypot(R[0] - W2[0], R[1] - W2[1])
    total_dist = dist1 + dist2
    
    # Generate frames
    frames = []
    num_frames = 30
    
    for i in range(num_frames):
        t = i / float(num_frames - 1)
        d = t * total_dist
        
        if d <= dist1:
            segment_t = d / dist1 if dist1 > 0 else 0
            cx = G[0] + segment_t * (W2[0] - G[0])
            cy = G[1] + segment_t * (W2[1] - G[1])
        else:
            segment_t = (d - dist1) / dist2 if dist2 > 0 else 0
            cx = W2[0] + segment_t * (R[0] - W2[0])
            cy = W2[1] + segment_t * (R[1] - W2[1])
            
        new_x = np.round(cx + dx).astype(int)
        new_y = np.round(cy + dy).astype(int)
        
        frame = clean_frame.copy()
        
        valid = (new_x >= 0) & (new_x < 1024) & (new_y >= 0) & (new_y < 1024)
        frame[new_y[valid], new_x[valid]] = colors[valid]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    
    # Write to video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    solve()
