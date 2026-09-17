import cv2
import numpy as np
import imageio
import os

def main():
    # Load first frame
    first_frame = cv2.imread('/app/first_frame.png')
    
    # 1. Create clean background (agent removed)
    clean_bg = first_frame.copy()
    agent_y = 49 + 2*93 + 21
    agent_x = 49 + 9*93 + 21
    # Replace the 47x47 agent bounding box with the green background color
    clean_bg[agent_y:agent_y+47, agent_x:agent_x+47] = [50, 200, 50]
    
    # 2. Extract agent and mask
    agent_crop = first_frame[agent_y:agent_y+47, agent_x:agent_x+47].copy()
    mask = np.any(agent_crop != [50, 200, 50], axis=-1)
    
    # 3. Define the path sequence (shortest path through targets)
    nodes = [
        (2, 9), (2, 8), (2, 7),                 # Start (2,9) to Pink (2,7)
        (2, 6), (1, 6),                         # Pink (2,7) to Purple (1,6)
        (1, 5), (1, 4), (1, 3), (2, 3), (3, 3), # Purple (1,6) to Yellow (3,3)
        (4, 3), (5, 3), (6, 3), (7, 3),         # Yellow (3,3) to Brown (7,3)
        (7, 2), (7, 1), (7, 0),                 # Brown (7,3) to Blue (7,0)
        (8, 0), (8, 1)                          # Blue (7,0) to Red (8,1)
    ]
    
    total_steps = len(nodes) - 1
    num_frames = 82
    
    # Ensure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    # Initialize video writer
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for f in range(num_frames):
        # Progress from 0.0 to 1.0 inclusive
        t = f / (num_frames - 1)
        progress = t * total_steps
        
        idx = int(progress)
        if idx == total_steps:
            idx = total_steps - 1
            frac = 1.0
        else:
            frac = progress - idx
            
        # Interpolate fractional row and col positions
        r = nodes[idx][0] * (1 - frac) + nodes[idx+1][0] * frac
        c = nodes[idx][1] * (1 - frac) + nodes[idx+1][1] * frac
        
        # Convert to pixel coordinates
        y_int = int(round(49 + r * 93 + 21))
        x_int = int(round(49 + c * 93 + 21))
        
        # Paste agent onto clean background
        frame = clean_bg.copy()
        frame[y_int:y_int+47, x_int:x_int+47][mask] = agent_crop[mask]
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()
    print("Video generated successfully.")

if __name__ == '__main__':
    main()
