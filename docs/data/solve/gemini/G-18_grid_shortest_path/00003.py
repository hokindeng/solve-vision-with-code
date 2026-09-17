import cv2
import numpy as np
import os
import subprocess

def solve():
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    
    # Create clean background by replacing red with orange
    bg_img = img.copy()
    red_mask = np.all(bg_img == [0, 0, 255], axis=-1)
    bg_img[red_mask] = [0, 165, 255]
    
    # Extract agent patch and mask
    agent_coords = np.argwhere(red_mask)
    rmin, cmin = agent_coords.min(axis=0)
    rmax, cmax = agent_coords.max(axis=0)
    
    agent_patch = img[rmin:rmax+1, cmin:cmax+1].copy()
    agent_mask = np.all(agent_patch == [0, 0, 255], axis=-1)
    
    agent_h, agent_w = agent_patch.shape[:2]
    cy_offset = agent_h // 2
    cx_offset = agent_w // 2
    
    # Define shortest path
    # Start: (6, 5), End: (5, 9)
    # Path moves UP from (6,5) to (5,5), then RIGHT from (5,5) to (5,9).
    path_cells = [(6, 5), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9)]
    
    centers = [(51 + c*102, 51 + r*102) for r, c in path_cells]
    
    frames_count = 45
    total_dist = (len(path_cells) - 1) * 102
    
    os.makedirs('/app/frames', exist_ok=True)
    os.makedirs('/app/output', exist_ok=True)
    
    for i in range(frames_count):
        d = i / (frames_count - 1) * total_dist
        seg = int(d // 102)
        if seg >= len(path_cells) - 1:
            seg = len(path_cells) - 2
            rem = 102
        else:
            rem = d % 102
            
        p1 = np.array(centers[seg], dtype=np.float64)
        p2 = np.array(centers[seg+1], dtype=np.float64)
        
        pos = p1 + (p2 - p1) * (rem / 102.0)
        x_c, y_c = pos
        
        y0 = int(round(y_c)) - cy_offset
        x0 = int(round(x_c)) - cx_offset
        
        frame = bg_img.copy()
        
        # Paste agent
        frame_region = frame[y0:y0+agent_h, x0:x0+agent_w]
        frame_region[agent_mask] = [0, 0, 255]
        
        # Ensure exact match for the very first frame
        if i == 0:
            frame = img.copy()
            
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)
        
    # Generate video
    cmd = [
        'ffmpeg', '-y', 
        '-framerate', '16',
        '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    solve()
