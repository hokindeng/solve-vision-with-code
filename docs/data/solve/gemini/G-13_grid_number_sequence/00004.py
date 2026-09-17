import cv2
import numpy as np
import os
import subprocess

def create_video():
    img_path = '/app/first_frame.png'
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'video.mp4')
    
    first_frame = cv2.imread(img_path)
    
    # 1. Create pristine background
    bg = first_frame.copy()
    agent_color = [0, 165, 255]
    mask = np.all(bg == agent_color, axis=-1)
    bg[mask] = [0, 255, 0] # Fill with green
    
    # 2. Get agent relative pixels
    y_coords, x_coords = np.where(mask)
    cy_agent = 969 # Center of cell (9,4)
    cx_agent = 459
    dy_agent = y_coords - cy_agent
    dx_agent = x_coords - cx_agent
    
    # 3. Define waypoints (x_idx, y_idx)
    waypoints_idx = [
        (4, 9),
        (8, 9), (8, 8),
        (1, 8), (1, 5),
        (4, 5), (4, 6),
        (1, 6), (1, 4)
    ]
    
    # Convert to pixel coordinates
    pix_points = [(51 + 102*x, 51 + 102*y) for x, y in waypoints_idx]
    
    # Calculate total distance and distances at waypoints
    total_dist = 0
    dists = [0]
    for i in range(1, len(pix_points)):
        x1, y1 = pix_points[i-1]
        x2, y2 = pix_points[i]
        dist = abs(x2 - x1) + abs(y2 - y1)
        total_dist += dist
        dists.append(total_dist)
        
    frames_count = 92
    
    # Initialize video writer
    # Using ffmpeg via subprocess for precise encoding control (H.264, yuv420p)
    # But we can also use cv2.VideoWriter and then convert, or imageio.
    # We will use cv2.VideoWriter and then convert with ffmpeg to ensure it's correct.
    
    temp_mp4 = '/app/temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_mp4, fourcc, 16.0, (1024, 1024))
    
    for i in range(frames_count):
        d = total_dist * i / (frames_count - 1)
        # find segment
        for j in range(1, len(dists)):
            if d <= dists[j] + 1e-5:
                segment_d = dists[j] - dists[j-1]
                if segment_d == 0:
                    t = 0
                else:
                    t = (d - dists[j-1]) / segment_d
                
                x1, y1 = pix_points[j-1]
                x2, y2 = pix_points[j]
                
                x = x1 + (x2 - x1) * t
                y = y1 + (y2 - y1) * t
                break
                
        # Draw frame
        frame = bg.copy()
        
        cx, cy = int(round(x)), int(round(y))
        
        # Paste agent
        # We can just do array indexing if it fits, but since agent is always within the frame
        # we can just use advanced indexing
        py = cy + dy_agent
        px = cx + dx_agent
        
        # Ensure within bounds (though it should be)
        valid = (py >= 0) & (py < 1024) & (px >= 0) & (px < 1024)
        frame[py[valid], px[valid]] = agent_color
        
        out.write(frame)
        
    out.release()
    
    # Convert to required format using ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_mp4,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        out_path
    ], check=True)
    os.remove(temp_mp4)
    print("Video generated successfully at", out_path)

if __name__ == "__main__":
    create_video()
