import cv2
import numpy as np
import math
import os
import imageio

def solve():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    
    # Create output dir if needed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path}")
        
    # Extract agent
    blue = np.array([255, 0, 0])
    dark_blue = np.array([139, 0, 0])
    
    mask1 = cv2.inRange(img, blue, blue)
    mask2 = cv2.inRange(img, dark_blue, dark_blue)
    agent_mask = cv2.bitwise_or(mask1, mask2)
    
    coords = cv2.findNonZero(agent_mask)
    x, y, w, h = cv2.boundingRect(coords)
    
    agent_rgba = np.zeros((h, w, 4), dtype=np.float32)
    agent_bgr = img[y:y+h, x:x+w].astype(np.float32)
    agent_mask_crop = agent_mask[y:y+h, x:x+w].astype(np.float32) / 255.0
    
    agent_rgba[..., :3] = agent_bgr
    agent_rgba[..., 3] = agent_mask_crop
    
    # Create clean background (fill agent position with green node color)
    clean_img = img.copy()
    clean_img[agent_mask > 0] = [0, 128, 0]
    
    # Path coordinates (determined by analysis)
    # Start: Green Node 4
    # Mid: Node 0
    # End: Red Node 2
    p0 = (458, 268)
    p1 = (220, 706)
    p2 = (566, 558)
    
    d1 = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
    d2 = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    total_d = d1 + d2
    
    num_frames = 30
    frames = []
    
    alpha_agent = agent_rgba[:, :, 3]
    alpha_bg = 1.0 - alpha_agent
    
    for i in range(num_frames):
        if i == 0:
            # First frame exactly as input
            frames.append(img.copy())
            continue
            
        dist = i * (total_d / (num_frames - 1))
        if dist <= d1:
            ratio = dist / d1
            cx = p0[0] + ratio * (p1[0] - p0[0])
            cy = p0[1] + ratio * (p1[1] - p0[1])
        else:
            ratio = (dist - d1) / d2
            ratio = min(1.0, ratio)
            cx = p1[0] + ratio * (p2[0] - p1[0])
            cy = p1[1] + ratio * (p2[1] - p1[1])
            
        # Top-left of the agent
        # From earlier analysis, agent center offset from bounding box top-left is exactly (30, 30)
        # Bounding box is 61x61, center is at 30.5, 30.5. 
        top_x = int(round(cx - 30.5))
        top_y = int(round(cy - 30.5))
        
        frame = clean_img.copy().astype(np.float32)
        
        # Alpha blending
        for c in range(3):
            frame[top_y:top_y+h, top_x:top_x+w, c] = (
                alpha_agent * agent_rgba[:, :, c] + 
                alpha_bg * frame[top_y:top_y+h, top_x:top_x+w, c]
            )
            
        frames.append(frame.astype(np.uint8))
        
    # Write video
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        # Convert BGR to RGB for imageio
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()

if __name__ == '__main__':
    solve()
