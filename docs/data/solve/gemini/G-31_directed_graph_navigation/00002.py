import cv2
import numpy as np
import imageio

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Erase the agent from the background
    mask_blue = np.all(img == [255, 0, 0], axis=-1)
    mask_dblue = np.all(img == [139, 0, 0], axis=-1)
    mask = mask_blue | mask_dblue
    
    bg = img.copy()
    bg[mask] = [0, 128, 0] # Fill with Green node color
    
    # 2. Extract agent RGBA
    ys, xs = np.where(mask)
    min_y, max_y = np.min(ys), np.max(ys)
    min_x, max_x = np.min(xs), np.max(xs)
    
    agent_h = max_y - min_y + 1
    agent_w = max_x - min_x + 1
    agent_rgba = np.zeros((agent_h, agent_w, 4), dtype=np.uint8)
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if mask[y, x]:
                agent_rgba[y - min_y, x - min_x, :3] = img[y, x]
                agent_rgba[y - min_y, x - min_x, 3] = 255
                
    agent_bbox_cx = (min_x + max_x) / 2.0
    agent_bbox_cy = (min_y + max_y) / 2.0
    
    # True Centers of nodes (based on bbox)
    # Green: 708, 750
    # W2: 304, 744
    # Red: 204, 270
    path = [
        np.array([708.0, 750.0]),
        np.array([304.0, 744.0]),
        np.array([204.0, 270.0])
    ]
    
    num_frames = 30
    dist1 = np.linalg.norm(path[1] - path[0])
    dist2 = np.linalg.norm(path[2] - path[1])
    total_dist = dist1 + dist2
    
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        d = t * total_dist
        
        if d <= dist1:
            if dist1 == 0:
                pos = path[0]
            else:
                pos = path[0] + (path[1] - path[0]) * (d / dist1)
        else:
            d2 = d - dist1
            if dist2 == 0:
                pos = path[1]
            else:
                pos = path[1] + (path[2] - path[1]) * (d2 / dist2)
                
        frame = bg.copy()
        
        # Determine top-left of the agent so its bbox center aligns with pos
        top_left_x = int(round(pos[0] - (agent_bbox_cx - min_x)))
        top_left_y = int(round(pos[1] - (agent_bbox_cy - min_y)))
        
        # Overlay agent
        for y in range(agent_h):
            for x in range(agent_w):
                if agent_rgba[y, x, 3] > 0:
                    fy = top_left_y + y
                    fx = top_left_x + x
                    if 0 <= fy < frame.shape[0] and 0 <= fx < frame.shape[1]:
                        frame[fy, fx] = agent_rgba[y, x, :3]
                        
        # Ensure we output RGB since imageio expects RGB for mp4
        # Wait, cv2 uses BGR, but we read as BGR and are putting BGR into frames.
        # We need to convert BGR to RGB for imageio.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # Ensure first frame is EXACTLY first_frame.png (without rounding errors)
    frames[0] = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
