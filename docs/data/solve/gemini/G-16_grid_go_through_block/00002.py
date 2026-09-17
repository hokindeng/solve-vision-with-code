import os
import cv2
import numpy as np
import imageio

def prepare_assets(img_path):
    img = cv2.imread(img_path)
    
    cx, cy = 372, 186
    r = 24
    agent_patch = img[cy-r:cy+r, cx-r:cx+r].copy()
    
    # Extract agent
    mask = np.zeros(agent_patch.shape[:2], dtype=np.uint8)
    green = np.array([50, 200, 50])
    for y in range(agent_patch.shape[0]):
        for x in range(agent_patch.shape[1]):
            if not np.array_equal(agent_patch[y, x], green):
                mask[y, x] = 255
                
    agent_rgba = cv2.cvtColor(agent_patch, cv2.COLOR_BGR2BGRA)
    agent_rgba[:, :, 3] = mask
    
    # Clean bg
    clean_bg = img.copy()
    for y in range(cy-r, cy+r):
        for x in range(cx-r, cx+r):
            if not np.array_equal(clean_bg[y, x], green):
                clean_bg[y, x] = green

    return clean_bg, agent_rgba

def draw_agent(bg, agent_rgba, cx, cy):
    frame = bg.copy()
    r = agent_rgba.shape[0] // 2
    
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    
    y1, y2 = cy_int - r, cy_int + r
    x1, x2 = cx_int - r, cx_int + r
    
    alpha = agent_rgba[:, :, 3] / 255.0
    for c in range(3):
        frame[y1:y2, x1:x2, c] = \
            frame[y1:y2, x1:x2, c] * (1 - alpha) + \
            agent_rgba[:, :, c] * alpha
            
    return frame

def main():
    img_path = '/app/first_frame.png'
    clean_bg, agent_rgba = prepare_assets(img_path)
    
    # Targets: Start (3, 1) -> Purple (0, 0) -> Brown (0, 8) -> Pink (5, 8) -> Blue (7, 9) -> Yellow (2, 5) -> Red (2, 7)
    segments = [
        ((3, 1), (0, 0)),
        ((0, 0), (0, 8)),
        ((0, 8), (5, 8)),
        ((5, 8), (7, 9)),
        ((7, 9), (2, 5)),
        ((2, 5), (2, 7))
    ]

    waypoints = [(3, 1)]
    for start, end in segments:
        x1, y1 = start
        x2, y2 = end
        if x1 != x2:
            waypoints.append((x2, y1))
        if y1 != y2:
            waypoints.append((x2, y2))
            
    # Convert grid points to pixels
    # Grid formula: cx = 93 + col * 93, cy = 93 + row * 93
    pixel_points = np.array([(93 + x*93, 93 + y*93) for x, y in waypoints], dtype=float)
    
    # Calculate cumulative distance
    distances = np.sqrt(np.sum(np.diff(pixel_points, axis=0)**2, axis=1))
    cumulative = np.insert(np.cumsum(distances), 0, 0)
    total_dist = cumulative[-1]
    
    num_frames = 134
    frames = []
    
    for i in range(num_frames):
        dist_along = total_dist * i / (num_frames - 1)
        
        # Find segment
        pos = pixel_points[-1] # fallback to last
        for j in range(len(cumulative) - 1):
            if cumulative[j] <= dist_along <= cumulative[j+1] + 1e-5:
                segment_length = distances[j]
                if segment_length > 0:
                    t = (dist_along - cumulative[j]) / segment_length
                    pos = pixel_points[j] * (1 - t) + pixel_points[j+1] * t
                else:
                    pos = pixel_points[j]
                break
                
        frame_bgr = draw_agent(clean_bg, agent_rgba, pos[0], pos[1])
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', macro_block_size=None, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    print(f"Generated {len(frames)} frames video at /app/output/video.mp4")

if __name__ == '__main__':
    main()
