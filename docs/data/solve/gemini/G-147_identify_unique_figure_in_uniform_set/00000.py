import cv2
import numpy as np
import imageio
from scipy.ndimage import label
import math
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not load /app/first_frame.png")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Identify all shapes
    # Background is white [255, 255, 255]
    non_white = np.any(img != [255, 255, 255], axis=-1)
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(non_white, structure=structure)
    
    shapes = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) > 100:  # ignore noise
            w = np.max(xs) - np.min(xs)
            h = np.max(ys) - np.min(ys)
            area = len(ys)
            
            shapes.append({
                'id': i,
                'area': area,
                'w': w,
                'h': h,
                'xs': xs,
                'ys': ys
            })
            
    if not shapes:
        print("No shapes found!")
        return

    # To find the unique shape, we can look at the features
    # Let's create a combined feature vector: area, w, h
    # We will compute distances between all pairs of shapes to find the outlier
    
    # Normalize features
    areas = [s['area'] for s in shapes]
    ws = [s['w'] for s in shapes]
    hs = [s['h'] for s in shapes]
    
    max_area = max(areas) if max(areas) > 0 else 1
    max_w = max(ws) if max(ws) > 0 else 1
    max_h = max(hs) if max(hs) > 0 else 1
    
    # Find the shape with the maximum average distance to all other shapes
    max_dist_sum = -1
    unique_shape_idx = 0
    
    for i in range(len(shapes)):
        dist_sum = 0
        for j in range(len(shapes)):
            if i == j:
                continue
            d_area = (shapes[i]['area'] - shapes[j]['area']) / max_area
            d_w = (shapes[i]['w'] - shapes[j]['w']) / max_w
            d_h = (shapes[i]['h'] - shapes[j]['h']) / max_h
            dist = math.sqrt(d_area**2 + d_w**2 + d_h**2)
            dist_sum += dist
        if dist_sum > max_dist_sum:
            max_dist_sum = dist_sum
            unique_shape_idx = i
            
    unique_shape = shapes[unique_shape_idx]
        
    print(f"Unique shape chosen: area={unique_shape['area']}, w={unique_shape['w']}, h={unique_shape['h']}")
    
    # Calculate bounding box center and radius
    min_x = np.min(unique_shape['xs'])
    max_x = np.max(unique_shape['xs'])
    min_y = np.min(unique_shape['ys'])
    max_y = np.max(unique_shape['ys'])
    
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2
    w = max_x - min_x
    h = max_y - min_y
    
    # Radius slightly larger than the shape
    radius = int(math.sqrt(w**2 + h**2) / 2) + 15
    
    os.makedirs('/app/output', exist_ok=True)
    num_frames = 60
    fps = 16
    
    # Writer using libx264, yuv420p
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=fps, 
        macro_block_size=1, 
        format='FFMPEG', 
        codec='libx264', 
        pixelformat='yuv420p'
    )
    
    for f in range(num_frames):
        frame = img_rgb.copy()
        
        progress = f / (num_frames - 1) # from 0 to 1
        current_angle = progress * 360
        
        if current_angle > 0:
            # Draw red circle (R,G,B) = (255, 0, 0)
            # The startAngle is -90 (top), endAngle is -90 + current_angle
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + current_angle, (255, 0, 0), 4, cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()
    print("Video saved to /app/output/video.mp4")

if __name__ == '__main__':
    solve()
