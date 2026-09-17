import cv2
import numpy as np
import math
import imageio
import os

def find_tangent_point(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # The background is white (255), shapes are colored.
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # Find local maxima to locate circle centers
    dilated = cv2.dilate(dist_transform, np.ones((30, 30)))
    local_max = (dist_transform == dilated) & (dist_transform > 20)
    y_coords, x_coords = np.where(local_max)

    circles = []
    for y, x in zip(y_coords, x_coords):
        r = dist_transform[y, x]
        circles.append({'x': x, 'y': y, 'r': r})

    best_pair = None
    min_diff = float('inf')

    # Find the two touching circles
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            c1, c2 = circles[i], circles[j]
            dist = math.hypot(c1['x'] - c2['x'], c1['y'] - c2['y'])
            sum_r = c1['r'] + c2['r']
            diff = abs(dist - sum_r)
            if diff < min_diff:
                min_diff = diff
                best_pair = (c1, c2)

    c1, c2 = best_pair
    # Interpolate to find the exact tangent point
    ratio = c1['r'] / (c1['r'] + c2['r'])
    tx = c1['x'] + ratio * (c2['x'] - c1['x'])
    ty = c1['y'] + ratio * (c2['y'] - c1['y'])
    
    return int(round(tx)), int(round(ty))

def make_video():
    first_frame_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 1. Find tangent point
    tx, ty = find_tangent_point(first_frame_path)
    center = (tx, ty)
    
    # 2. Setup video generation
    first_frame = cv2.imread(first_frame_path)
    
    num_frames = 60
    fps = 16
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    radius = 40
    thickness = 6
    color = (0, 0, 0) # Black
    
    # 3. Generate frames
    for i in range(num_frames):
        progress = i / (num_frames - 1)
        
        frame = first_frame.copy()
        
        if progress > 0:
            end_angle = -90 + progress * 360
            cv2.ellipse(frame, center, (radius, radius), 0, -90, end_angle, color, thickness, cv2.LINE_AA)
            
        # imageio expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    make_video()
