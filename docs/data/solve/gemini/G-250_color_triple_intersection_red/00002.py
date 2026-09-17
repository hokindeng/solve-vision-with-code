import cv2
import numpy as np
import imageio
import os

def make_video():
    # Read image
    img = cv2.imread('/app/first_frame.png')
    
    # We will identify the circles by their colors
    # Find unique colors that are not white
    pixels = img.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    circle_colors = []
    for c, count in zip(unique_colors, counts):
        if count < 1000000: # exclude background which is > 1M
            circle_colors.append(c)
            
    circles = []
    for c in circle_colors:
        mask = cv2.inRange(img, c, c)
        y, x = np.where(mask > 0)
        cx, cy = np.mean(x), np.mean(y)
        r = np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))
        circles.append((cx, cy, r))

    # Construct the mathematical intersection
    y, x = np.ogrid[:1024, :1024]
    intersection = np.ones((1024, 1024), dtype=bool)
    for cx, cy, r in circles:
        intersection &= ((x - cx)**2 + (y - cy)**2 <= r**2)

    # Find the center of the intersection
    y_int, x_int = np.where(intersection)
    cy_int, cx_int = int(np.mean(y_int)), int(np.mean(x_int))

    # Flood fill from the center to get the exact region to color
    h, w = img.shape[:2]
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    filled = img.copy()
    
    cv2.floodFill(filled, ff_mask, (cx_int, cy_int), (0, 0, 255), (0, 0, 0), (0, 0, 0), 4)

    # region_mask is where floodfill changed the image
    diff = cv2.absdiff(img, filled)
    region_mask = np.any(diff > 0, axis=2)
    
    frames = []
    num_frames = 60
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target_color = np.array([255, 0, 0], dtype=float)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        frame = img_rgb.copy()
        
        frame_region = frame[region_mask].astype(float)
        frame_region = (1 - alpha) * frame_region + alpha * target_color
        frame[region_mask] = frame_region.astype(np.uint8)
        
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    make_video()
