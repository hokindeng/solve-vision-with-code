import cv2
import numpy as np
import imageio
import os

def solve():
    img_bgr = cv2.imread('/app/first_frame.png')
    if img_bgr is None:
        print("Could not read image.")
        return
        
    # 1. Dynamically find the background color (most frequent color)
    pixels = img_bgr.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Create mask of background-colored pixels (regions we can color)
    bg_mask = np.all(img_bgr == bg_color, axis=-1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bg_mask, connectivity=4)
    
    # 2. Find the circles in the image
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Use robust parameters for HoughCircles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=50, 
                               param1=50, param2=30, minRadius=10, maxRadius=1000)
                               
    good_circles = []
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = c
            # Mask for circle perimeter with thickness 3 for slight tolerance
            mask = np.zeros_like(edges)
            cv2.circle(mask, (int(cx), int(cy)), int(r), 255, 3)
            
            overlap = cv2.bitwise_and(edges, mask)
            overlap_count = np.count_nonzero(overlap)
            perimeter = 2 * np.pi * r
            coverage = overlap_count / perimeter
            
            if coverage > 0.4:
                good_circles.append((cx, cy, r, coverage))
                
    # Sort by coverage and take top 3
    good_circles = sorted(good_circles, key=lambda x: x[3], reverse=True)[:3]
    
    if len(good_circles) < 3:
        print(f"Warning: Found only {len(good_circles)} circles, expected at least 3.")
    
    # 3. Find the triple intersection
    target_mask = np.zeros_like(bg_mask, dtype=bool)
    
    for i in range(1, num_labels):
        y, x = np.where(labels == i)
        if len(x) == 0:
            continue
            
        inside_count = 0
        for gc in good_circles:
            cx, cy, r = gc[:3]
            distances = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Check if region is inside this circle (>90% of its pixels)
            # r + 5 allows for anti-aliasing / edge thickness
            fraction_inside = np.mean(distances <= r + 5)
            if fraction_inside > 0.9:
                inside_count += 1
                
        # If inside all 3 circles, mark it
        if inside_count == 3:
            target_mask[labels == i] = True
            
    # 4. Generate video frames
    total_frames = 60
    fps = 16
    os.makedirs('/app/output', exist_ok=True)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    target_color = np.array([255, 0, 0], dtype=np.float32) # Red in RGB
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for i in range(total_frames):
        alpha = i / (total_frames - 1)
        frame = img_rgb.copy().astype(np.float32)
        
        # Alpha blend the target region
        orig_colors = frame[target_mask]
        new_colors = (1 - alpha) * orig_colors + alpha * target_color
        frame[target_mask] = new_colors
        
        writer.append_data(frame.astype(np.uint8))
        
    writer.close()

if __name__ == '__main__':
    solve()
