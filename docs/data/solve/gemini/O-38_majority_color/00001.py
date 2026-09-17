import cv2
import numpy as np
import imageio
import os

def solve():
    # Ensure output directory exists
    os.makedirs("/app/output", exist_ok=True)
    
    # 1. Load image (using cv2 for processing, so it's BGR)
    img_bgr = cv2.imread("/app/first_frame.png")
    if img_bgr is None:
        raise ValueError("Could not load /app/first_frame.png")
        
    h, w, c = img_bgr.shape
    
    # 2. Segment objects
    # Background is perfectly white [255, 255, 255] in BGR
    # We find all non-white pixels
    non_white_mask = cv2.bitwise_not(cv2.inRange(img_bgr, np.array([255, 255, 255]), np.array([255, 255, 255])))
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_white_mask, connectivity=8)
    
    # Analyze components to find majority color
    # We will map each label to its non-black color
    label_to_color = {}
    color_counts = {}
    
    for i in range(1, num_labels):
        # Pixels for this object
        obj_mask = (labels == i)
        obj_pixels = img_bgr[obj_mask]
        
        # Unique colors in this object
        unq_colors = np.unique(obj_pixels, axis=0)
        
        # Find the non-black color
        fill_color = None
        for color in unq_colors:
            if not np.array_equal(color, [0, 0, 0]):
                fill_color = tuple(color)
                break
                
        if fill_color is not None:
            label_to_color[i] = fill_color
            color_counts[fill_color] = color_counts.get(fill_color, 0) + 1
            
    # Find majority color
    majority_color = max(color_counts, key=color_counts.get)
    
    # 3. Create a mask of pixels to vanish (non-majority objects)
    vanish_mask = np.zeros((h, w), dtype=bool)
    for i in range(1, num_labels):
        if i in label_to_color and label_to_color[i] != majority_color:
            vanish_mask[labels == i] = True
            
    # 4. Generate video frames
    num_frames = 40
    fps = 16
    
    frames = []
    
    # The background to fade to is white
    bg_color = np.array([255, 255, 255], dtype=np.float32)
    
    # Convert BGR to RGB for imageio
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    for t in range(num_frames):
        # Alpha goes from 0 (original) to 1 (white)
        alpha = t / (num_frames - 1)
        
        # Copy original image
        frame = img_rgb.copy().astype(np.float32)
        
        # Blend vanishing pixels
        # original_color * (1 - alpha) + white * alpha
        frame[vanish_mask] = frame[vanish_mask] * (1 - alpha) + bg_color * alpha
        
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
        
    # 5. Save video using imageio
    # using H.264, yuv420p pixel format as requested
    writer = imageio.get_writer("/app/output/video.mp4", fps=fps, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == "__main__":
    solve()
