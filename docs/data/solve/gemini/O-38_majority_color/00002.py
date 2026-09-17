import numpy as np
from PIL import Image
import cv2
import imageio
import os

def solve():
    img_path = '/app/first_frame.png'
    img = np.array(Image.open(img_path).convert('RGB'))
    
    # 1. Identify background color
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    bg_color_idx = np.argmax(counts)
    bg_color = colors[bg_color_idx]
    
    # We assume black is the outline color. If not present, this safely handles it.
    outline_color = np.array([0, 0, 0])
    
    # 2. Count objects of each color
    object_colors = []
    for c in colors:
        if np.array_equal(c, bg_color) or np.array_equal(c, outline_color):
            continue
            
        # Connected components for this color
        mask = np.all(img == c, axis=-1).astype(np.uint8) * 255
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        
        # num_labels includes background (0), so num_objects is num_labels - 1
        num_objects = num_labels - 1
        if num_objects > 0:
            object_colors.append((tuple(c), num_objects))
            
    # Print color counts as required by prompt
    for c_tup, count in object_colors:
        print(f"Color {c_tup}: {count} objects")
    
    # 3. Identify majority color
    majority_color_tup, max_count = max(object_colors, key=lambda x: x[1])
    majority_color = np.array(majority_color_tup)
    print(f"Majority color is {majority_color_tup} with {max_count} objects")
    
    # 4. Create vanish mask
    mask_majority = np.all(img == majority_color, axis=-1).astype(np.uint8) * 255
    mask_outline = np.all(img == outline_color, axis=-1).astype(np.uint8) * 255
    mask_bg = np.all(img == bg_color, axis=-1).astype(np.uint8) * 255
    
    mask_other_colors = cv2.bitwise_not(mask_majority | mask_outline | mask_bg)
    
    dist_majority = cv2.distanceTransform(cv2.bitwise_not(mask_majority), cv2.DIST_L2, 3)
    dist_other = cv2.distanceTransform(cv2.bitwise_not(mask_other_colors), cv2.DIST_L2, 3)
    
    # Outline pixels closer to 'other colors' than to 'majority color' belong to other objects
    vanish_outline = np.logical_and(mask_outline > 0, dist_other < dist_majority)
    total_vanish = np.logical_or(mask_other_colors > 0, vanish_outline)
    
    # 5. Generate animation
    num_frames = 40
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    bg_pixel = bg_color.astype(np.float32)
    
    for t in range(num_frames):
        alpha = 1.0 - (t / (num_frames - 1))  # 1.0 at t=0, 0.0 at t=39
        
        frame = img.copy().astype(np.float32)
        
        # Blend the vanish pixels with background
        for c_idx in range(3):
            frame_c = frame[:, :, c_idx]
            frame_c[total_vanish] = frame_c[total_vanish] * alpha + bg_pixel[c_idx] * (1 - alpha)
            frame[:, :, c_idx] = frame_c
            
        frame = np.clip(np.round(frame), 0, 255).astype(np.uint8)
        writer.append_data(frame)
        
    writer.close()
    print("Video successfully generated at /app/output/video.mp4")

if __name__ == '__main__':
    solve()
