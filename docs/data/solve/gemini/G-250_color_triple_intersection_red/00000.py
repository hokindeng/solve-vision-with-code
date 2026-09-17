import cv2
import numpy as np
import imageio
import os

def main():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    
    # Identify unique colors excluding white
    pixels = img.reshape(-1, 3)
    packed = pixels[:, 0].astype(np.uint32) + (pixels[:, 1].astype(np.uint32) << 8) + (pixels[:, 2].astype(np.uint32) << 16)
    unique_packed = np.unique(packed)
    
    circles = []
    for p in unique_packed:
        b = p & 255
        g = (p >> 8) & 255
        r_color = (p >> 16) & 255
        
        # Skip background (white)
        if b == 255 and g == 255 and r_color == 255:
            continue
            
        color = [b, g, r_color]
        mask = cv2.inRange(img, np.array(color), np.array(color))
        
        y, x = np.where(mask > 0)
        if len(x) < 100:
            continue
            
        cx = (np.min(x) + np.max(x)) / 2.0
        cy = (np.min(y) + np.max(y)) / 2.0
        rx = (np.max(x) - np.min(x)) / 2.0
        ry = (np.max(y) - np.min(y)) / 2.0
        radius = (rx + ry) / 2.0
        
        circles.append((cx, cy, radius))
        
    circles = circles[:3]
    
    # Mask of perfectly white pixels
    white_mask = np.all(img == [255, 255, 255], axis=-1).astype(np.uint8)
    
    # Find connected components of white pixels
    num_labels, labels = cv2.connectedComponents(white_mask, connectivity=4)
    
    h, w = white_mask.shape
    Y, X = np.ogrid[:h, :w]
    inside_all = np.ones_like(white_mask, dtype=bool)
    
    for cx, cy, r in circles:
        dist_sq = (X - cx)**2 + (Y - cy)**2
        inside_all &= (dist_sq < (r - 5)**2)
        
    intersection_pixels = inside_all & (white_mask > 0)
    unique_labels = np.unique(labels[intersection_pixels])
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) == 0:
        raise ValueError("Could not find the triple intersection region.")
        
    target_label = unique_labels[0]
    target_mask = (labels == target_label)
    
    # Create the output directory if it doesn't exist
    os.makedirs('/app/output', exist_ok=True)
    
    # Setup video writer
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    num_frames = 60
    
    # Colors (RGB)
    white = np.array([255, 255, 255], dtype=float)
    red = np.array([255, 0, 0], dtype=float)
    
    base_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        current_color = (1 - alpha) * white + alpha * red
        
        frame = base_frame.copy()
        
        # Apply the color to the target region
        frame[target_mask] = current_color.astype(np.uint8)
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == "__main__":
    main()
