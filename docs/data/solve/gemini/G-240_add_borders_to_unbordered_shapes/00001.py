import cv2
import numpy as np
import imageio
import os

def solve():
    img_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read input image")

    # Dynamically find the background color (most frequent color)
    pixels = img.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Mask is 255 for non-background pixels
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8) * 255
    
    # Find outer contours of all shapes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    shapes_to_border = []
    for c in contours:
        c_flat = c.squeeze(1)
        c_colors = img[c_flat[:, 1], c_flat[:, 0]]
        # Check if the boundary is mostly black (black border test)
        # We consider a pixel dark if all RGB channels are < 50
        is_dark = (c_colors < 50).all(axis=-1)
        if np.mean(is_dark) < 0.5:
            shapes_to_border.append(c_flat)
            
    num_frames = 80
    frames = []
    
    for i in range(num_frames):
        fraction = i / (num_frames - 1)
        frame = img.copy()
        
        for c in shapes_to_border:
            n_points = int(len(c) * fraction)
            if n_points >= 2:
                pts = c[:n_points]
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 0), thickness=4, lineType=cv2.LINE_8)
            # Ensure the border is completely closed in the final frame
            if i == num_frames - 1:
                cv2.polylines(frame, [c], isClosed=True, color=(0, 0, 0), thickness=4, lineType=cv2.LINE_8)
                
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    imageio.mimwrite(
        output_path, 
        frames, 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p', 
        macro_block_size=None
    )

if __name__ == '__main__':
    solve()
