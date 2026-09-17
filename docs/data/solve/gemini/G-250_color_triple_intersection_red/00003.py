import cv2
import numpy as np
import imageio
import os

def generate_video():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
    
    # We want to find the triple intersection of the 3 circles.
    # The image has 4 colors: white background and 3 circle line colors.
    colors = [(140, 80, 80), (180, 130, 70), (60, 120, 120)]
    centers = []
    for c in colors:
        mask = cv2.inRange(img, c, c)
        y, x = np.where(mask > 0)
        if len(x) > 0:
            centers.append((int(np.mean(x)), int(np.mean(y))))
            
    # Calculate the centroid of the 3 circles
    if len(centers) == 3:
        cx = sum(x for x, y in centers) // 3
        cy = sum(y for x, y in centers) // 3
    else:
        # Fallback if colors are different, though we verified they are exact.
        cx, cy = 470, 467
    
    # Flood fill to find the triple intersection region
    # The background is white (255, 255, 255).
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    flood_mask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
    flags = 4 | (255 << 8)
    cv2.floodFill(thresh, flood_mask, (cx, cy), 128, flags=flags)
    
    mask_roi = flood_mask[1:-1, 1:-1] == 255
    
    # We want 60 frames, 16 fps
    num_frames = 60
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    target_color = np.array([0, 0, 255]) # Red in BGR
    original_color = np.array([255, 255, 255]) # White
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Calculate the color for this frame
        # alpha from 0 to 1
        alpha = i / (num_frames - 1)
        
        # BGR interpolation
        current_color = np.round((1 - alpha) * original_color + alpha * target_color).astype(np.uint8)
        
        # Apply the color to the region
        frame[mask_roi] = current_color
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
