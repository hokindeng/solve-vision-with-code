import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
    
    # Find the red border
    # Red is BGR (0, 0, 255)
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([100, 100, 255])
    mask = cv2.inRange(img, lower_red, upper_red)
    coords = cv2.findNonZero(mask)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
    else:
        # Fallback if no red found, but there should be
        x, y, w, h = 311, 446, 133, 133
        
    num_frames = 46
    fps = 16
    
    # We will fade out the region (x, y, w, h)
    # Let's add a small margin just in case
    margin = 2
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)
    
    # Original patch
    patch = img[y1:y2, x1:x2].copy()
    
    # White patch
    white_patch = np.ones_like(patch) * 255
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    fade_start = 5
    fade_end = 40
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i < fade_start:
            alpha = 1.0
        elif i > fade_end:
            alpha = 0.0
        else:
            alpha = 1.0 - (i - fade_start) / (fade_end - fade_start)
            
        blended_patch = cv2.addWeighted(patch, alpha, white_patch, 1 - alpha, 0)
        frame[y1:y2, x1:x2] = blended_patch
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
