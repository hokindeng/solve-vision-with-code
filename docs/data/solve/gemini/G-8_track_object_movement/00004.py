import cv2
import numpy as np
import os
import subprocess

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Identify moving object mask
    non_white = np.any(img != [255, 255, 255], axis=-1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_white, connectivity=8)
    
    moving_mask = np.zeros_like(non_white, dtype=bool)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if x >= 252 and y >= 382 and x+w <= 450 and y+h <= 610:
            moving_mask[labels == i] = True
            
    # The background to fill when object moves
    # Since background is pure white, we can just use white.
    bg_color = np.array([255, 255, 255], dtype=np.uint8)
    
    # Create base image (image without the moving object)
    base_img = img.copy()
    base_img[moving_mask] = bg_color
    
    # Extract the moving object pixels
    # We will just iterate and composite
    
    # Target translation: dx = 220, dy = 0
    total_frames = 60
    
    os.makedirs('/app/frames', exist_ok=True)
    
    for i in range(total_frames):
        frame = base_img.copy()
        
        # Calculate current displacement
        dx = int(round(220 * i / (total_frames - 1)))
        dy = 0
        
        # Find where moving_mask is true
        y_idx, x_idx = np.where(moving_mask)
        
        # New coordinates
        new_y = y_idx + dy
        new_x = x_idx + dx
        
        # Copy pixels from original image to new positions in frame
        frame[new_y, new_x] = img[y_idx, x_idx]
        
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)
        
    # Create video using ffmpeg
    os.makedirs('/app/output', exist_ok=True)
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(ffmpeg_cmd, check=True)

if __name__ == '__main__':
    generate_video()
