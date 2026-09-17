import cv2
import numpy as np
import os
import subprocess
from collections import Counter

def main():
    # 1. Load image
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # 2. Find background color
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # 3. Find connected components
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 4. Find fill color for each component
    fill_colors = []
    for i in range(1, num_labels):
        obj_mask = (labels == i)
        c, cnt = np.unique(img[obj_mask], axis=0, return_counts=True)
        main_color = tuple(c[np.argmax(cnt)])
        fill_colors.append((i, main_color))
        
    # 5. Find majority color
    color_counts = Counter([color for i, color in fill_colors])
    majority_color = max(color_counts, key=color_counts.get)
    
    # 7. Identify vanishing labels
    vanish_labels = [i for i, color in fill_colors if color != majority_color]
    vanish_mask = np.isin(labels, vanish_labels)
    
    # Generate frames
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/tmp/frames', exist_ok=True)
    
    num_frames = 40
    for frame_idx in range(num_frames):
        alpha = frame_idx / (num_frames - 1)
        new_frame = img.copy().astype(float)
        
        # Fade vanishing objects to background color
        new_frame[vanish_mask] = img[vanish_mask] * (1 - alpha) + bg_color * alpha
        
        new_frame = np.clip(new_frame, 0, 255).astype(np.uint8)
        cv2.imwrite(f'/tmp/frames/frame_{frame_idx:03d}.png', new_frame)
        
    # Run ffmpeg
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/tmp/frames/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
