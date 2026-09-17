import cv2
import numpy as np
import os
import subprocess

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    # Identify background color (most frequent)
    unique_colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    bg_color = unique_colors[np.argmax(counts)]
    
    # Identify black color
    black_color = np.array([0, 0, 0])
    
    # Mask of all shape pixels (not background, not black)
    is_bg = np.all(img == bg_color, axis=-1)
    is_black = np.all(img == black_color, axis=-1)
    shape_mask = (~is_bg & ~is_black).astype(np.uint8) * 255
    
    # Find connected components of shapes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(shape_mask, connectivity=8)
    
    shapes_to_outline = []
    
    for i in range(1, num_labels):
        comp_mask = (labels == i).astype(np.uint8) * 255
        
        # Check if it has a black border
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(comp_mask, kernel)
        border_region = (dilated > 0) & is_black
        
        if np.sum(border_region) < 20: 
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                shapes_to_outline.append((comp_mask, cnt))

    frames = 80
    temp_dir = '/app/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    for i in range(frames):
        frame = img.copy()
        
        for comp_mask, cnt in shapes_to_outline:
            k = int(len(cnt) * i / (frames - 1))
            if k > 0:
                is_closed = (i == frames - 1)
                pts = cnt[:k]
                cv2.polylines(frame, [pts], is_closed, (0, 0, 0), thickness=11)
                
            # Restore the original shape inside
            frame[comp_mask > 0] = img[comp_mask > 0]
            
        cv2.imwrite(f'{temp_dir}/frame_{i:03d}.png', frame)

    cmd = [
        'ffmpeg', '-y', 
        '-framerate', '16', 
        '-i', f'{temp_dir}/frame_%03d.png',
        '-c:v', 'libx264', 
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
