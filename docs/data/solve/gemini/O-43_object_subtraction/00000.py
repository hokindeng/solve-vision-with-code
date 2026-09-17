import cv2
import numpy as np
import os
import subprocess
import shutil

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Identify cubes based on colors we found.
    # The cubes are the Red, Green, and Yellow perfect squares.
    target_colors = [[0, 0, 255], [0, 200, 0], [0, 255, 255]]
    
    cubes = []
    for color in target_colors:
        mask = np.all(img == color, axis=-1).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cubes.append({"color": color, "bbox": (x, y, w, h)})
            
    num_frames = 96
    out_dir = '/tmp/frames'
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Scale decreases from 1.0 down to 0.0
        progress = i / (num_frames - 1)
        scale = max(0.0, 1.0 - progress)
        
        for cube in cubes:
            x, y, w, h = cube["bbox"]
            color = cube["color"]
            
            # 1. Erase the original cube with background color (white)
            frame[y:y+h, x:x+w] = [255, 255, 255]
            
            # 2. Draw scaled cube in the center
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            
            if new_w > 0 and new_h > 0:
                cx = x + w / 2.0
                cy = y + h / 2.0
                new_x = int(round(cx - new_w / 2.0))
                new_y = int(round(cy - new_h / 2.0))
                
                frame[new_y:new_y+new_h, new_x:new_x+new_w] = color
                
        cv2.imwrite(f'{out_dir}/frame_{i:04d}.png', frame)
        
    os.makedirs('/app/output', exist_ok=True)
    out_file = '/app/output/video.mp4'
    
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{out_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup frames
    shutil.rmtree(out_dir)

if __name__ == '__main__':
    solve()
