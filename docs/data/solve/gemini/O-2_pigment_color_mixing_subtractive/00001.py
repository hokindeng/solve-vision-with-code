import numpy as np
from PIL import Image
import imageio
import os

def solve():
    # Load first frame
    first_frame = Image.open('/app/first_frame.png').convert('RGB')
    arr = np.array(first_frame)
    
    # Define colors
    c1 = np.array([136, 158, 164], dtype=np.float32)
    c2 = np.array([68, 180, 242], dtype=np.float32)
    
    # Subtractive mix: per-channel product / 255, rounded
    mixed = np.round(c1 * c2 / 255.0).astype(np.uint8)
    
    # Target inner white box bounds to fill
    x_min, x_max = 395, 629  # inclusive
    y_min, y_max = 396, 628  # inclusive
    
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', 
                                fps=16, 
                                codec='libx264', 
                                ffmpeg_params=['-pix_fmt', 'yuv420p', '-profile:v', 'main', '-crf', '18'])
    
    num_frames = 44
    
    for i in range(num_frames):
        frame = arr.copy()
        
        # We fill from bottom to top over the duration
        filled_ratio = i / (num_frames - 1)
        
        height = y_max - y_min + 1
        fill_height = int(round(filled_ratio * height))
        
        if fill_height > 0:
            fill_y_start = y_max - fill_height + 1
            frame[fill_y_start:y_max+1, x_min:x_max+1] = mixed
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
