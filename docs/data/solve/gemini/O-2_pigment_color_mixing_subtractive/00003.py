import numpy as np
from PIL import Image
import os
import subprocess

def create_video():
    img_path = '/app/first_frame.png'
    first_frame = np.array(Image.open(img_path).convert('RGB'))
    
    # Left pigment: RGB(120, 25, 0)
    # Right pigment: RGB(149, 163, 214)
    # Subtractive mix: round(c1 * c2 / 255)
    r = round(120 * 149 / 255)
    g = round(25 * 163 / 255)
    b = round(0 * 214 / 255)
    mixed_color = np.array([r, g, b], dtype=np.uint8)
    
    fps = 16
    num_frames = 44
    
    y_min, y_max = 396, 629
    x_min, x_max = 395, 630
    
    frames_dir = '/app/output/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    for i in range(num_frames):
        frame = first_frame.copy()
        
        # Fade in the mixed color over time
        p = i / (num_frames - 1)
        
        interior = frame[y_min:y_max, x_min:x_max].astype(float)
        mixed_float = mixed_color.astype(float)
        
        blended = (1 - p) * interior + p * mixed_float
        frame[y_min:y_max, x_min:x_max] = np.round(blended).astype(np.uint8)
        
        frame_img = Image.fromarray(frame)
        frame_img.save(f'{frames_dir}/frame_{i:03d}.png')
        
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', f'{frames_dir}/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    create_video()
