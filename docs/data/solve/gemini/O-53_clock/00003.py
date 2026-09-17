import cv2
import numpy as np
from PIL import Image, ImageDraw
import math
import os
import subprocess
import shutil

def create_video():
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not find /app/first_frame.png")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create clean background by erasing hands
    clean = img.copy()
    hour_mask = np.all(img == [0, 100, 0], axis=-1)
    min_mask = np.all(img == [220, 20, 60], axis=-1)
    clean[hour_mask] = [255, 248, 240]
    clean[min_mask] = [255, 248, 240]
    
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs('/app/output', exist_ok=True)
    
    num_frames = 120
    total_hours = 21.0
    start_time_hours = 6.0 + 5.0 / 60.0  # 6:05
    
    for i in range(num_frames):
        t = i * (total_hours / (num_frames - 1))
        current_time_hours = start_time_hours + t
        
        # Calculate angles in degrees
        hour_angle_deg = (current_time_hours / 12.0) * 360.0 - 90.0
        min_angle_deg = current_time_hours * 360.0 - 90.0
        
        # Convert to radians
        hour_angle = hour_angle_deg * math.pi / 180.0
        min_angle = min_angle_deg * math.pi / 180.0
        
        # Draw frame
        pil_img = Image.fromarray(clean.copy())
        draw = ImageDraw.Draw(pil_img)
        
        # Draw Hour hand
        end_xh = 512 + 205 * math.cos(hour_angle)
        end_yh = 512 + 205 * math.sin(hour_angle)
        draw.line([(512, 512), (end_xh, end_yh)], fill=(0, 100, 0), width=8)
        
        # Draw Minute hand
        end_xm = 512 + 286 * math.cos(min_angle)
        end_ym = 512 + 286 * math.sin(min_angle)
        draw.line([(512, 512), (end_xm, end_ym)], fill=(220, 20, 60), width=5)
        
        # Draw Center dot (always on top)
        draw.ellipse([(512-10, 512-10), (512+10, 512+10)], fill=(0, 0, 0))
        
        # Save frame to disk
        frame = np.array(pil_img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frames_dir, f'frame_{i:03d}.png'), frame_bgr)
        
    # Encode with ffmpeg to match exact required specifications
    # H.264, yuv420p, 1024x1024, 16 fps
    output_path = '/app/output/video.mp4'
    cmd = [
        'ffmpeg', '-y', '-framerate', '16',
        '-i', os.path.join(frames_dir, 'frame_%03d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup frames directory to keep things tidy
    shutil.rmtree(frames_dir)
    print("Video generation complete.")

if __name__ == '__main__':
    create_video()
