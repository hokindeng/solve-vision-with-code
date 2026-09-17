import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import subprocess
import shutil

def main():
    # 1. Read first frame
    img = Image.open('/app/first_frame.png').convert('RGB')
    img_np = np.array(img)
    
    # 2. Extract clean background
    bg = img_np.copy()
    hour_mask = (img_np[:,:,0] == 139) & (img_np[:,:,1] == 69) & (img_np[:,:,2] == 19)
    minute_mask = (img_np[:,:,0] == 147) & (img_np[:,:,1] == 112) & (img_np[:,:,2] == 219)
    bg[hour_mask] = [220, 245, 245]
    bg[minute_mask] = [220, 245, 245]
    
    # 3. Extract black mask
    black_mask = (img_np[:,:,0] == 0) & (img_np[:,:,1] == 0) & (img_np[:,:,2] == 0)
    
    # We will save frames to a temporary directory
    os.makedirs('/tmp/frames', exist_ok=True)
    
    cx, cy = 512, 512
    start_hours = 7 + 44 / 60
    total_frames = 120
    
    for i in range(total_frames):
        if i == 0:
            frame = img_np.copy()
        else:
            elapsed_hours = 17.0 * i / (total_frames - 1)
            current_hours = start_hours + elapsed_hours
            
            # Calculate angles
            hour_angle_deg = (current_hours % 12) / 12 * 360
            minute_angle_deg = (current_hours % 1) * 360
            
            # Copy background
            frame = bg.copy()
            
            # Helper to draw hand
            def draw_hand(base_img, angle_deg, r_inner, length, width, color):
                angle_rad = np.radians(angle_deg)
                x1 = cx + r_inner * np.sin(angle_rad)
                y1 = cy - r_inner * np.cos(angle_rad)
                x2 = cx + (r_inner + length) * np.sin(angle_rad)
                y2 = cy - (r_inner + length) * np.cos(angle_rad)
                
                img_pil = Image.fromarray(base_img)
                draw = ImageDraw.Draw(img_pil)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
                return np.array(img_pil)
                
            # Draw hour hand
            frame = draw_hand(frame, hour_angle_deg, 11.4, 193.0, 8, (139, 69, 19))
            
            # Draw minute hand
            frame = draw_hand(frame, minute_angle_deg, 10.0, 275.6, 5, (147, 112, 219))
            
            # Re-apply black mask (center dot and ticks)
            frame[black_mask] = [0, 0, 0]
        
        # Save frame
        frame_img = Image.fromarray(frame)
        frame_img.save(f'/tmp/frames/frame_{i:04d}.png')
        
    # Generate video with ffmpeg
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    if os.path.exists(out_path):
        os.remove(out_path)
        
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', '/tmp/frames/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    shutil.rmtree('/tmp/frames')

if __name__ == '__main__':
    main()
