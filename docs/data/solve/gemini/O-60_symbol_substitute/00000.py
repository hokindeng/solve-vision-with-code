import cv2
import numpy as np
import os
import subprocess
import shutil

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Position of the solid right triangle at position 2
    y_start, y_end = 478, 478 + 69
    x_start, x_end = 321, 321 + 69
    
    old_crop = img[y_start:y_end, x_start:x_end].astype(np.float32)
    
    # Position of the target symbol in the reference panel
    new_y, new_x = 18 + 25, 887 + 25
    new_crop = img[new_y:new_y+69, new_x:new_x+69].astype(np.float32)
    
    white = np.full_like(old_crop, 255.0)
    
    os.makedirs('/app/temp_frames', exist_ok=True)
    os.makedirs('/app/output', exist_ok=True)
    
    num_frames = 52
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        if t <= 0.5:
            # Fade out old to white
            alpha = 1.0 - (t / 0.5)
            blended = old_crop * alpha + white * (1.0 - alpha)
        else:
            # Fade in new from white
            alpha = (t - 0.5) / 0.5
            blended = new_crop * alpha + white * (1.0 - alpha)
            
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        frame = img.copy()
        frame[y_start:y_end, x_start:x_end] = blended
        
        cv2.imwrite(f'/app/temp_frames/{i:04d}.png', frame)
        
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', '/app/temp_frames/%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clean up temp frames
    shutil.rmtree('/app/temp_frames')

if __name__ == '__main__':
    main()
