import cv2
import numpy as np
import os
import subprocess

def solve():
    img_orig = cv2.imread('/app/first_frame.png')
    
    # 1. Identify the annotation pixels
    above = img_orig[:510, :, :]
    mask = ~np.all(above == 255, axis=-1)
    mask[np.all(above == [255, 0, 0], axis=-1)] = False
    mask[np.all(above == [150, 150, 150], axis=-1)] = False
    
    # Create a full image mask
    full_mask = np.zeros(img_orig.shape[:2], dtype=bool)
    full_mask[:510, :] = mask

    # 2. Snell's law
    theta1 = 41.1 * np.pi / 180
    n1 = 1.00
    n2 = 1.627
    theta2 = np.arcsin(n1 * np.sin(theta1) / n2)
    
    start_point = (512, 513)
    dy = 1024 - start_point[1]
    dx = dy * np.tan(theta2)
    end_point = (start_point[0] + dx, 1024)
    
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/app/frames', exist_ok=True)
    
    num_frames = 70
    for i in range(num_frames):
        progress = i / (num_frames - 1)
        
        frame = img_orig.copy()
        
        # Fade out the annotation
        if progress > 0:
            # We fade the pixels towards white
            orig_colors = frame[full_mask]
            # frame[full_mask] = orig_colors * (1 - progress) + 255 * progress
            faded = orig_colors.astype(np.float32) * (1 - progress) + np.array([255, 255, 255], dtype=np.float32) * progress
            frame[full_mask] = np.clip(np.round(faded), 0, 255).astype(np.uint8)
        
        # Draw the refracted ray
        current_x = start_point[0] + progress * dx
        current_y = start_point[1] + progress * dy
        
        if progress > 0:
            cv2.line(frame, start_point, (int(round(current_x)), int(round(current_y))), (0, 0, 255), 4, lineType=cv2.LINE_AA)
            
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    solve()
