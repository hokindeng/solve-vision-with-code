import cv2
import numpy as np
import os
import subprocess
import shutil

def create_video():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        print("Error: Could not read /app/first_frame.png")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    bg = img.copy()
    objects = []

    for i in range(1, num_labels):
        area = stats[i][cv2.CC_STAT_AREA]
        if area > 1000:
            mask = (labels == i)
            # Remove object from background (replace with white)
            bg[mask] = [255, 255, 255]
            
            # Determine dx based on area
            if area > 30000:
                dx = 510
            elif area > 20000:
                dx = 275
            elif area > 14000:
                dx = 440
            else:
                dx = 601
                
            objects.append({
                'mask': mask,
                'dx': dx,
                'pixels': img[mask]
            })

    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 30
    for f in range(num_frames):
        t = f / (num_frames - 1)
        frame = bg.copy()
        
        for obj in objects:
            shift_x = int(round(obj['dx'] * t))
            y, x = np.where(obj['mask'])
            new_x = x + shift_x
            
            frame[y, new_x] = obj['pixels']
            
        cv2.imwrite(os.path.join(frames_dir, f'frame_{f:04d}.png'), frame)
        
    # generate video using ffmpeg
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    
    # cleanup frames directory
    shutil.rmtree(frames_dir)

if __name__ == '__main__':
    create_video()
