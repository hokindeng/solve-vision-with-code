import cv2
import numpy as np
import os
import subprocess
import shutil

def make_video():
    first_frame_path = '/app/first_frame.png'
    first_frame = cv2.imread(first_frame_path)
    
    # Parameters derived from first frame analysis
    color = (11, 134, 184) # BGR
    cx1_start, cy1_start, r1 = 462, 307, 444
    cx2_start, cy2_start, r2 = 787, 716, 220
    thickness = 8
    
    cx_end, cy_end = 512, 512
    
    num_frames = 40
    fps = 16
    
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/app/temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    for i in range(num_frames):
        if i == 0:
            frame = first_frame.copy()
        else:
            t = i / (num_frames - 1)
            cx1 = int(round(cx1_start + (cx_end - cx1_start) * t))
            cy1 = int(round(cy1_start + (cy_end - cy1_start) * t))
            cx2 = int(round(cx2_start + (cx_end - cx2_start) * t))
            cy2 = int(round(cy2_start + (cy_end - cy2_start) * t))
            
            frame = np.full((1024, 1024, 3), 255, dtype=np.uint8)
            cv2.circle(frame, (cx1, cy1), r1, color, thickness, cv2.LINE_8)
            cv2.circle(frame, (cx2, cy2), r2, color, thickness, cv2.LINE_8)
            
        cv2.imwrite(f'{temp_dir}/frame_{i:04d}.png', frame)
        
    out_path = f'{out_dir}/video.mp4'
    if os.path.exists(out_path):
        os.remove(out_path)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps), '-i', f'{temp_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clean up temp frames
    shutil.rmtree(temp_dir)
    
if __name__ == '__main__':
    make_video()
