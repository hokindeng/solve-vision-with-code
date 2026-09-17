import cv2
import numpy as np
import os
import subprocess

def generate_video():
    first_frame = cv2.imread('/app/first_frame.png')
    if first_frame is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
        
    h, w, _ = first_frame.shape
    
    # Create the clean background by removing the annotation
    clean_bg = first_frame.copy()
    roi = clean_bg[550:595, 600:780]
    mask = np.any(roi != [255, 255, 255], axis=-1) & np.any(roi != [255, 0, 0], axis=-1)
    roi[mask] = [255, 255, 255]
    clean_bg[550:595, 600:780] = roi

    # The ray reflection parameters
    start_pt = (627, 600)
    # Target point that is way outside, ensuring it goes to the edge
    target_pt = (1108, 49)
    color = (153, 0, 0) # 0.6 * 255 for Blue
    thickness = 2
    
    os.makedirs('/app/output', exist_ok=True)
    
    # We will save frames and then use ffmpeg
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 35
    
    for i in range(num_frames):
        if i == 0:
            frame = first_frame.copy()
        else:
            frame = clean_bg.copy()
            progress = i / (num_frames - 1)
            
            cur_x = int(start_pt[0] + (target_pt[0] - start_pt[0]) * progress)
            cur_y = int(start_pt[1] + (target_pt[1] - start_pt[1]) * progress)
            
            cv2.line(frame, start_pt, (cur_x, cur_y), color, thickness, lineType=cv2.LINE_8)
            
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)
        
    # Create video using ffmpeg
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    generate_video()
