import cv2
import numpy as np
import subprocess
import os

def main():
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Create clean background
    clean_bg = img.copy()
    mask1 = (clean_bg[:,:,0] == 0) & (clean_bg[:,:,1] == 165) & (clean_bg[:,:,2] == 255)
    mask2 = (clean_bg[:,:,0] == 0) & (clean_bg[:,:,1] == 120) & (clean_bg[:,:,2] == 200)
    clean_bg[mask1] = [50, 200, 50]
    clean_bg[mask2] = [50, 200, 50]
    
    # 2. Extract agent pixels
    cx, cy = 93, 744
    agent_pixels = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if mask1[y, x] or mask2[y, x]:
                agent_pixels.append((x - cx, y - cy, img[y, x].tolist()))
                
    # 3. Path
    path_rc = [
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (6, 4),
        (5, 4),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
        (4, 9),
        (3, 9),
        (3, 8),
        (3, 7),
        (3, 6),
        (3, 5),
        (3, 4),
        (3, 3),
        (3, 2),
        (3, 1)
    ]
    
    # 4. Generate frames
    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 94
    num_segments = len(path_rc) - 1
    
    for f in range(num_frames):
        d = f * num_segments / (num_frames - 1)
        segment = int(d)
        if segment >= num_segments:
            segment = num_segments - 1
            progress = 1.0
        else:
            progress = d - segment
            
        r1, c1 = path_rc[segment]
        r2, c2 = path_rc[segment + 1]
        
        cur_r = r1 + (r2 - r1) * progress
        cur_c = c1 + (c2 - c1) * progress
        
        cur_cx = int(round(93.0 + cur_c * 93.2))
        cur_cy = int(round(91.6 + cur_r * 93.2))
        
        frame = clean_bg.copy()
        
        # Draw agent
        for dx, dy, color in agent_pixels:
            x = cur_cx + dx
            y = cur_cy + dy
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                frame[y, x] = color
                
        cv2.imwrite(f'{frames_dir}/frame_{f:04d}.png', frame)
        
    # 5. Encode video
    # H.264, yuv420p, 1024x1024, 16 fps, about 94 frames (5.88 s).
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=1024:1024',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
