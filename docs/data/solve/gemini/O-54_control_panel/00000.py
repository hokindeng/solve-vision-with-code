import cv2
import numpy as np
import os
import subprocess

def create_video():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    dot_pattern = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ])
    
    def draw_left_box(lever_x):
        box = np.zeros((114, 232, 3), dtype=np.uint8)
        dot_positions = [(31, 54), (113, 54), (195, 54)]
        for dx, dy in dot_positions:
            for y in range(5):
                for x in range(5):
                    if dot_pattern[y, x] == 1:
                        box[dy+y, dx+x] = [240, 240, 240]
        box[28:28+57, lever_x:lever_x+57] = [128, 128, 128]
        return box
    
    purple_mask = (img[:,:,0] == 128) & (img[:,:,1] == 0) & (img[:,:,2] == 128)
    
    c_start = np.array([128, 0, 128], dtype=float)
    c_end = np.array([203, 192, 255], dtype=float)
    
    start_x = 87
    end_x = 164
    
    num_frames = 24
    fps = 16
    
    writer = cv2.VideoWriter('/app/output/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1024, 1024))
    
    for i in range(num_frames):
        # smoothstep
        raw_t = i / (num_frames - 1)
        t = raw_t * raw_t * (3 - 2 * raw_t)
        
        frame = img.copy()
        
        # update lever position
        lever_x = int(round(start_x + t * (end_x - start_x)))
        new_box = draw_left_box(lever_x)
        frame[620:734, 89:321] = new_box
        
        # update light color
        c_curr = c_start + t * (c_end - c_start)
        frame[purple_mask] = c_curr.astype(np.uint8)
        
        writer.write(frame)
        
    writer.release()
    
    # ensure it's specifically h.264 yuv420p
    subprocess.run([
        'ffmpeg', '-y', '-i', '/app/output/video.mp4', 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/final_video.mp4'
    ], check=True)
    os.replace('/app/output/final_video.mp4', '/app/output/video.mp4')

if __name__ == '__main__':
    create_video()
