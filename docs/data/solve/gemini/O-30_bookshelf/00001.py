import cv2
import numpy as np
import os
import subprocess
import shutil

def create_video():
    img_orig = cv2.imread('/app/first_frame.png')
    
    # Books: (inner_x_start, target_inner_x, top_y, height)
    books = [
        (737, 215, 367, 146), # Book A
        (772, 110, 373, 140), # Book B
        (807, 425, 308, 205), # Book C
    ]
    
    # Extract book images
    book_imgs = []
    for x_start, x_target, top_y, h in books:
        x_left = x_start - 2
        b_img = img_orig[top_y:513, x_left:x_left+30].copy()
        book_imgs.append(b_img)
        
    frames = []
    
    # Frame 0
    frames.append(img_orig.copy())
    
    # Number of frames per book insertion
    N = 7
    
    # We will build frame step by step
    for step in range(3):
        for frame_idx in range(1, N + 1):
            t = frame_idx / N
            img_frame = img_orig.copy()
            
            # Clear all books that have started moving or already moved
            for i in range(step + 1):
                x_start, _, top_y, h = books[i]
                x_left = x_start - 2
                img_frame[top_y:513, x_left:x_left+30] = (255, 255, 255)
            
            # Draw previously moved books at their final targets
            for i in range(step):
                x_start, x_target, top_y, h = books[i]
                x_left_target = x_target - 2
                img_frame[top_y:513, x_left_target:x_left_target+30] = book_imgs[i]
            
            # Draw the currently moving book
            x_start, x_target, top_y, h = books[step]
            x_left_start = x_start - 2
            x_left_target = x_target - 2
            
            # Arc trajectory
            cur_x = int(round((1 - t) * x_left_start + t * x_left_target))
            cur_y = int(round(top_y - 100 * np.sin(np.pi * t)))
            
            # Paste the moving book
            b_img = book_imgs[step]
            bh, bw = b_img.shape[:2]
            
            y1, y2 = cur_y, cur_y + bh
            x1, x2 = cur_x, cur_x + bw
            
            img_frame[y1:y2, x1:x2] = b_img
            
            frames.append(img_frame)
            
    # Save frames to a temp directory
    frames_dir = '/app/frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        cv2.imwrite(f'{frames_dir}/frame_{i:03d}.png', frame)
        
    # Create video using ffmpeg
    os.makedirs('/app/output', exist_ok=True)
    out_file = '/app/output/video.mp4'
    if os.path.exists(out_file):
        os.remove(out_file)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_file
    ]
    subprocess.run(cmd, check=True)
    print(f"Video saved to {out_file}, total frames: {len(frames)}")

if __name__ == '__main__':
    create_video()
