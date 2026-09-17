import cv2
import numpy as np
import os
import subprocess
import shutil

def main():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # Extract the top-left cell which is empty to create a clean background
    # The grid cells are 128x128 pixels based on our analysis.
    cell = img[0:128, 0:128]
    bg = np.tile(cell, (8, 8, 1))
    
    # Extract the teal block sprite
    # We found a block at (y=138, x=266) with size 108x108
    block = img[138:138+108, 266:266+108]
    
    # Original positions of the top-left corner of the four blocks (y, x)
    positions = [(138, 266), (266, 650), (650, 394), (650, 650)]
    
    # Number of frames for 2.19 seconds at 16 fps is ~35 frames.
    num_frames = 35
    total_shift = 256 # 2 grid cells (2 * 128 pixels)
    
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    for i in range(num_frames):
        # Start with the clean background
        frame = bg.copy()
        
        # Calculate current shift (0 to 256 pixels)
        shift = int(round(total_shift * i / (num_frames - 1)))
        
        for y, x in positions:
            cur_x = x - shift
            # Paste the block at the updated x position
            frame[y:y+108, cur_x:cur_x+108] = block
            
        # Write frame
        cv2.imwrite(os.path.join(frames_dir, f'frame_{i:04d}.png'), frame)
        
    os.makedirs('/app/output', exist_ok=True)
    
    # Run ffmpeg to encode video
    # H.264, yuv420p, 1024x1024, 16 fps
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    # Clean up frames
    shutil.rmtree(frames_dir)

if __name__ == '__main__':
    main()
