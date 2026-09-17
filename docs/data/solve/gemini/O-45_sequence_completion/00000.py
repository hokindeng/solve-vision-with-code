import cv2
import numpy as np
import imageio
import os

def solve():
    # Load first frame
    start_img = cv2.imread('/app/first_frame.png')
    # cv2 reads in BGR, convert to RGB for imageio
    start_img_rgb = cv2.cvtColor(start_img, cv2.COLOR_BGR2RGB)
    
    # Create the end frame
    end_img_rgb = start_img_rgb.copy()
    
    # The pattern is a sequence of 4 colors: Orange, Navy, Cyan, Purple.
    # Elements: 1: Orange, 2: Navy, 3: Cyan, 4: Purple, 5: Orange.
    # Therefore, the missing 6th element is Navy (the 2nd element).
    # The sequence centers are vertically at y=512.
    # x coordinates: 152, 296, 440, 583, 727, 871.5
    # The 2nd element is at x=296. The missing element (?) is at x=871.
    # We copy a 150x150 patch around x=296 to x=871 to seamlessly replace it.
    patch = start_img_rgb[512-75:512+75, 296-75:296+75]
    end_img_rgb[512-75:512+75, 871-75:871+75] = patch
    
    os.makedirs('/app/output', exist_ok=True)
    
    # Write the video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    num_frames = 25
    for i in range(num_frames):
        t = i / (num_frames - 1)
        # Linear crossfade for pacing
        frame = (start_img_rgb.astype(np.float32) * (1 - t) + end_img_rgb.astype(np.float32) * t)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
