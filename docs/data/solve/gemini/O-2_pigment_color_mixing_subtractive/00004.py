import cv2
import numpy as np
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    h, w, _ = img.shape
    
    # Target color: RGB(16, 26, 33) -> BGR(33, 26, 16)
    target_color = (33, 26, 16)
    
    # Create mask of the inner region
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    img_copy = img.copy()
    cv2.floodFill(img_copy, mask, (512, 512), (0, 0, 0))
    mask = mask[1:-1, 1:-1]
    
    ys, xs = np.where(mask == 1)
    top = ys.min()
    bottom = ys.max()
    
    frames_dir = '/tmp/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 44
    for i in range(num_frames):
        frame = img.copy()
        
        # Calculate how far up we have filled
        # i=0 -> fill 0 (nothing)
        # i=num_frames-1 -> fill to top
        if i == 0:
            current_top = bottom + 1
        else:
            progress = i / (num_frames - 1)
            # When progress is 1, current_top should be top
            current_top = bottom - int(progress * (bottom - top))
            
        # Apply fill
        # We want to change pixels where mask == 1 and y >= current_top
        fill_mask = (mask == 1) & (np.arange(h)[:, None] >= current_top)
        
        frame[fill_mask] = target_color
        
        cv2.imwrite(f"{frames_dir}/frame_{i:04d}.png", frame)
        
    os.makedirs('/app/output', exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', f"{frames_dir}/frame_%04d.png",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)

if __name__ == '__main__':
    main()
