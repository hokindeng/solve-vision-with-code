import cv2
import numpy as np
import imageio
import os

def solve():
    img_orig = cv2.imread('/app/first_frame.png')
    
    # 1. Erase QM to get a clean background
    clean_img = img_orig.copy()
    clean_img[684:1021, 683:1022] = 255
    
    # 2. Get Triangle pixels from Cell(0,1)
    mask = (img_orig[:, :, 1] == 150) & (img_orig[:, :, 0] == 0) & (img_orig[:, :, 2] == 0)
    y_all, x_all = np.where(mask)
    in_cell = (y_all >= 2) & (y_all <= 338) & (x_all >= 342) & (x_all <= 680)
    y_tri = y_all[in_cell]
    x_tri = x_all[in_cell]
    
    # 3. Sort Triangle pixels to animate drawing
    cy = np.mean(y_tri)
    cx = np.mean(x_tri)
    angles = np.arctan2(y_tri - cy, x_tri - cx)
    # Start at top (-pi/2) and go clockwise
    angles_mapped = (angles + np.pi/2) % (2 * np.pi)
    sort_idx = np.argsort(angles_mapped)
    
    y_tri_sorted = y_tri[sort_idx]
    x_tri_sorted = x_tri[sort_idx]
    
    # 4. Generate frames
    frames = []
    
    total_frames = 35 # Frame 0 to 34
    num_draw_frames = 34 # Frame 1 to 34
    
    # Frame 0: original with QM
    frames.append(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    
    for f in range(1, total_frames):
        frame_img = clean_img.copy()
        
        # Calculate how many pixels to draw by this frame
        ratio = f / num_draw_frames
        num_pixels = int(len(y_tri_sorted) * ratio)
        
        if num_pixels > 0:
            target_y = y_tri_sorted[:num_pixels] + 682
            target_x = x_tri_sorted[:num_pixels] + 341
            
            frame_img[target_y, target_x] = [0, 150, 0]
        
        frames.append(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
    
    # 5. Save video
    out_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
