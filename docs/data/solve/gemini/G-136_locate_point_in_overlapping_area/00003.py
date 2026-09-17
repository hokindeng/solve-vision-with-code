import cv2
import numpy as np
from PIL import Image
import imageio
import os

def solve():
    img_path = '/app/first_frame.png'
    # Load image in RGB
    orig_img = np.array(Image.open(img_path))
    
    # Identify black points
    black_mask = np.all(orig_img == [0, 0, 0], axis=-1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    
    # Identify colors
    # Exclude black and white, we are left with Shape 1, Shape 2, and Overlap.
    # From counts, we know overlap is [148, 148, 188]
    overlap_color = [148, 148, 188]
    
    fully_inside_pts = []
    
    for i in range(1, num_labels):
        pt_mask = (labels == i).astype(np.uint8)
        # Dilate to get the boundary of the point
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(pt_mask, kernel, iterations=1)
        boundary = dilated - pt_mask
        
        boundary_y, boundary_x = np.where(boundary == 1)
        if len(boundary_y) == 0:
            continue
            
        boundary_colors = orig_img[boundary_y, boundary_x]
        
        is_inside = True
        for color in boundary_colors:
            if not np.array_equal(color, overlap_color):
                is_inside = False
                break
                
        if is_inside:
            cx, cy = centroids[i]
            fully_inside_pts.append((int(cx), int(cy)))
            
    # Sort points for deterministic order (top-left to bottom-right)
    fully_inside_pts.sort(key=lambda p: (p[1], p[0]))
    
    frames = []
    total_frames = 37
    
    start_wait = 5
    end_wait = 6
    draw_frames = total_frames - start_wait - end_wait
    frames_per_pt = draw_frames // max(1, len(fully_inside_pts))
    
    # Red circle color
    circle_color = (255, 0, 0)
    thickness = 3
    radius = 20
    
    for f in range(total_frames):
        frame_img = orig_img.copy()
        
        if f >= start_wait:
            active_f = f - start_wait
            
            for pt_idx, pt in enumerate(fully_inside_pts):
                pt_start_f = pt_idx * frames_per_pt
                pt_end_f = pt_start_f + frames_per_pt
                
                if active_f >= pt_end_f:
                    # Fully drawn
                    cv2.circle(frame_img, pt, radius, circle_color, thickness)
                elif active_f > pt_start_f:
                    # Partially drawn
                    progress = (active_f - pt_start_f) / (frames_per_pt - 1)
                    end_angle = int(progress * 360)
                    if end_angle > 0:
                        cv2.ellipse(frame_img, pt, (radius, radius), 0, 0, end_angle, circle_color, thickness)
                        
        frames.append(frame_img)
        
    os.makedirs('/app/output', exist_ok=True)
    
    # Write video
    # using format='FFMPEG' and pixelformat='yuv420p' to ensure compatibility as requested
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
