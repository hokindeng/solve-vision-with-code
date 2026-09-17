import cv2
import numpy as np
import imageio
import os

def solve():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    # Use RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Coordinates
    src_y = 474
    src_x = 402
    h = 69
    w = 72
    
    dst_y = 474
    dst_x = 847
    
    roi_src = img_rgb[src_y:src_y+h, src_x:src_x+w]
    # Pink is [236, 72, 153] in RGB (BGR was 153, 72, 236)
    mask = np.all(roi_src == [236, 72, 153], axis=-1).astype(np.uint8)
    
    # Get contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_pts = contours[0].reshape(-1, 2)
    num_pts = len(contour_pts)
    
    fps = 16
    total_frames = 60
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for frame in range(total_frames):
        current_frame = img_rgb.copy()
        
        # Frames 0-4: show original (empty box)
        if frame < 5:
            pass
            
        # Frames 5-24: draw contour
        elif frame < 25:
            progress = (frame - 4) / 20.0
            pts_to_draw = int(num_pts * progress)
            for i in range(pts_to_draw):
                x, y = contour_pts[i]
                current_frame[dst_y + y, dst_x + x] = roi_src[y, x]
                
        # Frames 25-49: fill top to bottom + full contour
        elif frame < 50:
            # Draw full contour
            for i in range(num_pts):
                x, y = contour_pts[i]
                current_frame[dst_y + y, dst_x + x] = roi_src[y, x]
                
            progress = (frame - 24) / 25.0
            rows_to_fill = int(h * progress)
            
            for y in range(rows_to_fill):
                for x in range(w):
                    if mask[y, x]:
                        current_frame[dst_y + y, dst_x + x] = roi_src[y, x]
                        
        # Frames 50-59: show full filled shape
        else:
            for y in range(h):
                for x in range(w):
                    if mask[y, x]:
                        current_frame[dst_y + y, dst_x + x] = roi_src[y, x]
                        
        writer.append_data(current_frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
