import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    y0, y1 = 2, 339
    x0, x1 = 342, 681
    
    cell01 = img[y0:y1, x0:x1]
    
    color = np.array([0, 140, 255])
    mask = np.all(cell01 == color, axis=-1)
    
    pts_y, pts_x = np.where(mask)
    
    if len(pts_y) == 0:
        return
        
    cy, cx = np.mean(pts_y), np.mean(pts_x)
    
    angles = np.arctan2(pts_y - cy, pts_x - cx)
    adjusted_angles = (angles + np.pi/2) % (2*np.pi)
    
    sorted_indices = np.argsort(adjusted_angles)
    pts_y_sorted = pts_y[sorted_indices]
    pts_x_sorted = pts_x[sorted_indices]
    
    total_pixels = len(pts_y_sorted)
    
    num_frames = 35
    frames = []
    
    for i in range(num_frames):
        frame = img.copy()
        num_to_draw = int(total_pixels * i / (num_frames - 1))
        
        offset_y = 684
        offset_x = 683
        
        for j in range(num_to_draw):
            py = pts_y_sorted[j]
            px = pts_x_sorted[j]
            frame[offset_y + py, offset_x + px] = color
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', format='FFMPEG', 
                                pixelformat='yuv420p', macro_block_size=None, output_params=['-crf', '10'])
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    solve()
