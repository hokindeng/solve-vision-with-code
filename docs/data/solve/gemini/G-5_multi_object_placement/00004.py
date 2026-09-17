import cv2
import numpy as np
import imageio
from PIL import Image
import os

def solve():
    img_path = '/app/first_frame.png'
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    
    # We will build the static background (white + stars)
    background = np.ones_like(arr) * 255
    
    # Store movers: list of dicts with 'color', 'mask', 'start_center', 'end_center'
    movers = []
    
    # The image contains three colors: white background, and two object colors.
    colors = [[100, 100, 255], [255, 100, 255]]
    
    for c in colors:
        c_arr = np.array(c, dtype=np.uint8)
        mask = np.all(arr == c_arr, axis=-1).astype(np.uint8) * 255
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        star_centroid = None
        mover_info = {}
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1000:
                # It's a star marker, add to static background
                star_mask = (labels == i)
                background[star_mask] = c_arr
                star_centroid = centroids[i]
            else:
                # It's a mover
                mover_info['color'] = c_arr
                mover_info['mask'] = (labels == i)
                mover_info['start_center'] = centroids[i]
        
        mover_info['end_center'] = star_centroid
        movers.append(mover_info)

    # Now generate the frames
    num_frames = 48
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    # Using H.264, yuv420p, 16 fps as requested
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', macro_block_size=None, format='FFMPEG', pixelformat='yuv420p')
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        frame = background.copy()
        
        for mover in movers:
            start_x, start_y = mover['start_center']
            end_x, end_y = mover['end_center']
            
            curr_x = start_x + (end_x - start_x) * t
            curr_y = start_y + (end_y - start_y) * t
            
            # Displacement from original mask position
            dx = int(round(curr_x - start_x))
            dy = int(round(curr_y - start_y))
            
            y_coords, x_coords = np.where(mover['mask'])
            
            new_y = y_coords + dy
            new_x = x_coords + dx
            
            valid = (new_y >= 0) & (new_y < arr.shape[0]) & (new_x >= 0) & (new_x < arr.shape[1])
            new_y = new_y[valid]
            new_x = new_x[valid]
            
            frame[new_y, new_x] = mover['color']
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
