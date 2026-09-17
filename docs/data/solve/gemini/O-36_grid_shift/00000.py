import cv2
import numpy as np
import imageio
from scipy.ndimage import label
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Define colors
    purple = [128, 0, 128]
    black = [0, 0, 0]
    
    # 1. Find starting blocks
    mask = np.all(img == purple, axis=-1) | np.all(img == black, axis=-1)
    labeled, num_features = label(mask)
    
    grid_lines_ext = [0, 113, 227, 341, 455, 568, 682, 796, 910, 1024]
    
    starting_blocks = []
    for i in range(1, num_features + 1):
        coords = np.where(labeled == i)
        y0 = coords[0].min()
        x0 = coords[1].min()
        c = sum(1 for g in grid_lines_ext[:-1] if g <= x0) - 1
        r = sum(1 for g in grid_lines_ext[:-1] if g <= y0) - 1
        starting_blocks.append((c, r))

    def get_bb(c, r):
        x0 = grid_lines_ext[c] + 9
        x1 = grid_lines_ext[c+1] - 9
        y0 = grid_lines_ext[r] + 9
        if r == 8:
            y1 = grid_lines_ext[r+1] - 10
        else:
            y1 = grid_lines_ext[r+1] - 9
        return x0, y0, x1, y1

    N = 35
    half = (N - 1) // 2
    
    def get_current_bb(c_start, r, f):
        bb_start = get_bb(c_start, r)
        bb_mid = get_bb(c_start + 1, r)
        bb_end = get_bb(c_start + 2, r)
        
        if f <= half:
            p = f / half
            x0 = int(round(bb_start[0] + p * (bb_mid[0] - bb_start[0])))
            x1 = int(round(bb_start[2] + p * (bb_mid[2] - bb_start[2])))
        else:
            p = (f - half) / (N - 1 - half)
            x0 = int(round(bb_mid[0] + p * (bb_end[0] - bb_mid[0])))
            x1 = int(round(bb_mid[2] + p * (bb_end[2] - bb_mid[2])))
            
        return x0, bb_start[1], x1, bb_start[3]

    # Create base background (in RGB, same as BGR since colors are symmetric)
    bg = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    for g in grid_lines_ext[:-1]:
        bg[g, :] = [51, 51, 51]
        bg[:, g] = [51, 51, 51]

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for f in range(N):
        frame = bg.copy()
        for c_start, r in starting_blocks:
            x0, y0, x1, y1 = get_current_bb(c_start, r, f)
            # Draw black outline
            frame[y0:y1+1, x0:x1+1] = [0, 0, 0]
            # Draw purple interior
            frame[y0+2:y1-1, x0+2:x1-1] = [128, 0, 128]
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
