import cv2
import numpy as np
import imageio
import os

def main():
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Dynamically find grid edges
    grid_color = np.array([51, 51, 51])
    grid_pixels = np.all(img_rgb == grid_color, axis=-1)
    
    row_sums = np.sum(grid_pixels, axis=1)
    edges = []
    for i, s in enumerate(row_sums):
        if s > 20:  # Grid lines span the entire width
            edges.append(i)
    if edges[-1] != img.shape[0]:
        edges.append(img.shape[0])
        
    col_sums = np.sum(grid_pixels, axis=0)
    col_edges = []
    for i, s in enumerate(col_sums):
        if s > 20:
            col_edges.append(i)
    if col_edges[-1] != img.shape[1]:
        col_edges.append(img.shape[1])

    clean = img_rgb.copy()
    blocks = []

    for r in range(len(edges)-1):
        for c in range(len(col_edges)-1):
            r1, r2 = edges[r], edges[r+1]
            c1, c2 = col_edges[c], col_edges[c+1]
            
            if r2 <= r1 or c2 <= c1:
                continue
                
            cell = img_rgb[r1:r2, c1:c2]
            
            # Robust mask: pixel is not white and not grid line
            not_white = np.any(cell != [255, 255, 255], axis=-1)
            not_grid = np.any(cell != [51, 51, 51], axis=-1)
            mask = np.logical_and(not_white, not_grid)
            
            if np.any(mask):
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                block_img = cell[rmin:rmax+1, cmin:cmax+1].copy()
                
                blocks.append({
                    'r': r, 'c': c, 
                    'rmin': rmin, 'rmax': rmax,
                    'cmin': cmin, 'cmax': cmax,
                    'img': block_img
                })
                
                # Replace the block with white background
                clean[r1+rmin:r1+rmax+1, c1+cmin:c1+cmax+1] = [255, 255, 255]

    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', macro_block_size=None, quality=10, pixelformat='yuv420p')

    num_frames = 35

    for f in range(num_frames):
        frame = clean.copy()
        t = f / (num_frames - 1)
        
        for b in blocks:
            r_start = b['r']
            c_start = b['c']
            
            r_curr = r_start + 3 * t
            c_curr = c_start
            
            # Interpolate to find exact pixel location of the cell edges
            r_edge = np.interp(r_curr, range(len(edges)), edges)
            c_edge = np.interp(c_curr, range(len(col_edges)), col_edges)
            
            y_pixel = int(round(r_edge)) + b['rmin']
            x_pixel = int(round(c_edge)) + b['cmin']
            
            # Paste the block
            b_img = b['img']
            bh, bw = b_img.shape[:2]
            frame[y_pixel:y_pixel+bh, x_pixel:x_pixel+bw] = b_img
            
        writer.append_data(frame)

    writer.close()

if __name__ == '__main__':
    main()
